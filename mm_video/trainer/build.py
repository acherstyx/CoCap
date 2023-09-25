# -*- coding: utf-8 -*-
# @Time    : 2022/11/12 22:28
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : build.py

import os
from tqdm import tqdm
import logging
from typing import *

import torch
from torch import nn, optim
from torch.utils import data
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from fvcore.common.config import CfgNode
from fvcore.common.registry import Registry

from ..data.build import build_loader
from ..modeling.model import build_model
from ..modeling.optimizer import build_optimizer, build_scheduler
from ..modeling.loss import build_loss
from ..modeling.meter import build_meter

from ..utils.train_utils import CudaPreFetcher
from ..utils.checkpoint import save_checkpoint, load_checkpoint, auto_resume, load_model, save_model
from ..utils.writer import get_writer

logger = logging.getLogger(__name__)

TRAINER_REGISTRY = Registry("TRAINER")


def build_trainer(cfg):
    return TRAINER_REGISTRY.get(cfg.TRAINER.NAME)(cfg)


@TRAINER_REGISTRY.register()
class TrainerBase:

    def __init__(self, cfg: CfgNode):
        self.cfg = cfg
        self.debug = cfg.TRAINER.TRAINER_BASE.DEBUG
        self.write_profiler = cfg.TRAINER.TRAINER_BASE.WRITE_PROFILER
        self.write_histogram = cfg.TRAINER.TRAINER_BASE.WRITE_HISTOGRAM
        self.enable_amp = cfg.TRAINER.TRAINER_BASE.AMP

        # initialize in `build` method
        self.model = None
        self.dataloader = None
        self.optimizer = None
        self.scheduler = None
        self.loss_func = None
        self.scaler = None

        # initialize/update in `_before_train` method
        self.global_step = self.epoch_start = self.epoch = 0
        self.gradient_accumulation_step = cfg.TRAINER.TRAINER_BASE.GRADIENT_ACCUMULATION_STEPS
        self.writer = None
        self.meter = None

        self.build()
        
        if not dist.is_initialized() or dist.get_rank() == 0:
            self.info()

    def build(self):
        self.model = build_model(self.cfg)
        self.dataloader = build_loader(self.cfg, mode=("train", "test"))
        self.optimizer = build_optimizer(self.cfg, self.model.parameters())
        self.scheduler = build_scheduler(self.cfg, self.optimizer)
        self.loss_func = build_loss(self.cfg)
        self.scaler = GradScaler() if self.enable_amp else None

    def info(self):
        logger.info(f'Model total parameters: {sum(p.numel() for p in self.model.parameters()):,}')

    def __del__(self):
        self.writer.close()

    def _before_train(self):
        # resume from specified model
        if self.cfg.TRAINER.TRAINER_BASE.RESUME is not None:
            logger.info(f"Resume model parameters from {self.cfg.TRAINER.TRAINER_BASE.RESUME}.")
            load_model(self.cfg.TRAINER.TRAINER_BASE.RESUME, self.model, strict=False)
        # auto resume from checkpoint
        if self.cfg.TRAINER.TRAINER_BASE.AUTO_RESUME:
            logger.info("Auto resume is enabled, recover from the most recent checkpoint.")
            ckpt_dir = os.path.join(self.cfg.LOG.DIR, "checkpoint")
            ckpt_file = auto_resume(ckpt_dir)
            if ckpt_file is not None:
                logger.info(f"auto resume from checkpoint: {ckpt_file}")
                # resume from checkpoint
                self.epoch_start = load_checkpoint(ckpt_file, self.model, self.optimizer, self.scheduler,
                                                   restart_train=False)
            else:
                logger.info(f"No checkpoint was found in directory {ckpt_dir}.")
        else:
            logger.debug("Auto resume is disabled.")
        self.global_step = self.epoch_start * (len(self.dataloader["train"]) // self.gradient_accumulation_step) \
            if not self.debug else self.epoch_start * min(len(self.dataloader["train"]), 100)
        self.writer = get_writer(os.path.join(self.cfg.LOG.DIR, "tensorboard"), purge_step=self.global_step)
        self.meter = build_meter(cfg=self.cfg, writer=self.writer, mode="train")

    def _on_train(self):
        for epoch in range(self.epoch_start, self.cfg.TRAINER.TRAINER_BASE.EPOCH):
            self.epoch = epoch
            logger.debug(f"Epoch {epoch + 1}/{self.cfg.TRAINER.TRAINER_BASE.EPOCH}")
            torch.cuda.empty_cache()
            if self.cfg.TRAINER.TRAINER_BASE.TRAIN_ENABLE:
                self._before_train_epoch()
                self._on_train_epoch()
                self._after_train_epoch()
            else:
                logger.warning("Training is disabled!")
            torch.cuda.empty_cache()
            if self.cfg.TRAINER.TRAINER_BASE.TEST_ENABLE:
                self._before_test_epoch()
                self._on_test_epoch()
                self._after_test_epoch()
            else:
                logger.warning("Testing is disabled!")
            torch.cuda.empty_cache()

    def _after_train(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            save_model(model_file=os.path.join(self.cfg.LOG.DIR, "pytorch_model.bin"), model=self.model)

    def _before_train_epoch(self):
        if dist.is_initialized():
            dist.barrier()
        if "train_sampler" in self.dataloader:
            logger.debug(f"set train sampler step to {self.epoch}")
            self.dataloader["train_sampler"].set_epoch(self.epoch)

    def _on_train_epoch(self):
        dataloader = self.dataloader["train"]
        self.model.train()
        self.meter.set_mode("train")
        if torch.cuda.is_available():
            logger.debug("Building CudaPreFetcher...")
            dataloader = CudaPreFetcher(dataloader)  # prefetch to GPU
        bar = dataloader = tqdm(dataloader,
                                desc=f"Train: {self.epoch + 1}/{self.cfg.TRAINER.TRAINER_BASE.EPOCH}",
                                dynamic_ncols=True,
                                disable=dist.is_initialized() and dist.get_rank() != 0)
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=5, warmup=5, active=5, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(self.cfg.LOG.DIR, "profiler")),
            record_shapes=False,
            with_stack=False
        )
        if self.write_profiler:
            prof.start()
        loss_total = 0.
        logger.debug("Running train epoch for-loop...")
        for cur_step, inputs in enumerate(dataloader):
            # forward
            with autocast(enabled=self.enable_amp):
                outputs = self.model(inputs)
                loss_meta = self.loss_func(inputs, outputs)
            # backward
            if isinstance(loss_meta, dict):
                loss = sum([v for _, v in loss_meta.items()])
            else:
                loss = loss_meta
            loss /= self.gradient_accumulation_step
            if not self.enable_amp:
                loss.backward()
            else:
                self.scaler.scale(loss).backward()
            loss_total += loss.detach()
            if self.gradient_accumulation_step > 1:
                bar.set_postfix(
                    {"Accumulation Step": (cur_step + 1) % self.gradient_accumulation_step}
                )
            # write histogram
            with torch.no_grad():
                if self.debug and self.write_histogram and not self.enable_amp:
                    for n, p in self.model.named_parameters():
                        self.writer.add_histogram(f"weight/{n}", p, global_step=self.global_step)
                        if p.grad is not None:
                            self.writer.add_histogram(f"grad/{n}", p.grad, global_step=self.global_step)
            if (cur_step + 1) % self.gradient_accumulation_step == 0:
                # optimize
                if not self.enable_amp:
                    if self.cfg.TRAINER.TRAINER_BASE.CLIP_NORM is not None:  # clip by norm
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.TRAINER.TRAINER_BASE.CLIP_NORM)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                else:
                    if self.cfg.TRAINER.TRAINER_BASE.CLIP_NORM is not None:  # clip by norm
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.TRAINER.TRAINER_BASE.CLIP_NORM)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                # summary
                with torch.no_grad():
                    if cur_step % self.cfg.TRAINER.TRAINER_BASE.LOG_FREQ == 0:
                        if dist.is_initialized():
                            dist.all_reduce(loss)
                            loss = loss / dist.get_world_size()
                            logger.debug(
                                f"loss (rank {dist.get_rank()}, step {self.global_step}): {loss.cpu().detach().numpy()}"
                            )
                        else:
                            logger.debug(f"loss (step {self.global_step}): {loss.cpu().detach().numpy()}")
                        self.writer.add_scalar("train/loss", loss_total, global_step=self.global_step)
                        if isinstance(loss_meta, dict):
                            self.writer.add_scalars("train/loss_meta", loss_meta, global_step=self.global_step)
                        self.writer.add_scalars(
                            "train/lr",
                            {
                                f"param_group_{i}": group["lr"] if self.scheduler is None else group
                                for i, group in enumerate(self.optimizer.param_groups if self.scheduler is None
                                                          else self.scheduler.get_last_lr())
                            },
                            global_step=self.global_step
                        )
                    self.meter.update(inputs=inputs, outputs=outputs, global_step=self.global_step)
                loss_total = 0.
                self.global_step += 1
                if self.write_profiler:
                    prof.step()
            if self.debug and cur_step + 1 >= 100 * self.gradient_accumulation_step:
                logger.warning("Debug mode is enabled, only run for 100 step.")
                break
        logger.debug("Train epoch for-loop finished.")
        if self.write_profiler:
            prof.stop()
        self.optimizer.zero_grad()
        self.meter.summary(epoch=self.epoch + 1)
        self.meter.reset()

    def _after_train_epoch(self):
        if (self.epoch + 1) % self.cfg.TRAINER.TRAINER_BASE.SAVE_FREQ == 0:
            if not dist.is_initialized() or dist.get_rank() == 0:  # ddp is not enabled or global rank is 0
                save_checkpoint(ckpt_folder=os.path.join(self.cfg.LOG.DIR, "checkpoint"),
                                epoch=self.epoch + 1,
                                model=self.model,
                                optimizer=self.optimizer,
                                scheduler=self.scheduler,
                                config=self.cfg)
        if self.scheduler is not None:
            self.scheduler.step()

    def _before_test_epoch(self):
        pass

    @torch.no_grad()
    def _on_test_epoch(self):
        self.model.eval()
        self.meter.set_mode("val")
        dataloader = self.dataloader["test"]
        dataloader = tqdm(dataloader, desc=f"Eval epoch {self.epoch + 1}", dynamic_ncols=True,
                          disable=dist.is_initialized() and dist.get_rank() != 0)
        if torch.cuda.is_available():
            dataloader = CudaPreFetcher(dataloader)  # move to GPU
        for inputs in dataloader:
            outputs = self.model(inputs)
            self.meter.update(inputs=inputs, outputs=outputs)
        self.meter.summary(epoch=self.epoch + 1)
        self.meter.reset()

    def _after_test_epoch(self):
        pass

    def train(self):
        self._before_train()
        self._on_train()
        self._after_train()

    def eval(self):
        pass

# -*- coding: utf-8 -*-
# @Time    : 2022/11/12 22:28
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : trainer.py
import glob
import math
import os
import re

from tqdm import tqdm
import logging

from hydra.utils import get_object
from dataclasses import dataclass
from enum import Enum
from typing import *

import torch
from torch import nn, optim
from torch.utils import data
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    CPUOffload, ShardingStrategy
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy
)

from collections import defaultdict
import contextlib
from functools import partial

from mm_video.config import trainer_store
from mm_video.modeling.meter import Meter
from mm_video.modeling.optimization import get_linear_schedule_with_warmup
from mm_video.utils.train_utils import (
    CudaPreFetcher, get_trainable_parameters, compute_total_gradient_norm, get_world_size
)
from mm_video.utils.writer import get_writer
from mm_video.utils.profile import Timer
from .trainer_utils import barrier, get_module_class_from_name, load_state_dict, unwrap_model

__all__ = [
    "Trainer",
    "TrainingStrategy", "TrainingStrategyConfig", "DataLoaderConfig", "TrainingConfig"
]

logger = logging.getLogger(__name__)

PREFIX_CHECKPOINT_DIR = "checkpoint"

MODEL_NAME = "pytorch_model"
OPTIMIZER_NAME = "optimizer"
SCHEDULER_NAME = "scheduler"
MODEL_NAME_BIN = f"{MODEL_NAME}.bin"
OPTIMIZER_NAME_BIN = f"{OPTIMIZER_NAME}.bin"
SCHEDULER_NAME_BIN = f"{SCHEDULER_NAME}.bin"


@dataclass
class DataLoaderConfig:
    """DataLoader configuration options.

    Attributes:
        batch_size (int, optional): Batch size to use during training and evaluation, if not overriden.
            Defaults to 1.
        train_batch_size (int, optional): Batch size to use during training. Overrides `batch_size`, if set.
            Defaults to the value of `batch_size`.
        test_batch_size (int, optional): Batch size to use during testing. Overrides `batch_size`, if set.
            Defaults to the value of `batch_size`.
        eval_batch_size (int, optional): Batch size to use during evaluation. Overrides `batch_size`, if set.
            Defaults to the value of `batch_size`.
    """
    collate_fn: Optional[str] = None

    # batch size
    batch_size: int = 1
    train_batch_size: Optional[int] = None
    test_batch_size: Optional[int] = None
    eval_batch_size: Optional[int] = None

    num_workers: int = 0
    shuffle: bool = True
    prefetch_factor: Optional[int] = None
    multiprocessing_context: str = "spawn"
    pin_memory: bool = False


class TrainingStrategy(Enum):
    cpu = "cpu"
    ddp = "ddp"
    fsdp = "fsdp"


@dataclass
class TrainingStrategyConfig:
    strategy: TrainingStrategy = TrainingStrategy.ddp if torch.cuda.is_available() else TrainingStrategy.cpu

    ddp_find_unused_parameters: bool = False

    fsdp_offload: bool = False
    fsdp_transformer_layer_cls_to_wrap: Optional[List[str]] = None
    fsdp_sync_module_states: bool = False
    fsdp_use_orig_params: bool = False
    fsdp_sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD


@dataclass
class TrainingConfig:
    do_train: bool = False
    do_test: bool = False
    do_eval: bool = False

    num_train_epochs: int = 5

    amp: bool = False

    clip_norm: Optional[float] = None

    # Resume model from pretrained parameters
    resume: Optional[str] = None
    # Resume from last checkpoint saved in output directory
    resume_from_checkpoint: bool = False

    detect_anomaly: bool = False

    # Enable/disable PyTorch profiler
    write_profiler: bool = False
    # How often to write training loss, loss metadata, and learning rate to TensorBoard.
    # Set to `None` to disable.
    write_loss_and_learning_rate: Optional[int] = 1
    # How often to write parameter histograms to TensorBoard. Set to `None` to disable.
    write_histogram: Optional[int] = None
    # How often to compute and write total gradient norm to TensorBoard. Set to `None` to disable.
    write_gradient_norm: Optional[int] = None

    output_dir: str = "${hydra:runtime.output_dir}"
    save_freq: int = 1

    debug: bool = False

    # Optimizer and Scheduler options
    learning_rate: float = 1e-5
    warmup_ratio: Optional[float] = None
    warmup_steps: Optional[int] = None

    gradient_accumulation_steps: int = 1


@trainer_store(zen_partial=True)  # Set `zen_partial=True` if you inherit from this trainer
class Trainer:
    def __init__(
            self,
            datasets: Dict[str, data.Dataset], model: nn.Module, meter: Meter,
            training_config: TrainingConfig = TrainingConfig(),
            dataloader_config: DataLoaderConfig = DataLoaderConfig(),
            training_strategy_config: TrainingStrategyConfig = TrainingStrategyConfig(),
            optimizer: Optional[optim.Optimizer] = None,
            scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
    ):
        cfg: TrainingConfig = training_config
        self.cfg: TrainingConfig = cfg
        self.dataloader_cfg: DataLoaderConfig = dataloader_config
        self.training_strategy_cfg: TrainingStrategyConfig = training_strategy_config

        assert cfg.write_histogram is None or not cfg.amp, \
            "If using AMP, `write_histogram_freq` cannot be enabled and must be set to `None`."

        torch.autograd.set_detect_anomaly(cfg.detect_anomaly)

        # Initialize distribution
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))  # get RANK from environment

        self.dataset = datasets
        self.dataloader = self.build_loader(datasets, cfg=dataloader_config)

        self.model_wrapped = self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.meter = meter

        self.resume = cfg.resume
        self.output_dir = cfg.output_dir
        self.save_freq = cfg.save_freq

        def get_write_freq(x: Optional[int]):
            assert x is None or type(x) is int
            return float("inf") if x is None else x

        self.write_loss_and_learning_rate_freq = get_write_freq(cfg.write_loss_and_learning_rate)
        self.write_histogram_freq = get_write_freq(cfg.write_histogram)
        self.write_gradient_norm_freq = get_write_freq(cfg.write_gradient_norm)

        self.enable_amp = cfg.amp
        self.scaler: GradScaler = GradScaler() if cfg.amp else None
        self.clip_norm = cfg.clip_norm

        self.epoch_total = cfg.num_train_epochs
        self.global_step = self.epoch = 0  # Init to 0

        self.gradient_accumulation_step = cfg.gradient_accumulation_steps
        assert self.gradient_accumulation_step >= 1

        self.info()

    def create_optimizer(self):
        logger.debug("Creating optimizer...")
        self.optimizer = optim.AdamW(self.model_wrapped.parameters(), lr=self.cfg.learning_rate)
        logger.info("Optimizer created successfully.")

    def create_scheduler(self, optimizer: optim.Optimizer):
        logger.debug("Creating scheduler...")
        max_train_steps = math.ceil(len(self.dataloader["train"]) * self.epoch_total / self.gradient_accumulation_step)

        warmup_steps = 0
        assert (self.cfg.warmup_ratio is None) or (self.cfg.warmup_steps is None), \
            "Both warmup_ratio and warmup_steps should not be set simultaneously."
        if self.cfg.warmup_ratio is not None:
            warmup_steps = int(max_train_steps * self.cfg.warmup_ratio)
        if self.cfg.warmup_steps is not None:
            warmup_steps = self.cfg.warmup_steps

        self.scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                         num_warmup_steps=warmup_steps,
                                                         num_training_steps=max_train_steps)
        logger.info("Scheduler created successfully. Warmup steps: %d", warmup_steps)

    @property
    def should_write(self) -> bool:
        return not dist.is_initialized() or dist.get_rank() == 0

    def save_model(self, output_dir: Optional[str] = None, state_dict: Optional[Any] = None):
        """
        Save model (state dict) to disk. Make sure this function is only executed on process 0.

        @param output_dir:
        @param state_dict:
        """
        output_dir = output_dir if output_dir is not None else self.output_dir
        model_file = os.path.join(output_dir, MODEL_NAME_BIN)
        logger.info(f"Saving model to {model_file}")

        if state_dict is None:
            # Call the state_dict on each rank before saving on rank 0, required by FSDP model
            model = unwrap_model(self.model_wrapped)
            state_dict = model.state_dict()

        if self.should_write:
            os.makedirs(output_dir, exist_ok=True)
            torch.save(state_dict, model_file)

    def _save_checkpoint(self):
        """
        Save current training status to checkpoint

        """
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}"
        output_dir = os.path.join(self.output_dir, checkpoint_folder)
        logger.debug("Saving trainer checkpoint to %s", output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Save model status
        state_dict = self.model_wrapped.state_dict()
        if self.should_write:
            self.save_model(output_dir=output_dir, state_dict=state_dict)

        # Save optimizer
        if self.should_write:
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME_BIN))

        # Save scheduler
        if self.should_write and self.scheduler is not None:
            torch.save(self.scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME_BIN))

        logger.debug("Checkpoint is saved successfully.")

    def _load_checkpoint(self, checkpoint: str):
        """
        Resume saved training status from checkpoint

        """
        logger.info("Loading checkpoint from %s", checkpoint)

        # Load model states
        model_state_dict = torch.load(os.path.join(checkpoint, MODEL_NAME_BIN), map_location="cpu")
        self.model_wrapped.load_state_dict(model_state_dict)

        # Load optimizer
        optimizer_file = os.path.join(checkpoint, OPTIMIZER_NAME_BIN)
        if os.path.exists(optimizer_file):
            self.optimizer.load_state_dict(torch.load(optimizer_file, map_location="cpu"))

        # Load scheduler
        scheduler_file = os.path.join(checkpoint, SCHEDULER_NAME_BIN)
        if os.path.exists(scheduler_file):
            self.scheduler.load_state_dict(torch.load(scheduler_file, map_location="cpu"))

        logger.info("Checkpoint is loaded successfully.")

        # TODO: skip epoch and set train step after load checkpoint

    def _get_last_checkpoint(self) -> Optional[str]:
        """
        Get the last checkpoint listed in output directory

        """
        checkpoint_folder_pattern = f"{PREFIX_CHECKPOINT_DIR}-[0-9]*"
        checkpoint_folders = glob.glob(os.path.join(self.output_dir, checkpoint_folder_pattern))
        checkpoint_folders = [f for f in checkpoint_folders if os.path.isdir(f)]
        if checkpoint_folders:
            last_checkpoint_folder = sorted(checkpoint_folders, key=lambda x: int(re.findall(r"\d+", x)[-1]))[-1]
            logger.debug("Found the last checkpoint at: %s.", last_checkpoint_folder)
            return last_checkpoint_folder
        else:
            logger.debug("No previous checkpoint found in the output directory.")

    @staticmethod
    def build_loader(datasets: Dict[str, data.Dataset], cfg: DataLoaderConfig):
        assert all(split in ("train", "test", "eval") for split in datasets.keys()), \
            f"Invalid split found in {datasets.keys()}. Must be one of 'train', 'test', or 'eval'."
        timer = Timer(msg="Building dataloader...")
        world_size = get_world_size()
        loader_and_sampler = {}
        for split, dataset in datasets.items():
            shuffle = cfg.shuffle if split == "train" else False
            batch_size = getattr(cfg, f"{split}_batch_size")
            if batch_size is None:
                batch_size = getattr(cfg, "batch_size")
            collate_fn = get_object(cfg.collate_fn) if cfg.collate_fn is not None else None

            sampler = data.distributed.DistributedSampler(dataset, shuffle=shuffle) if world_size > 1 else None

            loader = data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False if world_size > 1 else shuffle,
                sampler=sampler,
                num_workers=cfg.num_workers,
                collate_fn=collate_fn,
                pin_memory=cfg.pin_memory,
                persistent_workers=False,
                prefetch_factor=cfg.prefetch_factor,
                multiprocessing_context=cfg.multiprocessing_context if cfg.num_workers else None
            )

            loader_and_sampler[split] = loader
            if sampler is not None:
                loader_and_sampler[f"{split}_sampler"] = sampler
        timer.end()
        return loader_and_sampler

    def _wrap_model(self, model: nn.Module, training: bool = True):
        cfg = self.training_strategy_cfg

        # Move to GPU
        if cfg.strategy in (TrainingStrategy.ddp, TrainingStrategy.fsdp):
            logger.debug("Moving model to device: %s...", torch.cuda.current_device())
            model.cuda()

        # Do not wrap model if not training.
        if not training:
            logger.debug("Not training, return unwrapped model.")
            return model

        # Avoid wrap model more than once.
        if isinstance(model, (FullyShardedDataParallel, nn.DataParallel)):
            return model

        # wrap model
        logger.debug("Applying training strategy...")
        if cfg.strategy in (TrainingStrategy.ddp, TrainingStrategy.fsdp):
            logger.debug("Model is moved to device: %s", torch.cuda.current_device())
            if cfg.strategy == TrainingStrategy.ddp:
                logger.debug("Building DistributedDataParallel, check whether the program is hanging...")
                model = nn.parallel.DistributedDataParallel(
                    model,
                    find_unused_parameters=cfg.ddp_find_unused_parameters
                )
            elif cfg.strategy == TrainingStrategy.fsdp:
                logger.debug("Building FullyShardedDataParallel, check whether the program is hanging...")

                # From Hugging Face Trainer
                auto_wrap_policy = None
                if cfg.fsdp_transformer_layer_cls_to_wrap is not None:
                    transformer_cls_to_wrap = set()
                    for layer_class in cfg.fsdp_transformer_layer_cls_to_wrap:
                        transformer_cls = get_module_class_from_name(model, layer_class)
                        if transformer_cls is None:
                            raise Exception("Could not find the transformer layer class to wrap in the model.")
                        else:
                            transformer_cls_to_wrap.add(transformer_cls)
                    auto_wrap_policy = partial(
                        transformer_auto_wrap_policy,
                        transformer_layer_cls=transformer_cls_to_wrap
                    )

                self.model = model = FullyShardedDataParallel(
                    model,
                    cpu_offload=CPUOffload(offload_params=cfg.fsdp_offload),
                    auto_wrap_policy=auto_wrap_policy,
                    sync_module_states=cfg.fsdp_sync_module_states,
                    use_orig_params=cfg.fsdp_use_orig_params,
                    sharding_strategy=cfg.fsdp_sharding_strategy
                )
            else:
                raise RuntimeError(f"Training strategy '{cfg.strategy}' is not supported!")
        elif cfg.strategy == TrainingStrategy.cpu:
            pass
        else:
            raise RuntimeError(f"Training strategy '{cfg.strategy}' is not supported!")
        logger.debug("Model is wrapped successfully.")
        return model

    def _prefetch_to_gpu(self, dataloader: data.DataLoader) -> data.DataLoader:
        """
        Wraps the given dataloader with `CudaPreFetcher` to prefetch tensors to GPU.

        This transformation is only applied when the training strategy is either ddp or FSDP.
        Otherwise, the dataloader remains unchanged.

        """
        if self.training_strategy_cfg.strategy in (TrainingStrategy.ddp, TrainingStrategy.fsdp):
            logger.debug("Building CudaPreFetcher. "
                         "This might take a moment as it waits for all Torch DataLoader workers to initialize...")
            dataloader = CudaPreFetcher(dataloader)
            logger.debug("CudaPreFetcher successfully built.")
        return dataloader

    def info(self):
        """CUSTOMIZE: to print some information"""
        logger.info("Train Epoch: %d", self.epoch_total)
        trainable_params, all_param, trainable_params_names = get_trainable_parameters(self.model)
        logger.info("Trainable params: %d", trainable_params)
        logger.debug("Trainable params: \n\t%s", '\n\t'.join(trainable_params_names))
        logger.info("All params: %d", all_param)
        logger.info("Trainable%%: %f", 100 * trainable_params / all_param)

    def _before_train(self):
        # Wrap model before training
        self.model_wrapped = self._wrap_model(self.model)

        # Resume from specified model, use for load pretrained weight
        if self.resume is not None:
            logger.info(f"Resume model parameters from {self.resume}.")
            load_state_dict(self.model_wrapped, model_file=self.resume, strict=False)

        # Build optimizer and scheduler before training if is not passed to trainer
        if self.optimizer is None:
            self.create_optimizer()
        if self.scheduler is None:
            self.create_scheduler(self.optimizer)

        # Resume from last checkpoint
        if self.cfg.resume_from_checkpoint:
            checkpoint_folder = self._get_last_checkpoint()
            if checkpoint_folder is not None:
                logger.info("Resuming from the last checkpoint: %s", checkpoint_folder)
                self._load_checkpoint(checkpoint_folder)
            else:
                logger.warning("`resume_from_checkpoint` is enabled, "
                               "but no checkpoint is found in the output directory.")

        self.writer = get_writer(os.path.join(self.output_dir, "tensorboard"), purge_step=self.global_step)

    def _on_train(self):
        for epoch in range(self.epoch, self.epoch_total):
            self.epoch = epoch
            logger.debug(f"Epoch {epoch + 1}/{self.epoch_total}")

            if self.cfg.do_train:
                self._before_train_epoch()
                self._on_train_epoch()
                self._after_train_epoch()
            else:
                logger.warning("Training is disabled. Skipping training process...")

            # TODO: Add an option to set test frequency
            if self.cfg.do_test:
                self._before_test_epoch()
                self._on_test_epoch()
                self._after_test_epoch()
            else:
                logger.info("Testing is disabled. Skipping testing process...")

            # Only run test for once if training is disabled
            if not self.cfg.do_train:
                break

    def _after_train(self):
        barrier()
        state_dict = self.model_wrapped.state_dict()
        if self.should_write:
            self.save_model(state_dict=state_dict)

    def _before_train_epoch(self):
        torch.cuda.empty_cache()
        barrier(debug_msg="training loop")
        if "train_sampler" in self.dataloader:
            logger.debug(f"set train sampler step to {self.epoch}")
            self.dataloader["train_sampler"].set_epoch(self.epoch)

    def _on_train_epoch(self):
        train_dataloader = self._prefetch_to_gpu(self.dataloader["train"])
        model = self.model_wrapped

        progress_bar = tqdm(desc=f"Train: {self.epoch + 1}/{self.epoch_total}",
                            dynamic_ncols=True,
                            total=len(train_dataloader) // self.gradient_accumulation_step,
                            disable=dist.is_initialized() and dist.get_rank() != 0)

        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=5, warmup=5, active=5, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(self.output_dir, "profiler")),
            record_shapes=False,
            with_stack=False
        ) if self.cfg.write_profiler else None
        if self.cfg.write_profiler:
            logger.warning("Torch profiler is enabled, performance may be impacted.")
            prof.start()

        loss_total = 0.0
        loss_meta_total = defaultdict(float)

        logger.debug("Running train epoch for-loop...")
        for cur_step, inputs in enumerate(train_dataloader):
            outputs, loss, loss_meta = self._train_step(model=model, inputs=inputs)

            loss_total += loss
            for k, v in loss_meta.items():
                loss_meta_total[k] += v

            if self.gradient_accumulation_step > 1:
                progress_bar.set_postfix_str(f"Accumulation Step={(cur_step + 1) % self.gradient_accumulation_step}")
            if (cur_step + 1) % self.gradient_accumulation_step == 0:
                # summary
                with torch.no_grad():
                    if (cur_step + 1) % self.write_histogram_freq == 0:
                        self._write_histogram()
                    if (cur_step + 1) % self.write_gradient_norm_freq == 0:
                        self._write_total_gradient_norm()
                    if (cur_step + 1) % self.write_loss_and_learning_rate_freq == 0:
                        # noinspection PyTypeChecker
                        self._write_loss_and_learning_rate(loss=loss_total, loss_meta=loss_meta_total)
                    self.meter.update(
                        inputs=inputs, outputs=outputs,
                        writer=self.writer, main_tag="train", global_step=self.global_step
                    )

                # Clip by norm
                if self.clip_norm is not None:
                    if self.enable_amp:
                        self.scaler.unscale_(self.optimizer)

                    if self.training_strategy_cfg.strategy == TrainingStrategy.fsdp:
                        model.clip_grad_norm_(self.clip_norm)
                    else:
                        nn.utils.clip_grad_norm_(model.parameters(), self.clip_norm)

                # Optimize
                if self.enable_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

                if self.scheduler is not None:
                    self.scheduler.step()
                self.global_step += 1
                loss_total = 0.
                loss_meta_total = defaultdict(float)
                progress_bar.update()
                if self.cfg.write_profiler:
                    prof.step()

            # Exit if debug
            if self.cfg.debug and cur_step + 1 >= 100 * self.gradient_accumulation_step:
                logger.warning("Debug mode is enabled, only run for 100 step.")
                break

        logger.debug("Train epoch for-loop finished.")

        if self.cfg.write_profiler:
            prof.stop()

    def _train_step(
            self,
            model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Performs a single training step on the model.
        """
        model.train()

        if self.enable_amp:
            autocast_context_manager = autocast()
        else:
            autocast_context_manager = contextlib.nullcontext()

        # Forward pass
        with autocast_context_manager:
            outputs = model(inputs)
            assert "loss" in outputs, \
                "The model forward function should return a dictionary with the key `loss` during training."
            loss = outputs["loss"]

        # Sum up loss if returned loss is a dictionary
        if isinstance(loss, dict):
            loss_meta = {k: v for k, v in loss.items()}
            loss = sum([v for _, v in loss.items()])
        else:
            loss_meta = {}

        if self.enable_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Divide with gradient accumulation steps and return
        loss = loss.detach() / self.gradient_accumulation_step
        loss_meta = {k: v.detach() / self.gradient_accumulation_step for k, v in loss_meta.items()}
        return outputs, loss, loss_meta

    def _after_train_epoch(self):
        # reset optimizer
        self.optimizer.zero_grad()
        # write metric and reset meter
        self.meter.summary(writer=self.writer, main_tag="train", global_step=self.global_step)
        self.meter.reset()
        # save checkpoint
        if (self.epoch + 1) % self.save_freq == 0:
            self._save_checkpoint()
        torch.cuda.empty_cache()

    def _before_test_epoch(self):
        torch.cuda.empty_cache()
        barrier(debug_msg="test epoch")
        if "test_sampler" in self.dataloader:
            logger.debug(f"set test sampler step to {self.epoch}")
            self.dataloader["test_sampler"].set_epoch(self.epoch)

    @torch.no_grad()
    def _on_test_epoch(self):
        model = self.model_wrapped
        dataloader = self._prefetch_to_gpu(self.dataloader["test"])

        progress_bar = tqdm(desc=f"Test on epoch {self.epoch + 1}", dynamic_ncols=True, total=len(dataloader),
                            disable=not self.should_write)

        for inputs in dataloader:
            model.eval()
            outputs = self.model(inputs)
            self.meter.update(
                inputs=inputs, outputs=outputs,
                writer=self.writer, main_tag="test", global_step=self.epoch
            )
            progress_bar.update()

    def _after_test_epoch(self):
        # write metric and reset meter
        self.meter.summary(writer=self.writer, main_tag="test", global_step=self.epoch)
        self.meter.reset()
        torch.cuda.empty_cache()

    def _before_eval(self):
        torch.cuda.empty_cache()
        barrier(debug_msg="evaluation")
        if "eval_sampler" in self.dataloader:
            logger.debug(f"Set eval sampler step to 0")
            self.dataloader["eval_sampler"].set_epoch(0)

        # Resume from specified model, use for load pretrained weight
        if self.resume is not None:
            logger.info(f"Resume model parameters from {self.resume}.")
            load_state_dict(self.model_wrapped, model_file=self.resume, strict=False)

        self.writer = get_writer(os.path.join(self.output_dir, "tensorboard"))

    def _on_eval(self):
        model = self._wrap_model(self.model_wrapped, training=False)
        dataloader = self._prefetch_to_gpu(self.dataloader["eval"])

        process_bar = tqdm(desc="Evaluation", dynamic_ncols=True, total=len(dataloader),
                           disable=not self.should_write)

        for inputs in dataloader:
            model.eval()
            outputs = self.model(inputs)
            self.meter.update(
                inputs=inputs, outputs=outputs,
                writer=self.writer, main_tag="eval", global_step=self.epoch
            )
            process_bar.update()

    def _after_eval(self):
        self.meter.summary(writer=self.writer, main_tag="eval", global_step=self.epoch)
        self.meter.reset()

    def run(self):
        if self.cfg.do_train or self.cfg.do_test:
            self._before_train()
            self._on_train()
            self._after_train()

        if self.cfg.do_eval:
            self._before_eval()
            self._on_eval()
            self._after_eval()

    @torch.no_grad()
    def _write_histogram(self):
        """
        Writes histograms of model parameters and gradients.

        The histograms are written to TensorBoard under the tags:
        "weights/{parameter name}" and "grads/{parameter name}", respectively.
        """
        with Timer("Writing histogram..."):
            for n, p in self.model.named_parameters():
                self.writer.add_histogram(f"weight/{n}", p.detach().float(), global_step=self.global_step)
                if p.grad is not None:
                    self.writer.add_histogram(f"grad/{n}", p.grad.detach().float(), global_step=self.global_step)

    @torch.no_grad()
    def _write_total_gradient_norm(self):
        """
        Compute and writes total gradient norm.

        This function should be called before gradient clipping. The total gradient norm over all model parameters is
        computed, and written to TensorBoard under the tag "train/norm".
        """
        with Timer("Writing total gradient norm..."):
            total_norm = compute_total_gradient_norm(self.model)
            logger.debug(f"Step: {self.global_step} | Total gradient norm: {total_norm}")
            self.writer.add_scalar("train/norm", total_norm, global_step=self.global_step)

    @torch.no_grad()
    def _write_loss_and_learning_rate(
            self, loss: torch.Tensor, loss_meta: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ):
        """
        Writes training loss, loss metadata, and learning rate to TensorBoard.

        :param loss: Training loss.
        :param loss_meta:
        :return:
        """
        if dist.is_initialized():
            dist.all_reduce(loss)
            loss /= dist.get_world_size()
            for k in sorted(loss_meta.keys()):
                dist.all_reduce(loss_meta[k])
                loss_meta[k] /= dist.get_world_size()

        self.writer.add_scalar("train/loss", loss, global_step=self.global_step)
        if isinstance(loss_meta, dict):
            self.writer.add_scalars("train/loss_meta", loss_meta, global_step=self.global_step)

        learning_rate = [group["lr"] if self.scheduler is None else group for group in
                         (self.optimizer.param_groups if self.scheduler is None else self.scheduler.get_last_lr())]
        self.writer.add_scalars("train/lr", {f"param_group_{i}": lr for i, lr in enumerate(learning_rate)},
                                global_step=self.global_step)

        logger.debug(
            f"Step: %s | Loss: %s | Learning rate: %s",
            self.global_step, loss.cpu().detach().numpy(),
            learning_rate[0] if len(learning_rate) == 1 else learning_rate
        )

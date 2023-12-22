# -*- coding: utf-8 -*-
# @Time    : 10/23/23
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : runner.py

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from dataclasses import dataclass
from typing import *

import os
import torch
import torch.distributed as dist
from torch import nn, optim
from torch.utils import data

from mm_video.utils.logging import get_timestamp
from mm_video.utils.train_utils import manual_seed

from mm_video.config import BaseConfig, register_runner_config
from mm_video.trainer import TrainerConfig, Trainer
from mm_video.modeling.meter import Meter, DummyMeter
from mm_video.utils.profile import Timer

import logging

logger = logging.getLogger(__name__)

__all__ = ["Runner", "RunnerConfig", "main"]


class Runner:
    cfg: BaseConfig
    dataset: Dict[str, data.Dataset]
    model: nn.Module
    meter: Meter
    trainer: Trainer

    def __init__(self, cfg: BaseConfig):
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))  # get RANK from environment
        manual_seed(cfg.system)

        self.cfg = cfg
        self.build_dataset(cfg.dataset)
        self.build_model(cfg.model)
        self.build_meter(cfg.meter)
        self.build_trainer(cfg.trainer)

    def build_dataset(self, dataset_config: DictConfig):
        with Timer("Building dataset from the configuration..."):
            self.dataset = {split: instantiate(dataset_config, split=split) for split in ("train", "test", "eval")}

    def build_model(self, model_builder_config: DictConfig):
        with Timer("Building model from the configuration..."):
            self.model = instantiate(model_builder_config)

    def build_meter(self, meter_config: DictConfig):
        self.meter: Meter = instantiate(meter_config)
        if self.meter is None:
            logger.info("Meter is not specified.")
            self.meter = DummyMeter()

    def build_trainer(self, trainer_config: TrainerConfig):
        self.trainer = instantiate(trainer_config)(
            datasets=self.dataset, model=self.model,
            meter=self.meter
        )

    def run(self):
        self.trainer.train()
        self.trainer.eval()


@register_runner_config(name=f"{Runner.__qualname__}")
@dataclass
class RunnerConfig:
    _target_: str = f"{__name__}.{Runner.__qualname__}"


@hydra.main(version_base=None, config_name="config",
            config_path=f"{os.path.dirname(os.path.abspath(__file__))}/../../configs")
def main(cfg: BaseConfig):
    runner = instantiate(cfg.runner, _partial_=True)(cfg)
    runner.run()

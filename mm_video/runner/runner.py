# -*- coding: utf-8 -*-
# @Time    : 10/23/23
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : runner.py

import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from dataclasses import dataclass
from typing import *

import os
import shutil
from torch.nn import Module
from torch.utils.data import Dataset

import mm_video
from mm_video.trainer.trainer_utils import manual_seed
from mm_video.config import BaseConfig, register_runner_config
from mm_video.modeling.meter import Meter, DummyMeter
from mm_video.utils.profile import Timer

import logging

logger = logging.getLogger(__name__)

__all__ = ["Runner", "RunnerConfig", "main"]


class Runner:
    """
    Runner is a basic entry point for building datasets and models, and running the training, testing, and evaluation
    loop.

    """

    @staticmethod
    def build_dataset(dataset_config: DictConfig) -> Dict[str, Dataset]:
        with Timer("Building dataset from the configuration..."):
            dataset = {split: instantiate(dataset_config, split=split) for split in ("train", "test", "eval")}
        return dataset

    @staticmethod
    def build_model(model_builder_config: DictConfig) -> Module:
        with Timer("Building model from the configuration..."):
            model = instantiate(model_builder_config)
        return model

    @staticmethod
    def build_meter(meter_config: DictConfig) -> Meter:
        meter = instantiate(meter_config)
        if meter is None:
            logger.info("Meter is not specified.")
            meter = DummyMeter()
        return meter

    @staticmethod
    def save_code():
        mm_video_root = mm_video.__path__[0]
        assert os.path.exists(mm_video.__path__[0])
        shutil.make_archive(base_name=os.path.join(HydraConfig.get().runtime.output_dir, "code"), format="zip",
                            root_dir=os.path.dirname(mm_video_root), base_dir=os.path.basename(mm_video_root))

    def run(self, cfg: BaseConfig):
        if cfg.system.deterministic:
            manual_seed(cfg.system.seed)

        self.save_code()
        dataset = self.build_dataset(cfg.dataset)
        model = self.build_model(cfg.model)
        meter = self.build_meter(cfg.meter)

        trainer = instantiate(cfg.trainer)(
            datasets=dataset,
            model=model,
            meter=meter
        )

        trainer.run()


@register_runner_config(name=f"{Runner.__qualname__}")
@dataclass
class RunnerConfig:
    _target_: str = f"{__name__}.{Runner.__qualname__}"


@hydra.main(version_base=None, config_name="config",
            config_path=f"{os.path.dirname(os.path.abspath(__file__))}/../../configs")
def main(cfg: BaseConfig):
    runner = instantiate(cfg.runner)
    runner.run(cfg)

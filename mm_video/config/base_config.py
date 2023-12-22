# -*- coding: utf-8 -*-
# @Time    : 10/11/23
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : base_config.py

import hydra
from hydra.core.config_store import ConfigStore

from omegaconf import OmegaConf, MISSING
from dataclasses import dataclass, field
from typing import List, Any, Optional

# import config nodes
from mm_video.utils.train_utils import SystemConfig
from mm_video.utils.logging import LogConfig

"""
Define the structure of base configuration for this template
"""

__all__ = ["BaseConfig"]


@dataclass
class BaseConfig:
    defaults: List[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"trainer": "Trainer"},
            {"runner": "Runner"}
        ]
    )

    # Basic information configuration
    system: SystemConfig = field(default_factory=SystemConfig)
    log: LogConfig = field(default_factory=LogConfig)

    # Main component configuration
    dataset: Any = MISSING
    model: Any = MISSING
    optimizer: Any = MISSING
    scheduler: Optional[Any] = None
    meter: Optional[Any] = None

    # Pipeline configuration
    trainer: Any = MISSING
    runner: Any = MISSING


ConfigStore.instance().store(name="base_config", node=BaseConfig)

if __name__ == "__main__":
    @hydra.main(version_base=None, config_name="config", config_path="../../configs")
    def main(cfg: BaseConfig):
        print(OmegaConf.to_yaml(cfg))


    main()

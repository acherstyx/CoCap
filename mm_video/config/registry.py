# -*- coding: utf-8 -*-
# @Time    : 10/11/23
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : registry.py

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, ListConfig
from typing import Union, Optional
from dataclasses import dataclass

__all__ = [
    "register_dataset_config", "register_model_config", "register_optimizer_config", "register_scheduler_config",
    "register_meter_config", "register_trainer_config", "register_runner_config"
]

# a list of hard-coded registry positions for config store
DATASET_CONFIG_GROUP = "dataset"
MODEL_CONFIG_GROUP = "model"
OPTIMIZER_CONFIG_GROUP = "optimizer"
SCHEDULER_CONFIG_GROUP = "scheduler"
METER_CONFIG_GROUP = "meter"
TRAINER_CONFIG_GROUP = "trainer"
RUNNER_CONFIG_GROUP = "runner"


def __register(name: str, cls: Optional[object], group: str):
    """
    :param name: The name of the config to be registered in the config store.
    :param cls: Class name. If specified, a new data class will be built and registered with `_target_` set to this
    class.
    :param group: hydra config group
    :return: A new Python decorator
    """
    if name is None:
        assert cls is not None, "class must be given if name is None"
        name = cls.__name__

    def register(config_node: Union[DictConfig, ListConfig]):
        if cls is not None:  # Add/Replace `_target_`
            @dataclass
            class ConfigNode(config_node):
                _target_: str = f"{cls.__module__}.{cls.__qualname__}"

            config_node = ConfigNode

        ConfigStore.instance().store(name=name, group=group, node=config_node)
        return config_node

    return register


def register_dataset_config(name: str = None, cls: object = None):
    return __register(name=name, cls=cls, group=DATASET_CONFIG_GROUP)


def register_model_config(name: str = None, cls: object = None):
    return __register(name=name, cls=cls, group=MODEL_CONFIG_GROUP)


def register_optimizer_config(name: str = None, cls: object = None):
    return __register(name=name, cls=cls, group=OPTIMIZER_CONFIG_GROUP)


def register_scheduler_config(name: str = None, cls: object = None):
    return __register(name=name, cls=cls, group=SCHEDULER_CONFIG_GROUP)


def register_meter_config(name: str = None, cls: object = None):
    return __register(name=name, cls=cls, group=METER_CONFIG_GROUP)


def register_trainer_config(name: str = None, cls: object = None):
    return __register(name=name, cls=cls, group=TRAINER_CONFIG_GROUP)


def register_runner_config(name: str = None, cls: object = None):
    return __register(name=name, cls=cls, group=RUNNER_CONFIG_GROUP)

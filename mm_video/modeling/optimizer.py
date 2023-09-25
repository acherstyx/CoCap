# -*- coding: utf-8 -*-
# @Time    : 2022/11/13 02:00
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : optimizer.py

import logging

from torch import optim
from fvcore.common.config import CfgNode
from fvcore.common.registry import Registry
from inspect import isclass

logger = logging.getLogger(__name__)

OPTIMIZER_REGISTRY = Registry("OPTIMIZER")
SCHEDULER_REGISTRY = Registry("SCHEDULER")

for attr_name in optim.__dict__.keys():
    if isclass(getattr(optim, attr_name)) and issubclass(getattr(optim, attr_name), optim.Optimizer):
        OPTIMIZER_REGISTRY.register(getattr(optim, attr_name))


def build_optimizer(cfg: CfgNode, params) -> optim.Optimizer:
    kwargs = dict(cfg.OPTIMIZER.PARAMETER)
    logger.debug("Parameter for optimizer %s is %s", cfg.OPTIMIZER.NAME, kwargs)
    return OPTIMIZER_REGISTRY.get(cfg.OPTIMIZER.NAME)(params, **kwargs)


for attr_name in optim.lr_scheduler.__dict__.keys():
    if isclass(getattr(optim.lr_scheduler, attr_name)) and \
            issubclass(getattr(optim.lr_scheduler, attr_name), getattr(optim.lr_scheduler, "_LRScheduler")):
        SCHEDULER_REGISTRY.register(getattr(optim.lr_scheduler, attr_name))


def build_scheduler(cfg: CfgNode, optimizer: optim.Optimizer) -> optim.lr_scheduler.ExponentialLR:
    if cfg.SCHEDULER.NAME is not None:
        kwargs = {}
        if cfg.SCHEDULER.NAME in cfg.SCHEDULER:
            kwargs = cfg.SCHEDULER[cfg.SCHEDULER.NAME]
        logger.debug("Parameter for scheduler {} is {}".format(cfg.SCHUEDULER.NAME, kwargs))
        return SCHEDULER_REGISTRY.get(cfg.SCHEDULER.NAME)(optimizer, **kwargs)
    else:
        logger.debug("Scheduler is not specified.")  # and return None

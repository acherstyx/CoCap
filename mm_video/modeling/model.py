# -*- coding: utf-8 -*-
# @Time    : 2022/11/12 21:57
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : model.py

import logging

import torch
import torch.nn as nn
import torch.distributed as dist

from fvcore.common.config import CfgNode
from fvcore.common.registry import Registry
from typing import AnyStr, Dict

MODEL_REGISTRY = Registry("MODEL")

logger = logging.getLogger(__name__)

__all__ = ["MODEL_REGISTRY", "BaseModule", "build_model"]


def build_model(cfg: CfgNode):
    logger.debug("Building the model from the configuration...")
    assert cfg.MODEL.NAME is not None, "specify a model to load in config: MODEL.NAME"
    assert cfg.MODEL.PARALLELISM.lower() in {"dp", "ddp", "fsdp"}, "MODEL.PARALLELISM should be one of {ddp, fsdp}"
    model = MODEL_REGISTRY.get(cfg.MODEL.NAME)(cfg)
    logger.debug("Successfully built the model from the configuration.")
    # model parallelism
    if cfg.SYS.MULTIPROCESS:
        device_id = cfg.SYS.GPU_DEVICES[dist.get_rank() % cfg.SYS.NUM_GPU]
        logger.debug("Moving model to device: %s...", device_id)
        model.to(device_id)
        logger.debug("Model is moved to device: %s", device_id)
        if cfg.MODEL.PARALLELISM.lower() == "ddp":
            logger.debug("Building DistributedDataParallel, check whether the program is hanging...")
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[cfg.SYS.GPU_DEVICES[dist.get_rank() % cfg.SYS.NUM_GPU]],
                output_device=cfg.SYS.GPU_DEVICES[dist.get_rank() % cfg.SYS.NUM_GPU],
                find_unused_parameters=cfg.MODEL.DDP.FIND_UNUSED_PARAMETERS
            )
        elif cfg.MODEL.PARALLELISM.lower() == "fsdp":
            logger.debug("Building FullyShardedDataParallel, check whether the program is hanging...")
            raise NotImplementedError("FSPD is not supported yet.")
        else:
            raise RuntimeError(f"Model parallelism '{cfg.MODEL.PARALLELISM}' is not supported!")
    elif cfg.SYS.NUM_GPU > 0:
        model = model.cuda()
        model = nn.parallel.DataParallel(model)
    logger.debug("Model build finished.")
    return model


class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: Dict[AnyStr, torch.Tensor]):
        raise NotImplementedError

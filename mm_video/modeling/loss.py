# -*- coding: utf-8 -*-
# @Time    : 2022/11/13 01:58
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : loss.py

import logging
import torch.nn as nn
from fvcore.common.registry import Registry
from fvcore.common.config import CfgNode

__all__ = ["LOSS_REGISTRY", "LossBase", "build_loss"]

LOSS_REGISTRY = Registry("LOSS")

logger = logging.getLogger(__name__)


class LossBase(nn.Module):
    def __init__(self, cfg: CfgNode):
        super(LossBase, self).__init__()
        self.cfg = cfg

    def forward(self, inputs, outputs):
        raise NotImplementedError


def build_loss(cfg):
    if cfg.LOSS.NAME is not None:
        loss_builder = LOSS_REGISTRY.get(cfg.LOSS.NAME)
        if issubclass(loss_builder, LossBase):
            return loss_builder(cfg)
        else:
            return loss_builder()
    else:
        logger.warning("Loss is not specified!")
        return None


@LOSS_REGISTRY.register()
class MultiObjectiveLoss(LossBase):
    def __init__(self, cfg):
        super(MultiObjectiveLoss, self).__init__(cfg)
        self.losses_name = cfg.LOSS.MultiObjectiveLoss.LOSSES
        self.losses = [LOSS_REGISTRY.get(n)(cfg) for n in self.losses_name]
        self.weight = cfg.LOSS.MultiObjectiveLoss.WEIGHT \
            if cfg.LOSS.MultiObjectiveLoss.WEIGHT is not None else [1] * len(self.losses)

    def forward(self, inputs, outputs):
        loss_meta = {}
        for cur_loss_func, cur_weight, cur_loss_func_name in zip(self.losses, self.weight, self.losses_name):
            loss_meta[cur_loss_func_name] = cur_loss_func(inputs, outputs) * cur_weight
        return loss_meta

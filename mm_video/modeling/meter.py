# -*- coding: utf-8 -*-
# @Time    : 2022/11/13 01:50
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : meter.py

import logging
import torch

from torch.utils.tensorboard import SummaryWriter
from fvcore.common.config import CfgNode
from fvcore.common.registry import Registry

METER_REGISTRY = Registry("METER")
logger = logging.getLogger(__name__)


class MeterBase(object):
    """
    Interface
    """

    def __init__(self, cfg: CfgNode, writer: SummaryWriter, mode: str):
        """
        build meter
        :param cfg: config
        :param writer: initialized tensorboard summary writer
        :param mode: train, val or test mode, for tagging tensorboard, etc.
        """
        assert mode in ["train", "test", "val"], f"mode is invalid: {mode}"
        self.cfg = cfg
        self.writer = writer
        self.mode = mode

    def set_mode(self, mode: str):
        assert mode in ["train", "test", "val"], f"mode is invalid: {mode}"
        self.mode = mode

    @torch.no_grad()
    def update(self, inputs, outputs, global_step=None):
        """
        call on each step
        update inner status based on the input
        :param inputs: the dataloader outputs/model inputs
        :param outputs: the model outputs
        :param global_step: global step, use `self.step` if step is None
        """
        raise NotImplementedError

    @torch.no_grad()
    def summary(self, epoch):
        """
        call at the end of the epoch
        """
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


def build_meter(cfg: CfgNode, writer: SummaryWriter, mode: str):
    if cfg.METER.NAME is not None and cfg.METER.NAME:
        logger.debug("Use meter: %s", cfg.METER.NAME)
        return METER_REGISTRY.get(cfg.METER.NAME)(cfg=cfg, writer=writer, mode=mode)
    else:
        logger.warning("Meter is not specified!")
        return DummyMeter(cfg=cfg, writer=writer, mode=mode)


class DummyMeter(MeterBase):

    def update(self, inputs, outputs, n=None, global_step=None):
        pass

    def summary(self, epoch):
        pass

    def reset(self):
        pass

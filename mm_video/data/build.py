# -*- coding: utf-8 -*-
# @Time    : 2022/11/12 21:46
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : build.py

import logging
from typing import *

from fvcore.common.config import CfgNode
from fvcore.common.registry import Registry

import torch.distributed as dist
from torch.utils import data

DATASET_REGISTRY = Registry("DATASET")
COLLATE_FN_REGISTER = Registry("COLLATE_FN")

logger = logging.getLogger(__name__)


def build_loader(cfg: CfgNode, mode=("train", "test", "val")):
    logger.debug("Building dataloader...")
    assert cfg.DATA.DATASET.NAME is not None, "specify a dataset to load in config: DATASET.NAME"

    if not isinstance(mode, tuple):
        mode = (mode,)

    dataset_builder: Callable[[CfgNode, str], data.Dataset] = DATASET_REGISTRY.get(cfg.DATA.DATASET.NAME)
    set_list = [dataset_builder(cfg, _mode) for _mode in mode]

    if cfg.SYS.MULTIPROCESS:
        sampler_list = [data.distributed.DistributedSampler(dataset,
                                                            rank=dist.get_rank(),
                                                            shuffle=cfg.DATA.LOADER.SHUFFLE)
                        for _mode, dataset in zip(mode, set_list)]
    else:
        sampler_list = [None for _mode in mode]
    collate_fn = COLLATE_FN_REGISTER.get(
        str(cfg.DATA.LOADER.COLLATE_FN)) if cfg.DATA.LOADER.COLLATE_FN is not None else None
    kwargs_default = {
        "batch_size": cfg.DATA.LOADER.BATCH_SIZE,
        "num_workers": cfg.DATA.LOADER.NUM_WORKERS,
        "pin_memory": True,
        "persistent_workers": False,
        "shuffle": False if cfg.SYS.MULTIPROCESS else cfg.DATA.LOADER.SHUFFLE,
        "prefetch_factor": cfg.DATA.LOADER.PREFETCH_FACTOR,
        "collate_fn": collate_fn,
        "multiprocessing_context": cfg.DATA.LOADER.MULTIPROCESSING_CONTEXT if cfg.DATA.LOADER.NUM_WORKERS else None
    }
    loader_list = [data.DataLoader(dataset=_dataset, sampler=_sampler, **kwargs_default)
                   for _mode, _dataset, _sampler in zip(mode, set_list, sampler_list)]

    logger.debug("Dataloader build finished.")
    if cfg.SYS.MULTIPROCESS and all(sampler is not None for sampler in sampler_list):
        res = {}
        res.update({_mode: loader for _mode, loader in zip(mode, loader_list)})
        res.update({f"{_mode}_sampler": sampler for _mode, sampler in zip(mode, sampler_list)})
        return res
    else:
        return {_mode: loader for _mode, loader in zip(mode, loader_list)}

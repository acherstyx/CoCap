# -*- coding: utf-8 -*-
# @Time    : 2022/11/12 22:31
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : train_utils.py

import datetime
import hashlib
import os
import pickle
import typing
import torch
import time
import random
import itertools
import numpy as np
import logging
from typing import *

import torch.distributed as dist
from fvcore.common.config import CfgNode

logger = logging.getLogger(__name__)


class CudaPreFetcher:
    def __init__(self, data_loader):
        self.dl = data_loader
        self.loader = iter(data_loader)
        self.stream = torch.cuda.Stream()
        self.batch = None

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            self.batch = self.cuda(self.batch)

    @staticmethod
    def cuda(x: typing.Any):
        if isinstance(x, list) or isinstance(x, tuple):
            return [CudaPreFetcher.cuda(i) for i in x]
        elif isinstance(x, dict):
            return {k: CudaPreFetcher.cuda(v) for k, v in x.items()}
        elif isinstance(x, torch.Tensor):
            return x.cuda(non_blocking=True)
        else:
            return x

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch

    def __iter__(self):
        self.preload()
        return self

    def __len__(self):
        return len(self.dl)


def manual_seed(cfg: CfgNode):
    if cfg.SYS.DETERMINISTIC:
        torch.manual_seed(cfg.SYS.SEED)
        random.seed(cfg.SYS.SEED)
        np.random.seed(cfg.SYS.SEED)
        torch.cuda.manual_seed(cfg.SYS.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        logger.debug("Manual seed is set")
    else:
        logger.warning("Manual seed is not used")


def init_distributed(proc: int, cfg: CfgNode):
    if cfg.SYS.MULTIPROCESS:  # initialize multiprocess
        word_size = cfg.SYS.NUM_GPU * cfg.SYS.NUM_SHARDS
        rank = cfg.SYS.NUM_GPU * cfg.SYS.SHARD_ID + proc
        dist.init_process_group(backend="nccl", init_method=cfg.SYS.INIT_METHOD, world_size=word_size, rank=rank)
        torch.cuda.set_device(cfg.SYS.GPU_DEVICES[proc])


def save_config(cfg: CfgNode):
    if not dist.is_initialized() or dist.get_rank() == 0:
        config_file = os.path.join(cfg.LOG.DIR, f"config_{get_timestamp()}.yaml")
        with open(config_file, "w") as f:
            f.write(cfg.dump())
        logger.debug("config is saved to %s", config_file)


def get_timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')


def gather_object_multiple_gpu(list_object: List[Any], backend: AnyStr = "nccl", shared_folder=None,
                               retry=600, sleep=0.1):
    """
    gather a list of something from multiple GPU
    """
    assert type(list_object) == list, "`list_object` only receive list."
    assert backend in ["nccl", "filesystem"]
    if backend == "nccl":
        gathered_objects = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_objects, list_object)
        return list(itertools.chain(*gathered_objects))
    else:
        assert shared_folder is not None, "`share_folder` should be set if backend is `filesystem`"
        os.makedirs(shared_folder, exist_ok=True)
        uuid = torch.randint(99999999, 99999999999, size=(1,), dtype=torch.long).cuda()
        dist.all_reduce(uuid)
        uuid = hex(uuid.cpu().item())[-8:]
        with open(os.path.join(shared_folder, f"{uuid}_rank_{dist.get_rank():04d}.pkl"), "wb") as f:
            data = pickle.dumps(list_object)
            f.write(data)
        with open(os.path.join(shared_folder, f"{uuid}_rank_{dist.get_rank():04d}.md5"), "wb") as f:
            checksum = hashlib.md5(data).hexdigest()
            pickle.dump(checksum, f)
        gathered_list = []
        dist.barrier()
        for rank in range(dist.get_world_size()):
            data_filename = os.path.join(shared_folder, f"{uuid}_rank_{rank:04d}.pkl")
            checksum_filename = os.path.join(shared_folder, f"{uuid}_rank_{rank:04d}.md5")
            data = None
            for _ in range(retry):
                time.sleep(sleep)
                try:
                    if not os.path.exists(data_filename):
                        continue
                    if not os.path.exists(checksum_filename):
                        continue
                    raw_data = open(data_filename, "rb").read()
                    checksum = pickle.load(open(checksum_filename, "rb"))
                    assert checksum == hashlib.md5(raw_data).hexdigest()
                    data = pickle.loads(raw_data)
                    break
                except Exception:
                    pass
            assert data is not None, f"Gather from filesystem failed after retry for {retry} times."
            gathered_list.extend(data)
        dist.barrier()
        return gathered_list

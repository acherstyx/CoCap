# -*- coding: utf-8 -*-
# @Time    : 2022/11/12 22:31
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : train_utils.py

import hashlib
import os
import pickle
import time
import random
import itertools
import numpy as np
import math
import logging
from typing import *

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.fsdp.fully_sharded_data_parallel import _get_grad_norm

from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SystemConfig:
    # deterministic
    deterministic: bool = True
    seed: int = 222


def cuda(x: Union[torch.Tensor, Dict, List]):
    """
    Recursively move to GPU
    :param x:
    :return:
    """
    if isinstance(x, list) or isinstance(x, tuple):
        return [cuda(i) for i in x]
    elif isinstance(x, dict):
        return {k: cuda(v) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.cuda(non_blocking=True)
    else:
        return x


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
    def cuda(x: Any):
        return cuda(x)

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


def manual_seed(cfg: SystemConfig):
    if cfg.deterministic:
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        logger.debug("Manual seed is set to %s", cfg.seed)
    else:
        logger.warning("Manual seed is not used")


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


def conditional_gather_object_multiple_gpu(
        list_object: List[Any],
        backend: AnyStr = "nccl", shared_folder=None, retry=600, sleep=0.1
):
    if dist.is_initialized():
        return gather_object_multiple_gpu(
            list_object=list_object,
            backend=backend,
            shared_folder=shared_folder,
            retry=retry,
            sleep=sleep
        )
    else:
        return list_object


# from peft: https://github.com/huggingface/peft/blob/main/src/peft/peft_model.py
def get_trainable_parameters(model: torch.nn.Module):
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params = 0
    trainable_params_names = []
    all_param = 0
    for param_name, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
            trainable_params_names.append(param_name)

    return trainable_params, all_param, trainable_params_names


def compute_total_gradient_norm(model: nn.Module, norm_type: float = 2.0) -> torch.Tensor:
    """
    Compute gradient norm over all parameters.
    This needs to be called on all ranks since it uses collective communications.
    """
    norm_type = float(norm_type)
    if isinstance(model, FullyShardedDataParallel):
        # The logic for computing the total gradient norm for FSDP is adopted from the `clip_grad_norm_` method
        # of FullyShardedDataParallel

        # If every FSDP instance uses `NO_SHARD`, then we can directly use
        # the normal `_get_grad_norm` to calculate the total gradient norm
        all_no_shard = all(
            not handle.uses_sharded_strategy for handle in model._all_handles
        )
        if all_no_shard:
            return _get_grad_norm(model.parameters(), norm_type=norm_type)
        sharded_params = set()
        nonsharded_params = set()  # `NO_SHARD` or not FSDP-managed
        grads: List[torch.Tensor] = []
        for handle in model._all_handles:
            target_set = (
                sharded_params if handle.uses_sharded_strategy else nonsharded_params
            )
            if handle._use_orig_params:
                for param in handle.flat_param._params:
                    target_set.add(param)
                    if param.grad is not None:
                        grads.append(param.grad)
            else:
                target_set.add(handle.flat_param)
                if handle.flat_param.grad is not None:
                    grads.append(handle.flat_param.grad)
        for param in model.parameters():
            not_fsdp_managed = (
                    param not in sharded_params and param not in nonsharded_params
            )
            if not_fsdp_managed:
                nonsharded_params.add(param)
                if param.grad is not None:
                    grads.append(param.grad)
        # Compute local norms (forced to be in FP32)
        local_sharded_norm = _get_grad_norm(sharded_params, norm_type).to(
            model.compute_device
        )
        local_nonsharded_norm = _get_grad_norm(nonsharded_params, norm_type).to(
            model.compute_device
        )
        # Reconstruct the total gradient norm depending on the norm type
        if norm_type == math.inf:
            total_norm = torch.maximum(local_sharded_norm, local_nonsharded_norm)
            dist.all_reduce(
                total_norm, op=torch.distributed.ReduceOp.MAX, group=model.process_group
            )
        else:
            total_norm = local_sharded_norm ** norm_type
            dist.all_reduce(total_norm, group=model.process_group)
            # All-reducing the local non-sharded norm would count it an extra
            # world-size-many times
            total_norm += local_nonsharded_norm ** norm_type
            total_norm = total_norm ** (1.0 / norm_type)
        if model.cpu_offload.offload_params:
            total_norm = total_norm.cpu()
        return total_norm.to(torch.float32)
    else:
        parameters = [p for p in model.parameters() if p.grad is not None]
        return _get_grad_norm(parameters, norm_type=norm_type)


def get_world_size() -> int:
    """
    Get world size from environment variable set by `torchrun`.

    """
    return int(os.environ.get("WORLD_SIZE", 1))

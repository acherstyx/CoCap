# -*- coding: utf-8 -*-
# @Time    : 2022/11/14 03:02
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : torchrun_net.py

import os
import torch
import torch.distributed

from mm_video.config.base import get_config
from mm_video.trainer.build import build_trainer
from mm_video.utils.logging import setup_logging, show_registry
from mm_video.utils.train_utils import manual_seed, get_timestamp, save_config


def main():
    cfg = get_config()
    # overwrite some config for torchrun
    cfg.defrost()
    cfg.SYS.MULTIPROCESS = True
    cfg.freeze()

    run_trainer(cfg=cfg)


def run_trainer(cfg):
    print(f"{get_timestamp()} => Run trainer")
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    setup_logging(cfg)
    save_config(cfg)
    manual_seed(cfg)
    show_registry()

    trainer = build_trainer(cfg)
    trainer.train()
    print(f"{get_timestamp()} => Finished!")


if __name__ == '__main__':
    main()

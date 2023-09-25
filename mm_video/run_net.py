# -*- coding: utf-8 -*-
# @Time    : 2022/11/14 03:02
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : run_net.py

import torch.multiprocessing as mp

from mm_video.config.base import get_config
from mm_video.trainer.build import build_trainer
from mm_video.utils.logging import setup_logging, show_registry
from mm_video.utils.train_utils import init_distributed, manual_seed, get_timestamp, save_config


def main():
    cfg = get_config()
    print(f"{get_timestamp()} => Run trainer")
    if cfg.SYS.MULTIPROCESS:
        mp.spawn(run_trainer, args=(cfg,), nprocs=cfg.SYS.NUM_GPU)
    else:
        run_trainer(proc=0, cfg=cfg)
    print(f"{get_timestamp()} => Finished!")


def run_trainer(proc, cfg):
    init_distributed(proc, cfg)
    setup_logging(cfg)
    save_config(cfg)
    manual_seed(cfg)
    show_registry()

    trainer = build_trainer(cfg)
    trainer.train()


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
# @Time    : 2022/11/12 22:32
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : logging.py


import os
import logging
import colorlog
import torch.distributed as dist

level_dict = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "notset": logging.NOTSET
}


# noinspection SpellCheckingInspection
def setup_logging(cfg):
    # log file
    if len(str(cfg.LOG.LOGGER_FILE).split(".")) == 2:
        file_name, extension = str(cfg.LOG.LOGGER_FILE).split(".")
        log_file_debug = os.path.join(cfg.LOG.DIR, f"{file_name}_debug.{extension}")
        log_file_info = os.path.join(cfg.LOG.DIR, f"{file_name}_info.{extension}")
    elif len(str(cfg.LOG.LOGGER_FILE).split(".")) == 1:
        file_name = cfg.LOG.LOGGER_FILE
        log_file_debug = os.path.join(cfg.LOG.DIR, f"{file_name}_debug")
        log_file_info = os.path.join(cfg.LOG.DIR, f"{file_name}_info")
    else:
        raise ValueError("cfg.LOG.LOGGER_FILE is invalid: %s", cfg.LOG.LOGGER_FILE)
    logger = logging.getLogger(__name__.split(".")[0])
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    formatter = logging.Formatter(
        f"[%(asctime)s][%(levelname)s]{f'[Rank {dist.get_rank()}]' if dist.is_initialized() else ''} "
        "%(filename)s: %(lineno)3d: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )
    color_formatter = colorlog.ColoredFormatter(
        f"%(log_color)s%(bold)s%(levelname)-8s%(reset)s"
        f"%(log_color)s[%(asctime)s]"
        f"{f'[Rank {dist.get_rank()}]' if dist.is_initialized() else ''}"
        "[%(filename)s: %(lineno)3d]:%(reset)s "
        "%(message)s",
        datefmt="%m/%d %H:%M:%S",
    )
    # log file
    if os.path.dirname(log_file_debug):  # dir name is not empty
        os.makedirs(os.path.dirname(log_file_debug), exist_ok=True)
    # console
    handler_console = logging.StreamHandler()
    assert cfg.LOG.LOGGER_CONSOLE_LEVEL.lower() in level_dict, \
        f"Log level {cfg.LOG.LOGGER_CONSOLE_LEVEL} is invalid"
    handler_console.setLevel(level_dict[cfg.LOG.LOGGER_CONSOLE_LEVEL.lower()])
    handler_console.setFormatter(color_formatter if cfg.LOG.LOGGER_CONSOLE_COLORFUL else formatter)
    logger.addHandler(handler_console)
    # debug level
    handler_debug = logging.FileHandler(log_file_debug, mode="a")
    handler_debug.setLevel(logging.DEBUG)
    handler_debug.setFormatter(formatter)
    logger.addHandler(handler_debug)
    # info level
    handler_info = logging.FileHandler(log_file_info, mode="a")
    handler_info.setLevel(logging.INFO)
    handler_info.setFormatter(formatter)
    logger.addHandler(handler_info)

    logger.propagate = False


def show_registry():
    from mm_video.data.build import DATASET_REGISTRY, COLLATE_FN_REGISTER
    from mm_video.modeling.model import MODEL_REGISTRY
    from mm_video.modeling.optimizer import OPTIMIZER_REGISTRY
    from mm_video.modeling.loss import LOSS_REGISTRY
    from mm_video.modeling.meter import METER_REGISTRY

    logger = logging.getLogger(__name__)
    logger.debug(DATASET_REGISTRY)
    logger.debug(COLLATE_FN_REGISTER)
    logger.debug(MODEL_REGISTRY)
    logger.debug(OPTIMIZER_REGISTRY)
    logger.debug(LOSS_REGISTRY)
    logger.debug(METER_REGISTRY)

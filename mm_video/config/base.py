# -*- coding: utf-8 -*-
# @Time    : 2022/11/11 16:02
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : meter.py

import argparse
import logging
import os

import torch
from fvcore.common.config import CfgNode
from fvcore.common.registry import Registry
from ipaddress import ip_address, IPv4Address
import yaml
from yaml.loader import SafeLoader

logger = logging.getLogger(__name__)

CUSTOM_CONFIG_REGISTRY = Registry("CUSTOM_CONFIG")
CUSTOM_CONFIG_CHECK_REGISTRY = Registry("CUSTOM_CONFIG_CHECK")

_C = CfgNode()

# project info
_C.INFO = CfgNode()
_C.INFO.PROJECT_NAME = None
_C.INFO.EXPERIMENT_NAME = "experiment"

# system config
_C.SYS = CfgNode()
_C.SYS.MULTIPROCESS = False
_C.SYS.INIT_METHOD = "tcp://localhost:2222"
_C.SYS.NUM_GPU = torch.cuda.device_count()
_C.SYS.GPU_DEVICES = list(range(torch.cuda.device_count()))
_C.SYS.NUM_SHARDS = 1
_C.SYS.SHARD_ID = 0
_C.SYS.DETERMINISTIC = False
_C.SYS.SEED = 222

# log config
_C.LOG = CfgNode()
_C.LOG.DIR = None
_C.LOG.LOGGER_FILE = "logger.log"
_C.LOG.LOGGER_CONSOLE_LEVEL = "info"
_C.LOG.LOGGER_CONSOLE_COLORFUL = True

# build config for base dataloader
_C.DATA = CfgNode()
_C.DATA.DATASET = CfgNode()
_C.DATA.DATASET.NAME = None
_C.DATA.LOADER = CfgNode()
_C.DATA.LOADER.COLLATE_FN = None
_C.DATA.LOADER.BATCH_SIZE = 1
_C.DATA.LOADER.NUM_WORKERS = 0
_C.DATA.LOADER.SHUFFLE = False
_C.DATA.LOADER.PREFETCH_FACTOR = 2
_C.DATA.LOADER.MULTIPROCESSING_CONTEXT = "spawn"

# build config for base model
_C.MODEL = CfgNode()
_C.MODEL.NAME = None
_C.MODEL.PARALLELISM = "ddp"
_C.MODEL.DDP = CfgNode()
_C.MODEL.DDP.FIND_UNUSED_PARAMETERS = False

# optimizer
_C.OPTIMIZER = CfgNode()
_C.OPTIMIZER.NAME = "Adam"
_C.OPTIMIZER.PARAMETER = CfgNode(new_allowed=True)

# scheduler
_C.SCHEDULER = CfgNode()
_C.SCHEDULER.NAME = None

# build config for loss
_C.LOSS = CfgNode()
_C.LOSS.NAME = None
_C.LOSS.MultiObjectiveLoss = CfgNode()
_C.LOSS.MultiObjectiveLoss.LOSSES = []
_C.LOSS.MultiObjectiveLoss.WEIGHT = None

# build config for meter
_C.METER = CfgNode()
_C.METER.NAME = None

# trainer
_C.TRAINER = CfgNode()
_C.TRAINER.NAME = "TrainerBase"
# trainer base
_C.TRAINER.TRAINER_BASE = CfgNode()
_C.TRAINER.TRAINER_BASE.TEST_ENABLE = True
_C.TRAINER.TRAINER_BASE.TRAIN_ENABLE = True
_C.TRAINER.TRAINER_BASE.EPOCH = 50
_C.TRAINER.TRAINER_BASE.GRADIENT_ACCUMULATION_STEPS = 1
_C.TRAINER.TRAINER_BASE.RESUME = None
_C.TRAINER.TRAINER_BASE.AUTO_RESUME = False
_C.TRAINER.TRAINER_BASE.CLIP_NORM = None
_C.TRAINER.TRAINER_BASE.SAVE_FREQ = 1
_C.TRAINER.TRAINER_BASE.LOG_FREQ = 1
_C.TRAINER.TRAINER_BASE.AMP = False
_C.TRAINER.TRAINER_BASE.DEBUG = False
_C.TRAINER.TRAINER_BASE.WRITE_HISTOGRAM = False
_C.TRAINER.TRAINER_BASE.WRITE_PROFILER = False

base_config = _C.clone()
base_config.freeze()


def check_config(cfg: CfgNode):
    # default check config
    if cfg.LOG.DIR is None:
        info = [i for i in [cfg.INFO.PROJECT_NAME, cfg.INFO.EXPERIMENT_NAME] if i]
        if info:  # not empty
            cfg.LOG.DIR = os.path.join("log", "_".join(info))
        else:
            cfg.LOG.DIR = os.path.join("log", "default")
    assert cfg.MODEL.PARALLELISM.lower() in {"dp", "ddp", "fsdp"}, "MODEL.PARALLELISM should be one of {dp, ddp, fsdp}"
    assert not cfg.SYS.MULTIPROCESS or cfg.SYS.NUM_GPU > 0, "At least 1 GPU is required to enable ddp."
    assert cfg.TRAINER.TRAINER_BASE.GRADIENT_ACCUMULATION_STEPS > 0, "gradient accumulation step should greater than 0."
    assert cfg.LOSS.MultiObjectiveLoss.WEIGHT is None or \
           len(cfg.LOSS.MultiObjectiveLoss.WEIGHT) == len(cfg.LOSS.MultiObjectiveLoss.LOSSES)


def get_config():
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", help="Which project to run. The config will be loaded according to project. "
                                          "Load all registered configs if project is not set.",
                        default=None)
    parser.add_argument("--cfg", "-c",
                        help="path to the additional config file",
                        default=None,
                        type=str)
    parser.add_argument("--debug",
                        help="set trainer to debug mode",
                        action="store_true")
    parser.add_argument("opts",
                        help="see config/custom_config.py for all options",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    # build base config and custom config
    cfg = base_config.clone()
    cfg.defrost()
    add_arnold_ddp_env(cfg)
    # get project from args or config file, add custom config
    project_name = cfg.INFO.PROJECT_NAME
    if args.project is not None:
        project_name = args.project
    elif args.cfg is not None:
        cfg_base = args.cfg
        while True:
            cfg_from_file = yaml.load(open(cfg_base), SafeLoader)
            if "INFO" in cfg_from_file and "PROJECT_NAME" in cfg_from_file["INFO"]:
                project_name = cfg_from_file["INFO"]["PROJECT_NAME"]
                break
            elif "_BASE_" in cfg_from_file:
                cfg_base = os.path.join(os.path.dirname(cfg_base), cfg_from_file["_BASE_"])
            else:
                break
    print("Project:", project_name)
    add_custom_config(cfg, project_name)
    # merge config from args and config file
    if args.cfg is not None:
        cfg.merge_from_file(args.cfg)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    # apply debug option
    if args.debug:
        cfg.TRAINER.TRAINER_BASE.DEBUG = True
        cfg.LOG.LOGGER_CONSOLE_LEVEL = "debug"
    check_config(cfg)
    check_custom_config(cfg, project_name=project_name)
    cfg.freeze()
    return cfg


def dump(cfg: CfgNode, config_file: str):
    with open(config_file, "w") as f:
        f.write(cfg.dump())


def add_arnold_ddp_env(cfg: CfgNode):
    """
    Setup ddp configuration base on the environment variable in Arnold.
    """

    def is_ipv6(ip: str) -> bool:
        try:
            return False if type(ip_address(ip)) is IPv4Address else True
        except ValueError:
            return False

    cfg.SYS.NUM_SHARDS = int(os.getenv("ARNOLD_WORKER_NUM")) if os.getenv("ARNOLD_WORKER_NUM") is not None else 1
    cfg.SYS.SHARD_ID = int(os.getenv("ARNOLD_ID")) if os.getenv("ARNOLD_ID") is not None else 0
    master_address = os.getenv("METIS_WORKER_0_HOST") if \
        os.getenv("METIS_WORKER_0_HOST") is not None and os.getenv("WORKSPACE_ENVS_SET") is None else "localhost"
    master_port = int(os.getenv("METIS_WORKER_0_PORT").split(",")[0]) \
        if os.getenv("METIS_WORKER_0_PORT") is not None else 2222
    cfg.SYS.INIT_METHOD = f"tcp://{master_address}:{master_port}" if not is_ipv6(master_address) \
        else f"tcp://[{master_address}]:{master_port}"


def add_custom_config(cfg: CfgNode, project_name: str = None):
    assert project_name is None or type(project_name) == str, "`project_name` should be str or None"
    config_list = []
    if project_name is None:
        config_list = [cfg for _, cfg in list(CUSTOM_CONFIG_REGISTRY)]
    else:
        if f"{project_name}_config" in CUSTOM_CONFIG_REGISTRY:
            config_list = [CUSTOM_CONFIG_REGISTRY.get(f"{project_name}_config")]
        else:
            logger.warning(f"Project '{project_name}' do not have registered custom config. Proceeding without it.")
    for add_cfg in config_list:
        add_cfg(cfg)


def check_custom_config(cfg: CfgNode, project_name: str = None):
    assert project_name is None or type(project_name) == str, "`project_name` should be str or None"
    check_list = []
    if project_name is None:
        check_list = [cfg for _, cfg in list(CUSTOM_CONFIG_CHECK_REGISTRY)]
    else:
        if f"check_{project_name}_config" in CUSTOM_CONFIG_CHECK_REGISTRY:
            check_list = [CUSTOM_CONFIG_CHECK_REGISTRY.get(f"check_{project_name}_config")]
        else:
            logger.warning(f"Project '{project_name}' do not have registered config check. Proceeding without it.")
    for check_cfg in check_list:
        check_cfg(cfg)


if __name__ == '__main__':
    print(get_config())

# -*- coding: utf-8 -*-
# @Time    : 7/16/23
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : compressed_video.py

from fvcore.common.config import CfgNode
from ..base import CUSTOM_CONFIG_REGISTRY, CUSTOM_CONFIG_CHECK_REGISTRY


@CUSTOM_CONFIG_REGISTRY.register()
def compressed_video_config(cfg: CfgNode):
    # msrvtt
    cfg.DATA.DATASET.MSRVTT = CfgNode()
    cfg.DATA.DATASET.MSRVTT.VIDEO_ROOT = None
    cfg.DATA.DATASET.MSRVTT.METADATA = None
    cfg.DATA.DATASET.MSRVTT.VIDEO_READER = None
    cfg.DATA.DATASET.MSRVTT.MAX_FRAMES = None
    cfg.DATA.DATASET.MSRVTT.VIDEO_SIZE = None
    cfg.DATA.DATASET.MSRVTT.MAX_WORDS = None
    cfg.DATA.DATASET.MSRVTT.UNFOLD_SENTENCES = None
    # msvd
    cfg.DATA.DATASET.MSVD = CfgNode()
    cfg.DATA.DATASET.MSVD.VIDEO_ROOT = None
    cfg.DATA.DATASET.MSVD.METADATA = None
    cfg.DATA.DATASET.MSVD.VIDEO_READER = None
    cfg.DATA.DATASET.MSVD.MAX_FRAMES = None
    cfg.DATA.DATASET.MSVD.VIDEO_SIZE = None
    cfg.DATA.DATASET.MSVD.MAX_WORDS = None
    cfg.DATA.DATASET.MSVD.UNFOLD_SENTENCES = None
    # vatex
    cfg.DATA.DATASET.VATEX = CfgNode()
    cfg.DATA.DATASET.VATEX.VIDEO_ROOT = None
    cfg.DATA.DATASET.VATEX.METADATA = None
    cfg.DATA.DATASET.VATEX.VIDEO_READER = None
    cfg.DATA.DATASET.VATEX.MAX_FRAMES = None
    cfg.DATA.DATASET.VATEX.VIDEO_SIZE = None
    cfg.DATA.DATASET.VATEX.MAX_WORDS = None
    cfg.DATA.DATASET.VATEX.UNFOLD_SENTENCES = None

    # h265 config for h265 video readers
    cfg.CV_CONFIG = CfgNode()
    cfg.CV_CONFIG.NUM_GOP = None
    cfg.CV_CONFIG.NUM_MV = None
    cfg.CV_CONFIG.NUM_RES = None
    cfg.CV_CONFIG.WITH_RESIDUAL = None
    cfg.CV_CONFIG.USE_PRE_EXTRACT = None
    cfg.CV_CONFIG.SAMPLE = None

    cfg.MODEL.COCAP = CfgNode()
    cfg.MODEL.COCAP.PRETRAINED_CLIP = None
    cfg.MODEL.COCAP.MOTION_DROPOUT_PROB = None
    cfg.MODEL.COCAP.RESIDUAL_DROPOUT_PROB = None
    cfg.MODEL.COCAP.MOTION_ENCODER = CfgNode()
    cfg.MODEL.COCAP.MOTION_ENCODER.N_LAYERS = None
    cfg.MODEL.COCAP.MOTION_ENCODER.PATCH_SIZE = None
    cfg.MODEL.COCAP.MOTION_ENCODER.N_HEADS = None
    cfg.MODEL.COCAP.RESIDUAL_ENCODER = CfgNode()
    cfg.MODEL.COCAP.RESIDUAL_ENCODER.N_LAYERS = None
    cfg.MODEL.COCAP.RESIDUAL_ENCODER.PATCH_SIZE = None
    cfg.MODEL.COCAP.RESIDUAL_ENCODER.N_HEADS = None
    cfg.MODEL.COCAP.ACTION_ENCODER = CfgNode()
    cfg.MODEL.COCAP.ACTION_ENCODER.N_LAYERS = None
    cfg.MODEL.COCAP.ACTION_ENCODER.N_HEADS = None
    cfg.MODEL.COCAP.TASK_TYPE = None

    cfg.TRAINER.CAPTION_TRAINER = CfgNode()
    cfg.TRAINER.CAPTION_TRAINER.TASK_TYPE = None
    cfg.TRAINER.CAPTION_TRAINER.CLIP_LR = None
    cfg.TRAINER.CAPTION_TRAINER.LR_DECAY_GAMMA = None


@CUSTOM_CONFIG_CHECK_REGISTRY.register()
def check_compressed_video_config(cfg: CfgNode):
    assert cfg.CV_CONFIG.NUM_MV == cfg.CV_CONFIG.NUM_RES, \
        "The number of motion vectors and residuals in each GOP should be equal"
    assert cfg.TRAINER.CAPTION_TRAINER.TASK_TYPE == cfg.MODEL.COCAP.TASK_TYPE
    assert cfg.TRAINER.CAPTION_TRAINER.TASK_TYPE == "captioning"
    assert cfg.MODEL.COCAP.PRETRAINED_CLIP in ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64',
                                               'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']

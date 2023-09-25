# -*- coding: utf-8 -*-
# @Time    : 8/5/23
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : test_modeling.py

import unittest

import torch

from mm_video.modeling.compressed_video.modeling import CaptionHead, CoCap

from fvcore.common.config import CfgNode


class TestCaptionHead(unittest.TestCase):
    def test_build(self):
        model = CaptionHead.from_pretrained(
            pretrained_clip_name_or_path="ViT-B/16",
            max_v_len=16, max_t_len=22
        )
        print(model)


class TestCoCap(unittest.TestCase):
    cfg = CfgNode()
    cfg.MODEL = CfgNode()
    cfg.MODEL.COCAP = CfgNode()
    cfg.MODEL.COCAP.TASK_TYPE = "captioning"
    cfg.MODEL.COCAP.PRETRAINED_CLIP = "ViT-B/16"
    cfg.MODEL.COCAP.MOTION_DROPOUT_PROB = 0.2
    cfg.MODEL.COCAP.RESIDUAL_DROPOUT_PROB = 0.2
    cfg.MODEL.COCAP.MOTION_ENCODER = CfgNode()
    cfg.MODEL.COCAP.MOTION_ENCODER.PATCH_SIZE = 8
    cfg.MODEL.COCAP.MOTION_ENCODER.N_LAYERS = 2
    cfg.MODEL.COCAP.MOTION_ENCODER.N_HEADS = 8
    cfg.MODEL.COCAP.RESIDUAL_ENCODER = CfgNode()
    cfg.MODEL.COCAP.RESIDUAL_ENCODER.N_LAYERS = 2
    cfg.MODEL.COCAP.RESIDUAL_ENCODER.PATCH_SIZE = 64
    cfg.MODEL.COCAP.RESIDUAL_ENCODER.N_HEADS = 12
    cfg.MODEL.COCAP.ACTION_ENCODER = CfgNode()
    cfg.MODEL.COCAP.ACTION_ENCODER.N_LAYERS = 1
    cfg.MODEL.COCAP.ACTION_ENCODER.N_HEADS = 8

    cfg.CV_CONFIG = CfgNode()
    cfg.CV_CONFIG.NUM_MV = 59
    cfg.CV_CONFIG.NUM_GOP = 8

    def test_build(self):
        model = CoCap(self.cfg)
        print(model)

    def test_forward(self):
        model = CoCap(self.cfg)

        outputs = model(
            {
                "video": {
                    "iframe": torch.randn(8, 8, 3, 224, 224),
                    "motion_vector": torch.randn(8, 8, 59, 4, 56, 56),
                    "residual": torch.randint(0, 255, size=(8, 8, 59, 3, 224, 224)),
                    "type_ids_mv": torch.randint(0, 1, size=(8, 8, 59))
                },
                "input_ids": torch.randint(0, 1000, size=(8, 77)),
                "input_mask": torch.ones((8, 77), dtype=torch.long),
            }
        )
        print(outputs["prediction_scores"].shape)
        print(outputs["visual_output"]["feature_context"].shape)
        print(outputs["visual_output"]["feature_action"].shape)
        print(outputs["visual_output"]["iframe_attention_map"].shape)
        print(outputs["visual_output"]["motion_vector_attention_map"].shape)
        print(outputs["visual_output"]["residual_attention_map"].shape)


if __name__ == '__main__':
    unittest.main()

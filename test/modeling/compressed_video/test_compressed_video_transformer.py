# -*- coding: utf-8 -*-
# @Time    : 8/2/23
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : test_compressed_video_transformer.py

import unittest
import torch

from mm_video.modeling.compressed_video.compressed_video_transformer import CompressedVideoTransformer


class TestCompressedVideoTransformer(unittest.TestCase):
    def test_from_pretrained(self):
        model = CompressedVideoTransformer.from_pretrained(
            pretrained_clip_name_or_path="ViT-B/16",
            motion_patch_size=8, motion_layers=2, motion_heads=8,
            residual_patch_size=64, residual_layers=2, residual_heads=12,
            action_layers=1, action_heads=8, n_bp=59
        )
        print(model)

    def test_forward(self):
        model = CompressedVideoTransformer.from_pretrained(
            pretrained_clip_name_or_path="ViT-B/16",
            motion_patch_size=8, motion_layers=2, motion_heads=8,
            residual_patch_size=64, residual_layers=2, residual_heads=12,
            action_layers=1, action_heads=8, n_bp=59
        )

        output = model(
            iframe=torch.rand(5, 8, 3, 224, 224),
            motion=torch.rand(5, 8, 59, 4, 56, 56),
            residual=torch.rand(5, 8, 59, 3, 224, 224),
            bp_type_ids=torch.randint(0, 1, (5, 8, 59))
        )
        for k, v in output.items():
            print(k, v.shape)


if __name__ == '__main__':
    unittest.main()

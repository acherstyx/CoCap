# -*- coding: utf-8 -*-
# @Time    : 2022/12/12 12:36
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : test_video_readers.py

import os
import unittest
import cv_reader
import numpy as np
import matplotlib.pyplot as plt
import flow_vis
import tqdm

from mm_video.data.datasets.compressed_video.video_readers import read_frames_compressed_domain


class TestCVReader(unittest.TestCase):
    """
    test case for basic video reader
    """
    video = "dataset/msrvtt/videos_h264_keyint_60/video0.mp4"

    def test_cv_reader(self):
        """read video and print a brief view of data and visualize"""
        data = cv_reader.read_video(self.video)
        print("Number of frames: {}".format(len(data)))
        for k, v in data[0].items():
            if type(v) is np.ndarray:
                print(k, v.shape, v.reshape(-1)[:20])
            else:
                print(k, v)

        output_dir = "test_output/data/dataset/compressed_video/test_video_readers/visualize"
        os.makedirs(output_dir, exist_ok=True)
        for frame_idx in tqdm.tqdm(list(range(len(data)))):
            plt.imsave(os.path.join(output_dir, f"residual_{frame_idx:05d}.jpg"),
                       data[frame_idx]["residual"])
            plt.imsave(os.path.join(output_dir, f"motion_vector_{frame_idx:05d}.jpg"),
                       flow_vis.flow_to_color(data[frame_idx]["motion_vector"][..., :2]))


class TestVideoReader(unittest.TestCase):
    video = "dataset/msrvtt/videos_h264_keyint_60/video0.mp4"

    def test_read_frames_compressed_domain(self):
        data, is_success = read_frames_compressed_domain(
            video_path=self.video,
            resample_num_gop=8, resample_num_mv=59, resample_num_res=59,
            with_residual=True, with_bp_rgb=False, pre_extract=False, sample="rand"
        )

        print(is_success)
        for k, v in data.items():
            print(f"{k}: {v.shape}")


if __name__ == '__main__':
    unittest.main()

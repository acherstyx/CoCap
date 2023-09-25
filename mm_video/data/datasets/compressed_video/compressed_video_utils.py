# -*- coding: utf-8 -*-
# @Time    : 2022/12/9 20:42
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : compressed_video_utils.py

import numpy as np
from mm_video.utils.image import byte_imread, byte_imwrite


def serialize(data_to_serialize, quality):
    """
    Compress rgb and residual by converting to JPEG format
    :param data_to_serialize: python dict
    :param quality: quality parameter for JPEG format
    :return:
    """
    data = {}
    for k, v in data_to_serialize.items():
        if k in ["rgb_full", "rgb_gop", "residual"]:
            data[k] = [byte_imwrite(img, quality=quality) for img in v]
        else:
            data[k] = v
    return data


def deserialize(serialized_data):
    """
    Reverse version of serialize()
    :param serialized_data:
    :return:
    """
    data = {}
    for k, v in serialized_data.items():
        if k in ["rgb_full", "rgb_gop", "residual"]:
            data[k] = [np.array(byte_imread(img)) for img in v]
        else:
            data[k] = v
    return data

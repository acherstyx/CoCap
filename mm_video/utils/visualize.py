# -*- coding: utf-8 -*-
# @Time    : 9/5/23
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : visualize.py

import torch
import numpy as np

from torchvision.transforms.functional import normalize
from PIL import Image


def _convert_to_numpy_array(data):
    # convert any type to numpy array
    if isinstance(data, np.ndarray):
        pass
    elif isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif isinstance(data, Image.Image):
        data = np.array(data)
    else:
        raise ValueError("Data type is not supported to convert to numpy array: {}".format(type(data)))
    return data


def _convert_to_torch_tensor(data):
    # convert any type tot torch tensor
    if isinstance(data, torch.Tensor):
        pass
    elif isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    else:
        raise ValueError("Data type is not supported to convert to torch.Tensor: {}".format(type(data)))
    return data.detach().cpu()


def inv_normalize(image_or_video, mean, std):
    # convert to torch.Tensor
    data = _convert_to_torch_tensor(image_or_video)
    data = normalize(data, mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std])
    return data.numpy()

# -*- coding: utf-8 -*-
# @Time    : 2022/11/7 14:51
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : transforms.py

import einops
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from typing import Sequence

__all__ = ["DictRandomResizedCrop", "DictCenterCrop", "DictResize", "DictNormalize"]

"""
Custom transforms
"""

IMAGE_KEYS = ["iframe", "residual", "motion_vector"]
NORMALIZE_KEYS = ["iframe"]


# iframe: ..., c, h, w
# motion vector: n_gop, n_mv, c, h, w
# residual: n_gop, n_res, c, h, w


class DictRandomResizedCrop(transforms.RandomResizedCrop):
    """apply same random resized crop to the items in dict"""

    def forward(self, img: dict):
        assert any(k in img for k in IMAGE_KEYS), "input do not contain any valid image"
        for k in IMAGE_KEYS:
            if k in img:
                i, j, h, w = self.get_params(img[k], self.scale, self.ratio)
                break

        ret = {}
        for k, v in img.items():
            if k not in IMAGE_KEYS:
                ret[k] = v
                continue
            if len(v.shape) == 5:  # handle extra dimension for GOP
                num_gop = v.size(0)
                num_frame = v.size(1)
                v = einops.rearrange(v, "num_gop num_frame c h w->(num_gop num_frame) c h w")
                if k == "motion_vector":
                    if isinstance(self.size, Sequence) and len(self.size) == 2:
                        mv_size = (self.size[0] // 4, self.size[1] // 4)
                    elif isinstance(self.size, int):
                        mv_size = self.size // 4
                    else:
                        raise ValueError("Image size is not supported: {}".format(self.size))
                    v = torch.stack(
                        [F.resized_crop(v[..., i, :, :], i // 4, j // 4, h // 4, w // 4, mv_size,
                                        F.InterpolationMode.NEAREST)
                         for i in range(v.size(-3))],
                        dim=-3
                    )
                else:
                    v = F.resized_crop(v, i, j, h, w, self.size, self.interpolation)
                v = einops.rearrange(v, "(num_gop num_frame) c h w->num_gop num_frame c h w",
                                     num_gop=num_gop, num_frame=num_frame)
                ret[k] = v
            else:
                ret[k] = F.resized_crop(v, i, j, h, w, self.size, self.interpolation)
        return ret


class DictRandomCrop(transforms.RandomCrop):

    def forward(self, img: dict):
        assert self.padding is None and not self.pad_if_needed, "Padding is not supported by DictRandoCrop"

        assert any(k in img for k in IMAGE_KEYS), "input do not contain any valid image"
        for k in IMAGE_KEYS:
            if k in img:
                i, j, h, w = self.get_params(img[k], self.size)
                break

        ret = {}
        for k, v in img.items():
            if k not in IMAGE_KEYS:
                ret[k] = v
                continue
            if len(v.shape) == 5:  # handle extra dimension for GOP
                num_gop = v.size(0)
                num_frame = v.size(1)
                v = einops.rearrange(v, "num_gop num_frame c h w->(num_gop num_frame) c h w")
                if k == "motion_vector":
                    v = torch.stack(
                        [F.crop(v[..., i, :, :], i // 4, j // 4, h // 4, w // 4) for i in range(v.size(-3))],
                        dim=-3
                    )
                else:
                    v = F.crop(v, i, j, h, w)
                v = einops.rearrange(v, "(num_gop num_frame) c h w->num_gop num_frame c h w",
                                     num_gop=num_gop, num_frame=num_frame)
                ret[k] = v
            else:
                ret[k] = F.crop(v, i, j, h, w)
        return ret


class DictCenterCrop(transforms.CenterCrop):
    def forward(self, img: dict):
        ret = {}
        for k, v in img.items():
            if k not in IMAGE_KEYS:
                ret[k] = v
                continue
            if len(v.shape) == 5:  # handle extra dimension for GOP
                num_gop = v.size(0)
                num_frame = v.size(1)
                v = einops.rearrange(v, "num_gop num_frame c h w->(num_gop num_frame) c h w")
                if k == "motion_vector":
                    if isinstance(self.size, Sequence) and len(self.size) == 2:
                        mv_size = (self.size[0] // 4, self.size[1] // 4)
                    elif isinstance(self.size, int):
                        mv_size = self.size // 4
                    else:
                        raise ValueError("Image size is not supported: {}".format(self.size))
                    v = torch.stack(
                        [F.center_crop(v[..., i, :, :], mv_size)
                         for i in range(v.size(-3))],
                        dim=-3
                    )
                else:
                    v = F.center_crop(v, self.size)
                v = einops.rearrange(v, "(num_gop num_frame) c h w->num_gop num_frame c h w",
                                     num_gop=num_gop, num_frame=num_frame)
                ret[k] = v
            else:
                ret[k] = F.center_crop(v, self.size)
        return ret


class DictResize(transforms.Resize):

    def forward(self, img: dict):
        ret = {}
        for k, v in img.items():
            if k not in IMAGE_KEYS:
                ret[k] = v
                continue
            if len(v.shape) == 5:  # handle extra dimension for GOP
                num_gop = v.size(0)
                num_frame = v.size(1)
                v = einops.rearrange(v, "num_gop num_frame c h w->(num_gop num_frame) c h w")
                if k == "motion_vector":
                    if isinstance(self.size, Sequence) and len(self.size) == 2:
                        mv_size = (self.size[0] // 4, self.size[1] // 4)
                    elif isinstance(self.size, int):
                        mv_size = self.size // 4
                    else:
                        raise ValueError("Image size is not supported: {}".format(self.size))
                    v = torch.stack(
                        [F.resize(v[..., i, :, :], mv_size, F.InterpolationMode.NEAREST, self.max_size, self.antialias)
                         for i in range(v.size(-3))],
                        dim=-3
                    )
                else:
                    v = F.resize(v, self.size, self.interpolation, self.max_size, self.antialias)
                v = einops.rearrange(v, "(num_gop num_frame) c h w->num_gop num_frame c h w",
                                     num_gop=num_gop, num_frame=num_frame)
                ret[k] = v
            else:
                ret[k] = F.resize(v, self.size, self.interpolation, self.max_size, self.antialias)
        return ret


class DictNormalize(transforms.Normalize):
    def forward(self, data: dict):
        return {k: F.normalize(v, self.mean, self.std, self.inplace) if k in NORMALIZE_KEYS else v
                for k, v in data.items()}


class DictRandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def forward(self, img: dict):
        if torch.rand(1) < self.p:
            return {k: F.hflip(v) if k in IMAGE_KEYS else v for k, v in img.items()}
        else:
            return img

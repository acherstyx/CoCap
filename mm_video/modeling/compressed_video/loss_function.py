# -*- coding: utf-8 -*-
# @Time    : 7/19/23
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : loss_function.py

from ..loss import LOSS_REGISTRY, LossBase

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import logging

logger = logging.getLogger(__name__)


@LOSS_REGISTRY.register()
class LabelSmoothingLoss(LossBase):
    def __init__(self, cfg):
        label_smoothing = 0.1
        self.tgt_vocab_size = 49408
        ignore_index = 0
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__(cfg)

        self.log_softmax = nn.LogSoftmax(dim=-1)

        smoothing_value = label_smoothing / (self.tgt_vocab_size - 1)  # count for the ground-truth word
        one_hot = torch.full((self.tgt_vocab_size,), smoothing_value)
        # one_hot[self.ignore_index] = 0
        self.register_buffer("one_hot", one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, target, output):
        output = output["prediction_scores"]
        output = output.view(-1, self.tgt_vocab_size)
        target = target['input_labels'].reshape(-1).long()
        valid_indices = target != self.ignore_index  # ignore examples with target value -1
        target = target[valid_indices]
        output = self.log_softmax(output[valid_indices])

        model_prob = self.one_hot.repeat(target.size(0), 1).to(target.device)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        return F.kl_div(output, model_prob, reduction="sum")

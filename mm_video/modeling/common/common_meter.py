# -*- coding: utf-8 -*-
# @Time    : 7/19/23
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : common_meter.py

from mm_video.modeling.meter import METER_REGISTRY, MeterBase

import torch
import numpy as np
import datetime
import json
import logging
import os
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from mm_video.utils.train_utils import gather_object_multiple_gpu
from fvcore.common.config import CfgNode
from typing import Dict, Any

logger = logging.getLogger(__name__)


@METER_REGISTRY.register()
class AccuracyMeter(MeterBase):
    """
    accuracy meter for video action recognition
    """

    def __init__(self, cfg: CfgNode, writer: SummaryWriter, mode: str):
        super(AccuracyMeter, self).__init__(cfg=cfg, writer=writer, mode=mode)

        self._top1 = []
        self._top5 = []
        self._rank = []
        self._id = []

        self._label = []
        self._predict = []

    @staticmethod
    def topk_accuracy(logits: torch.Tensor, target: torch.LongTensor, topk=(1, 5), average=False, verbose=False):
        assert len(logits.shape) == 2
        assert len(target.shape) == 1
        assert logits.shape[0] == target.shape[0]

        num_class = logits.size(1)

        _, pred = logits.topk(num_class, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        ret = {}
        for k in topk:
            if average:
                cur_acc = correct[:k].float().sum(0).mean().cpu().numpy().item() * 100.0
            else:
                cur_acc = correct[:k].float().sum(0).cpu().numpy().tolist()
            ret[f"R@{k}"] = cur_acc

        _, rank = correct.float().topk(1, dim=0)
        rank = rank[0]
        if average:
            # res.append(torch.mean(rank.float()))
            ret["rank"] = torch.mean(rank.float()).cpu().numpy().item()
        else:
            ret["rank"] = rank.cpu().numpy().tolist()

        if verbose:  # debug log
            logger.debug("[pred, gt]: %s", list(zip(torch.argmax(logits, dim=-1).cpu().numpy(), target.cpu().numpy())))
            logger.debug("Accuracy: %s", ret)
        return ret

    @torch.no_grad()
    def _update(self, labels, outputs, global_step=None, idx=None):
        """
        update for each step
        :param labels: 1D LongTensor for classification
        :param outputs: 2D logits with shape (BATCH_SIZE, NUM_CLASSES)
        :param global_step: specify global step manually
        :param idx: unique id for each sample
        """
        assert len(labels.shape) == 1, "Label should be 1 dim tensor"
        assert len(outputs.shape) == 2, "Output should be 2 dim tensor"
        assert labels.size(0) == outputs.size(0), "Label and output should have same batch size"
        if idx is not None and isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        elif idx is None:
            idx = [None] * labels.size(0)
        accuracy_result = self.topk_accuracy(logits=outputs, target=labels, topk=(1, 5), average=False, verbose=True)
        self._top1.extend(accuracy_result["R@1"])
        self._top5.extend(accuracy_result["R@5"])
        self._rank.extend(accuracy_result["rank"])
        self._id.extend(idx)

        for cur_label in labels:
            self._label.append(cur_label.item())
        for cur_logits in outputs:
            assert len(cur_logits.shape) == 1, f"logits should be 1 dim tensor, received {cur_logits.shape}"
            pred = torch.argmax(cur_logits).item()
            self._predict.append(pred)
        assert len(self._label) == len(self._predict)

        # write tensorboard
        if self.mode in ["train"] and self.writer is not None and (not dist.is_initialized() or dist.get_rank() == 0):
            self.writer.add_scalar(f"{self.mode}/R@1", np.mean(accuracy_result["R@1"]), global_step=global_step)
            self.writer.add_scalar(f"{self.mode}/R@5", np.mean(accuracy_result["R@5"]), global_step=global_step)
            self.writer.add_scalar(f"{self.mode}/rank", np.mean(accuracy_result["rank"]), global_step=global_step)

    @torch.no_grad()
    def update(self, inputs, outputs, global_step=None):
        self._update(labels=inputs["label"], outputs=outputs["cls_logits"], global_step=global_step,
                     idx=inputs["id"] if "id" in inputs else None)

    @torch.no_grad()
    def summary(self, epoch):
        acc_top1 = gather_object_multiple_gpu(self._top1) if dist.is_initialized() else self._top1
        acc_top5 = gather_object_multiple_gpu(self._top5) if dist.is_initialized() else self._top5
        rank = gather_object_multiple_gpu(self._rank) if dist.is_initialized() else self._rank
        idx = gather_object_multiple_gpu(self._id) if dist.is_initialized() else self._id
        assert len(acc_top1) == len(acc_top5) == len(rank) == len(idx)

        all_metric = {}
        reassign_idx = 0
        for r1, r5, r, i in zip(acc_top1, acc_top5, rank, idx):
            if i is None:
                while reassign_idx in all_metric:
                    reassign_idx += 1
                i = reassign_idx
            all_metric[i] = {"R@1": r1, "R@5": r5, "Rank": r}
        with open(os.path.join(self.cfg.LOG.DIR, f"accuracy_epoch_{epoch}_{self.mode}.json"), "w") as f:
            json.dump(all_metric, f)
        acc_top1 = [x["R@1"] for i, x in all_metric.items()]
        acc_top5 = [x["R@5"] for i, x in all_metric.items()]
        rank = [x["Rank"] for i, x in all_metric.items()]
        avg_acc_top1 = np.mean(acc_top1)
        avg_acc_top5 = np.mean(acc_top5)
        avg_rank = np.mean(rank)
        mid_rank = np.median(rank)

        if dist.is_initialized():
            labels = np.array(gather_object_multiple_gpu(self._label))
            predicts = np.array(gather_object_multiple_gpu(self._predict))
        else:
            labels = np.array(self._label)
            predicts = np.array(self._predict)
        correct = (labels == predicts).astype(int)
        individual_labels = list(set(labels))
        accuracy_per_class = {}
        for cur_label in individual_labels:
            correct_selected = correct[labels == cur_label]
            accuracy_per_class[cur_label] = correct_selected.sum() / len(correct_selected)
        logger.debug("accuracy for each class: %s", accuracy_per_class)
        if not dist.is_initialized() or dist.get_rank() == 0:
            img = self.visualize_per_class_accuracy(accuracy_per_class, avg_acc_top1)
            self.writer.add_image(f"{self.mode}/accuracy_epoch", img, global_step=epoch, dataformats="HWC")

        logger.debug("Accuracy meter summary got {} samples".format(len(acc_top1)))
        if self.writer is not None and (not dist.is_initialized() or dist.get_rank() == 0) and epoch is not None:
            self.writer.add_scalar(f"{self.mode}/R@1_epoch", avg_acc_top1, global_step=epoch)
            self.writer.add_scalar(f"{self.mode}/R@5_epoch", avg_acc_top5, global_step=epoch)
            self.writer.add_scalar(f"{self.mode}/MeanR_epoch", avg_rank, global_step=epoch)
            self.writer.add_scalar(f"{self.mode}/MedianR_epoch", mid_rank, global_step=epoch)
        if self.mode in ["val", "test"] and (not dist.is_initialized() or dist.get_rank() == 0):
            logger.info(f">>> Epoch {epoch} ({self.mode}): R@1: {avg_acc_top1} - R@5: {avg_acc_top5}"
                        f" - MeanR: {avg_rank} - MedianR: {mid_rank}")

        return {"R@1": avg_acc_top1,
                "R@5": avg_acc_top5,
                "MeanR": avg_rank,
                "MedianR": mid_rank}

    def reset(self):
        self._top1.clear()
        self._top5.clear()
        self._label.clear()
        self._predict.clear()
        self._rank.clear()
        self._id.clear()

    def visualize_per_class_accuracy(self, accuracy_dict: Dict[Any, str], average_acc: float):
        import matplotlib.pyplot as plt

        labels = [k for k, v in accuracy_dict.items()]
        acc = [v * 100 for k, v in accuracy_dict.items()]

        save_path = os.path.join(self.cfg.LOG.DIR,
                                 f"Accuracy-{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}.png")

        ax: plt.Axes = plt.gca()
        ax.bar(labels, acc)
        ax.set_xlabel("Label")
        ax.set_ylabel("Accuracy(%)")
        ax.set_ylim(0.0, 100.0)
        ax.axhline(average_acc, color="r", label="average", linestyle=":")
        plt.savefig(save_path)
        logger.debug("Accuracy for each class is saved to %s", save_path)
        plt.close()
        img = plt.imread(save_path)
        return img

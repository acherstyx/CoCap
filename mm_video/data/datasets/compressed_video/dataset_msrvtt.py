# -*- coding: utf-8 -*-
# @Time    : 2022/11/17 16:54
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : dataset_msrvtt.py

import os
import torch
import random
from torch.utils import data
import json
import pandas as pd
from collections import defaultdict
from torchvision import transforms

from mm_video.layers.clip import clip

from mm_video.data.build import DATASET_REGISTRY

from .video_text_base import get_video
from .transforms import (DictNormalize, DictCenterCrop, DictRandomHorizontalFlip)
from .video_readers import VIDEO_READER_REGISTRY

from mm_video.utils.json import load_json


@DATASET_REGISTRY.register()
class MSRVTTCaptioningDatasetForCLIP(data.Dataset):

    def __init__(self, cfg, split):
        self.split = split
        self.video_root = cfg.DATA.DATASET.MSRVTT.VIDEO_ROOT
        self.max_words = cfg.DATA.DATASET.MSRVTT.MAX_WORDS
        self.max_frames = cfg.DATA.DATASET.MSRVTT.MAX_FRAMES
        self.unfold_sentences = cfg.DATA.DATASET.MSRVTT.UNFOLD_SENTENCES  # only affect the train split
        self.height, self.width = cfg.DATA.DATASET.MSRVTT.VIDEO_SIZE
        self.sentences = []  # (vid, [sentence, ...])
        self.h265_cfg = cfg.CV_CONFIG
        metadata = load_json(cfg.DATA.DATASET.MSRVTT.METADATA)
        video_ids = [metadata['videos'][idx]['video_id'] for idx in range(len(metadata['videos']))]
        all_split_video_ids = {"train": video_ids[:6513], "val": video_ids[6513:6513 + 497],
                               "test": video_ids[6513 + 497:]}

        split_video_ids = all_split_video_ids[split].copy()
        if self.unfold_sentences:
            for item in metadata["sentences"]:
                if item["video_id"] in split_video_ids:
                    self.sentences.append([item["video_id"], [item["caption"]]])
                    if split == "test":
                        split_video_ids.remove(item["video_id"])
        else:
            vid2sentence = defaultdict(list)
            for item in metadata["sentences"]:
                if item["video_id"] in split_video_ids:
                    vid2sentence[item["video_id"]].append(item["caption"])
            self.sentences = list(vid2sentence.items())

        # self.sentences = self.sentences[:50000]
        self.video_reader = VIDEO_READER_REGISTRY.get(cfg.DATA.DATASET.MSRVTT.VIDEO_READER)
        # transforms
        normalize = DictNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        if split == "train":
            self.transform = transforms.Compose([
                DictCenterCrop((self.height, self.width)),
                DictRandomHorizontalFlip(),
                normalize
            ])
        elif split == "test":
            self.transform = transforms.Compose([
                DictCenterCrop((self.height, self.width)),
                normalize
            ])
        else:
            raise NotImplementedError

        if split == "test":
            json_ref = {k: [] for k in all_split_video_ids[split]}
            for sentence in metadata["sentences"]:
                if sentence["video_id"] in json_ref:
                    json_ref[sentence["video_id"]].append(sentence["caption"])
            # verify
            assert all(len(v) == 20 for _, v in json_ref.items())
            self.json_ref = {k[len("video"):]: v for k, v in json_ref.items()}

    def __len__(self):
        return len(self.sentences)

    def _get_video(self, video_id):
        video, video_mask = get_video(video_reader=self.video_reader,
                                      video_path=os.path.join(self.video_root, f"{video_id}.mp4"),
                                      max_frames=self.max_frames,
                                      sample="rand" if self.split == "train" else "uniform",
                                      hevc_config=self.h265_cfg)
        if self.transform is not None:
            video = self.transform(video)
        return video, video_mask

    def __getitem__(self, idx):
        video_id, sentence_list = self.sentences[idx]
        sentence = random.choice(sentence_list)

        input_ids = clip.tokenize(sentence, context_length=self.max_words, truncate=True)[0]
        input_mask = torch.zeros(self.max_words, dtype=torch.long)
        input_mask[:len(clip._tokenizer.encode(sentence)) + 2] = 1

        video, video_mask = self._get_video(video_id)
        input_labels = torch.cat((input_ids[1:], torch.IntTensor([0])))
        return {
            # video
            "video": video,
            "video_mask": video_mask,
            # text
            "input_ids": input_ids,
            "input_labels": input_labels,
            "input_mask": input_mask,
            # metadata
            "metadata": (video_id, sentence)
        }

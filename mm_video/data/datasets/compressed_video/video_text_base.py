# -*- coding: utf-8 -*-
# @Time    : 2022/12/3 14:54
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : video_text_base.py

import random
import logging

import torch

logger = logging.getLogger(__name__)


def get_tokenized_words(sentence: str, tokenizer, max_words):
    words = tokenizer.tokenize(sentence)
    words = ["[CLS]"] + words
    total_length_with_cls = max_words - 1
    if len(words) > total_length_with_cls:
        words = words[:total_length_with_cls]
    words = words + ["[SEP]"]
    return words


def get_text_inputs(sentence: str, tokenizer, max_words):
    """
    1. tokenize
    2. add [CLS] and [SEP] token, limit the length
    3. create mask and token type
    4. pad to max_words
    :param sentence:
    :param tokenizer:
    :param max_words:
    :return: 1 dim tensor, shape is (max_words,)
    """
    words = get_tokenized_words(sentence, tokenizer, max_words)
    input_ids = tokenizer.convert_tokens_to_ids(words)
    input_mask = [1] * len(input_ids)  # 1 is keep, 0 is mask out
    segment_ids = [0] * len(input_ids)
    while len(input_ids) < max_words:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    assert len(input_ids) == len(input_mask) == len(segment_ids) == max_words
    return torch.tensor(input_ids), torch.tensor(input_mask), torch.tensor(segment_ids)


def get_text_inputs_with_mlm(sentence: str, tokenizer, max_words):
    """
    Add mlm inputs and labels based on `get_text_inputs`
    :param sentence:
    :param tokenizer:
    :param max_words:
    :return: 1 dim tensor, shape is (max_words,)
    """
    input_ids, input_mask, segment_ids = get_text_inputs(sentence, tokenizer, max_words)

    # Mask Language Model <-----
    token_labels = []
    masked_tokens = get_tokenized_words(sentence, tokenizer, max_words)
    for token_id, token in enumerate(masked_tokens):
        if token_id == 0 or token_id == len(masked_tokens) - 1:
            token_labels.append(-1)
            continue
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15
            # 80% randomly change token to mask token
            if prob < 0.8:
                masked_tokens[token_id] = "[MASK]"
            # 10% randomly change token to random token
            elif prob < 0.9:
                masked_tokens[token_id] = random.choice(list(tokenizer.vocab.items()))[0]
            # -> rest 10% randomly keep current token
            # append current token to output (we will predict these later)
            try:
                token_labels.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                token_labels.append(tokenizer.vocab["[UNK]"])
                logger.debug("Cannot find token '{}' in vocab. Using [UNK] instead".format(token))
        else:
            # no masking token (will be ignored by loss function later)
            token_labels.append(-1)
    # -----> Mask Language Model
    masked_token_ids = tokenizer.convert_tokens_to_ids(masked_tokens)

    while len(masked_token_ids) < max_words:
        masked_token_ids.append(0)
        token_labels.append(-1)
    assert len(masked_token_ids) == len(token_labels) == max_words
    return input_ids, input_mask, segment_ids, torch.tensor(masked_token_ids), torch.tensor(token_labels)


def get_video(video_reader, video_path, max_frames, sample, hevc_config=None):
    video_mask = torch.ones((max_frames,), dtype=torch.int)
    if video_reader.__name__ in ["read_frames_compressed_domain"]:
        assert hevc_config is not None, "hevc_config should be set when using read_frames_compressed_domain"
        video, _ = video_reader(video_path,
                                resample_num_gop=hevc_config.NUM_GOP, resample_num_mv=hevc_config.NUM_MV,
                                resample_num_res=hevc_config.NUM_RES,
                                with_residual=hevc_config.WITH_RESIDUAL,
                                pre_extract=hevc_config.USE_PRE_EXTRACT,
                                sample=hevc_config.SAMPLE if hevc_config.SAMPLE == "pad" else sample)
    else:
        video, _ = video_reader(video_path, max_frames, sample)
    return video, video_mask

# -*- coding: utf-8 -*-
# @Time    : 8/6/23
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : modeling.py

import logging
from typing import *

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from mm_video.layers.clip.model import Transformer, LayerNorm
from mm_video.layers.bert import BertSelfEncoder, BertLMPredictionHead
from mm_video.modeling.model import MODEL_REGISTRY
from mm_video.modeling.compressed_video.compressed_video_transformer import CompressedVideoTransformer

from mm_video.layers.clip.model import CLIP
from mm_video.layers.clip.clip import get_model_path

from easydict import EasyDict as edict

logger = logging.getLogger(__name__)


class CaptionHead(nn.Module):

    def __init__(
            self,
            word_embedding_size: int, visual_feature_size: int,
            max_v_len: int, max_t_len: int, hidden_size: int,
            vocab_size: int, verbose: Optional[Union[int, bool]] = False
    ):
        super(CaptionHead, self).__init__()
        self.model_network = "Self"
        self.cap_config = edict(
            word_vec_size=word_embedding_size,
            max_v_len=max_v_len,
            max_t_len=max_t_len,
            hidden_size=hidden_size,
            video_feature_size=visual_feature_size,
            layer_norm_eps=1e-12,  # bert layernorm
            hidden_dropout_prob=0.1,  # applies everywhere except attention
            num_hidden_layers=2,  # number of transformer layers
            num_attention_heads=8,
            share_wd_cls_weight=False,
            vocab_size=vocab_size,
            BOS_id=vocab_size - 2,
            EOS_id=vocab_size - 1,
            PAD_id=0
        )
        logger.debug("Caption Head Configuration: %s", self.cap_config)
        self.cap_sa_decoder = BertSelfEncoder(self.cap_config)
        self.prediction_head = BertLMPredictionHead(self.cap_config, self.cap_sa_decoder.word_embeddings.weight)
        # debug output cfgs
        if verbose:
            if isinstance(verbose, bool):
                self.log_interval = 1
            else:
                self.log_interval = int(verbose)
        else:
            self.log_interval = float("inf")
        self.step_counter = 1

    @staticmethod
    @torch.no_grad()
    def probability2text(predict_scores=None):
        predict_ids = predict_scores.max(-1)[1]
        return CaptionHead.ids2text(predict_ids)

    @staticmethod
    @torch.no_grad()
    def ids2text(gt_ids: Union[np.ndarray, torch.Tensor]):
        from mm_video.trainer.cocap_trainer import convert_ids_to_sentence
        if isinstance(gt_ids, np.ndarray) or isinstance(gt_ids, torch.Tensor):
            assert 0 < len(gt_ids.shape) <= 2, f"gt_ids should be a 1 dim or 2 dim array/tensor, got {gt_ids.shape}"
        else:
            raise ValueError("gt_ids should be np.ndarray or torch.Tensor")
        if isinstance(gt_ids, torch.Tensor):
            gt_ids = gt_ids.detach().cpu().numpy()
        if len(gt_ids.shape) == 1:
            return convert_ids_to_sentence(gt_ids.tolist())
        else:
            return [convert_ids_to_sentence(_gt_ids) for _gt_ids in gt_ids.tolist()]

    def forward(self, visual_output, input_ids, input_mask):
        assert input_ids.size(1) == self.cap_config.max_t_len, f"{input_ids.size(1)} vs {self.cap_config.max_t_len}"

        input_types = torch.concat(
            [
                torch.full((visual_output["feature_context"].size(0), visual_output["feature_context"].size(1)),
                           fill_value=1, dtype=torch.long, device=visual_output["feature_context"].device),
                torch.full((visual_output["feature_action"].size(0), visual_output["feature_action"].size(1)),
                           fill_value=0, dtype=torch.long, device=visual_output["feature_action"].device),
                torch.full((input_ids.size(0), input_ids.size(1)),
                           fill_value=2, dtype=torch.long, device=input_ids.device)
            ], dim=1
        )
        visual_output = torch.cat([visual_output["feature_context"], visual_output["feature_action"]], dim=1)
        input_mask = torch.concat(
            [
                torch.ones(size=(visual_output.size(0), visual_output.size(1)),
                           dtype=torch.long, device=visual_output.device),
                input_mask
            ], dim=1
        )
        hidden = self.cap_sa_decoder.forward(visual_output, input_ids, input_mask, input_types)
        prediction_scores = self.prediction_head(hidden[:, -self.cap_config.max_t_len:])
        if self.step_counter % self.log_interval == 0:
            logger.debug("GT  : %s", self.ids2text(input_ids))
            logger.debug("Pred: %s", self.probability2text(prediction_scores))
        self.step_counter += 1
        return prediction_scores

    @classmethod
    def from_pretrained(
            cls, pretrained_clip_name_or_path: str, max_v_len: int, max_t_len: int,
            verbose: Optional[Union[int, bool]] = False
    ):
        model_path = get_model_path(pretrained_clip_name_or_path, download_root="model_zoo/clip_model")
        pretrained_model: CLIP = torch.jit.load(model_path, map_location="cpu")
        state_dict = pretrained_model.state_dict()

        embed_dim = state_dict["text_projection"].shape[1]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]

        head = cls(
            word_embedding_size=transformer_width,
            visual_feature_size=embed_dim,
            max_v_len=max_v_len,
            max_t_len=max_t_len,
            hidden_size=embed_dim,
            vocab_size=vocab_size,
            verbose=verbose
        )
        logger.debug(
            "Pretrained embedding parameters: %s",
            [k for k, v in state_dict.items() if k.startswith("token_embedding")]
        )
        pretrained_embedding = {k.lstrip("token_embedding."): v for k, v in state_dict.items()
                                if k.startswith("token_embedding")}
        head.cap_sa_decoder.word_embeddings.load_state_dict(pretrained_embedding, strict=True)
        head.prediction_head.decoder.load_state_dict(pretrained_embedding, strict=True)
        assert torch.equal(head.cap_sa_decoder.word_embeddings.weight, head.prediction_head.decoder.weight)
        return head


@MODEL_REGISTRY.register()
class CoCap(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        cfg_model = cfg.MODEL.COCAP

        self.compressed_video_transformer = CompressedVideoTransformer.from_pretrained(
            pretrained_clip_name_or_path=cfg_model.PRETRAINED_CLIP,
            # motion
            motion_patch_size=cfg_model.MOTION_ENCODER.PATCH_SIZE,
            motion_layers=cfg_model.MOTION_ENCODER.N_LAYERS,
            motion_heads=cfg_model.MOTION_ENCODER.N_HEADS,
            # residual
            residual_patch_size=cfg_model.RESIDUAL_ENCODER.PATCH_SIZE,
            residual_layers=cfg_model.RESIDUAL_ENCODER.N_LAYERS,
            residual_heads=cfg_model.RESIDUAL_ENCODER.N_HEADS,
            # action
            action_layers=cfg_model.ACTION_ENCODER.N_LAYERS,
            action_heads=cfg_model.ACTION_ENCODER.N_HEADS,
            n_bp=cfg.CV_CONFIG.NUM_MV
        )

        self.dropout_motion = nn.Dropout(cfg_model.MOTION_DROPOUT_PROB)
        self.dropout_residual = nn.Dropout(cfg_model.RESIDUAL_DROPOUT_PROB)

        self.task_type = cfg_model.TASK_TYPE

        if self.task_type == "captioning":
            self.caption_head = CaptionHead.from_pretrained(
                pretrained_clip_name_or_path=cfg_model.PRETRAINED_CLIP,
                max_t_len=77, max_v_len=cfg.CV_CONFIG.NUM_GOP * 2, verbose=10
            )
        else:
            raise ValueError("Task type not supported: %s" % self.task_type)

    def forward(self, inputs: Dict[str, torch.Tensor]):
        """

        :param inputs:
            video:
                iframe:         batch_size n_gop c h w
                motion_vector:  batch_size n_gop n_mv c=4|9 h/4 w/4
                residual:       batch_size n_gop n_res c h w
                input_mask_gop: batch_size n_gop
                input_mask_mv:  batch_size n_gop n_mv
        :return:
        """
        if "visual_output" not in inputs:
            iframe = inputs["video"]["iframe"]
            motion = inputs["video"]["motion_vector"]
            residual = inputs["video"]["residual"] / 128 - 1  # for saving memory
            bp_type_ids = inputs["video"]["type_ids_mv"]

            motion = self.dropout_motion(motion)
            residual = self.dropout_residual(residual)
            compressed_visual_features = self.compressed_video_transformer(
                iframe=iframe,
                motion=motion,
                residual=residual,
                bp_type_ids=bp_type_ids
            )
        else:
            # reuse pre-extracted visual features
            compressed_visual_features = inputs["visual_output"]

        if self.task_type == "captioning":
            prediction_scores = self.caption_head(
                compressed_visual_features,
                inputs["input_ids"],
                inputs["input_mask"],
            )
            return {"prediction_scores": prediction_scores, "visual_output": compressed_visual_features}
        else:
            raise ValueError("Task type not supported: %s" % self.task_type)

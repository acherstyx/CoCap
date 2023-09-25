# -*- coding: utf-8 -*-
# @Time    : 8/2/23
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : compressed_video_transformer.py

import torch
import torch.nn as nn
import einops
from typing import Optional, Dict, Tuple

from mm_video.layers.clip.model import VisionTransformer, CrossResidualAttentionBlock, LayerNorm, CLIP
from mm_video.layers.clip.clip import get_model_path


class IFrameEncoder(VisionTransformer):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 in_channels: int = 3):
        super().__init__(
            input_resolution=input_resolution, patch_size=patch_size, width=width, layers=layers, heads=heads,
            output_dim=output_dim, in_channels=in_channels
        )

        scale = width ** -0.5
        self.ln_post_hidden = LayerNorm(width)
        self.proj_hidden = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, output_all_features: bool = False, output_attention_map: bool = False):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        grid = x.size(2)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) +
             torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x],
            dim=1
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, attn = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        cls_feature = self.ln_post(x[:, 0, :]) @ self.proj

        outputs = (cls_feature,)
        if output_all_features:
            # cls token is not included
            outputs += (self.ln_post_hidden(x[:, 1:, :]) @ self.proj_hidden,)
        if output_attention_map:
            # attention_map: n_layers, batch_size, n_heads, h, w
            outputs += (einops.rearrange(attn[:, :, :, 0, 1:],
                                         "n_layers b n_heads (h w)->n_layers b n_heads h w", h=grid, w=grid),)
        return outputs

    @classmethod
    def from_pretrained(cls, pretrained_clip_name_or_path: str) -> Tuple["IFrameEncoder", int, int, int]:
        """
        Load from pretrained CLIP model
        :param pretrained_clip_name_or_path: the name of pretrained CLIP model
        :return: IFrameEncoder, image_resolution, vision_width, embed_dim
        """
        model_path = get_model_path(pretrained_clip_name_or_path, download_root="model_zoo/clip_model")
        pretrained_model: CLIP = torch.jit.load(model_path, map_location="cpu")
        state_dict = pretrained_model.state_dict()

        vision_width: int = state_dict["visual.conv1.weight"].shape[0]
        vision_layers: int = len([k for k in state_dict.keys()
                                  if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        embed_dim: int = state_dict["text_projection"].shape[1]
        vision_patch_size: int = state_dict["visual.conv1.weight"].shape[-1]
        grid_size: int = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution: int = vision_patch_size * grid_size
        vision_heads = vision_width // 64

        rgb_encoder = cls(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width, layers=vision_layers, heads=vision_heads,
            output_dim=embed_dim
        )
        visual_state_dict = pretrained_model.visual.state_dict()
        # manually build the state dict to make sure pretrained weights are loaded exactly
        visual_state_dict.update({k: v for k, v in rgb_encoder.state_dict().items() if k.startswith("ln_post_hidden")})
        visual_state_dict.update({k: v for k, v in rgb_encoder.state_dict().items() if k.startswith("proj_hidden")})
        rgb_encoder.load_state_dict(visual_state_dict, strict=True)

        return rgb_encoder, image_resolution, vision_width, embed_dim


class ActionEncoder(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, n_bp: int, n_bp_type: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[CrossResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

        self.positional_embedding = nn.Embedding(n_bp, width)
        self.bp_type_embedding = nn.Embedding(n_bp_type, width)

        self.ln_post = LayerNorm(width)

    def forward(
            self,
            feature_bp: torch.FloatTensor,
            bp_type_ids: torch.LongTensor,
            feature_ctx: torch.FloatTensor = None,
            self_mask: torch.Tensor = None
    ):
        """
        Fuse I-frames, motion vectors and residuals.
        The feature dimension of feature_bp and feature_ctx are the same.
        :param feature_bp:  (bsz n_gop) n_bp c
        :param bp_type_ids: (bsz n_gop) n_bp
        :param feature_ctx: (bsz n_gop) (h w) c
        :param self_mask: attention mask
        :return:
        """
        assert feature_bp.size(1) == self.positional_embedding.num_embeddings
        bsz = feature_bp.size(0)

        positional_embedding = self.positional_embedding(
            torch.LongTensor(list(range(feature_bp.size(1)))).to(feature_bp.device).unsqueeze(0).repeat(bsz, 1)
        )
        bp_type_embedding = self.bp_type_embedding(bp_type_ids)
        feature_bp = feature_bp + positional_embedding + bp_type_embedding

        feature_bp, _, _ = self.resblocks([feature_bp.permute(1, 0, 2), feature_ctx.permute(1, 0, 2), self_mask])
        feature_bp = torch.mean(feature_bp.permute(1, 0, 2), dim=1)  # pooling
        return self.ln_post(feature_bp)


class CompressedVideoTransformer(nn.Module):
    def __init__(
            self,
            rgb_encoder: nn.Module,
            motion_encoder: Optional[nn.Module],
            residual_encoder: Optional[nn.Module],
            action_encoder: Optional[nn.Module],
            output_dim: int,
    ):
        """
        Encode visual feature from video compressed domain
        :param rgb_encoder:
        :param motion_encoder:
        :param residual_encoder:
        :param action_encoder: ActionEncoder used to fuse the rgb, motion and residual features
        :param output_dim: width of output visual feature
        """
        super().__init__()
        assert motion_encoder is not None or residual_encoder is not None
        self.rgb_encoder = rgb_encoder
        self.motion_encoder = motion_encoder
        self.residual_encoder = residual_encoder
        self.action_encoder = action_encoder
        self.output_dim = output_dim

    def forward(
            self,
            iframe: torch.FloatTensor,
            motion: torch.FloatTensor,
            residual: torch.FloatTensor,
            bp_type_ids: torch.LongTensor
    ) -> Dict[str, torch.Tensor]:
        """

        :param iframe:      bsz n_gop c h w
        :param motion:      bsz n_gop n_bp c_mv h/4 w/4
        :param residual:    bsz n_gop n_bp c h w
        :param bp_type_ids: bsz n_gop n_bp
        """
        # a long list of dimension check
        assert (len(iframe.shape) == 5 and len(motion.shape) == 6 and len(residual.shape) == 6 and
                len(bp_type_ids.shape) == 3)
        assert iframe.size(0) == motion.size(0) == residual.size(0) == bp_type_ids.size(0), "batch size should be equal"
        assert iframe.size(1) == motion.size(1) == residual.size(1) == bp_type_ids.size(1), "n_gop should be equal"
        assert motion.size(2) == residual.size(2) == bp_type_ids.size(2), "n_mv and n_res should be equal"
        assert iframe.size(2) == 3 and motion.size(3) == 4 and residual.size(3) == 3, "channel number is not correct"
        assert iframe.size(3) == residual.size(4) and motion.size(4) == iframe.size(3) // 4, "height is not correct"
        assert iframe.size(4) == residual.size(5) and motion.size(5) == iframe.size(4) // 4, "width is not correct"

        _bsz, n_gop, n_bp = iframe.size(0), motion.size(1), motion.size(2)

        # encode iframe in batches
        f_ctx_cls, f_ctx_all_hidden, iframe_attn = self.rgb_encoder(
            einops.rearrange(iframe, "bsz n_gop c h w->(bsz n_gop) c h w"),
            output_all_features=True, output_attention_map=True
        )
        f_ctx_cls = einops.rearrange(f_ctx_cls, "(bsz n_gop) c->bsz n_gop c", bsz=_bsz)
        f_ctx_all_hidden = einops.rearrange(f_ctx_all_hidden, "(bsz n_gop) hw c->bsz n_gop hw c", bsz=_bsz)
        iframe_attn = einops.rearrange(
            iframe_attn, "n_layers (bsz n_gop) n_heads h w->n_layers bsz n_gop n_heads h w",
            bsz=_bsz
        )
        # encode motion in batches
        mv_cls, mv_attn = self.motion_encoder(
            einops.rearrange(motion, "bsz n_gop n_bp c_mv h_4 w_4->(bsz n_gop n_bp) c_mv h_4 w_4"),
            output_all_features=False, output_attention_map=True
        )
        mv_cls = einops.rearrange(mv_cls, "(bsz n_gop n_bp) c->bsz n_gop n_bp c",
                                  bsz=_bsz, n_gop=n_gop, n_bp=n_bp)
        mv_attn = einops.rearrange(
            mv_attn, "n_layers (bsz n_gop n_bp) n_heads h_4 w_4->n_layers bsz n_gop n_bp n_heads h_4 w_4",
            bsz=_bsz, n_gop=n_gop, n_bp=n_bp
        )
        # encode residual in batches
        res_cls, res_attn = self.residual_encoder(
            einops.rearrange(residual, "bsz n_gop n_bp c h w->(bsz n_gop n_bp) c h w"),
            output_all_features=False, output_attention_map=True
        )
        res_cls = einops.rearrange(res_cls, "(bsz n_gop n_bp) c->bsz n_gop n_bp c",
                                   bsz=_bsz, n_gop=n_gop, n_bp=n_bp)
        res_attn = einops.rearrange(
            res_attn, "n_layers (bsz n_gop n_bp) n_heads h w->n_layers bsz n_gop n_bp n_heads h w",
            bsz=_bsz, n_gop=n_gop, n_bp=n_bp
        )

        # fuse rgb, mv and res features through action encoder
        f_bp = mv_cls + res_cls
        f_act = self.action_encoder(  # squeeze n_gop into batch before forwarding
            einops.rearrange(f_bp, "bsz n_gop n_bp c->(bsz n_gop) n_bp c"),
            einops.rearrange(bp_type_ids, "bsz n_gop n_bp->(bsz n_gop) n_bp"),
            einops.rearrange(f_ctx_all_hidden, "bsz n_gop hw c->(bsz n_gop) hw c")
        )
        f_act = einops.rearrange(f_act, "(bsz n_gop) c->bsz n_gop c", bsz=_bsz, n_gop=n_gop)

        return {
            "feature_context": f_ctx_cls,
            "feature_action": f_act,
            "iframe_attention_map": iframe_attn,
            "motion_vector_attention_map": mv_attn,
            "residual_attention_map": res_attn,
        }

    @classmethod
    def from_pretrained(
            cls,
            # rgb encoder cfgs
            pretrained_clip_name_or_path: str,
            # motion encoder cfgs
            motion_patch_size: int, motion_layers: int, motion_heads: int,
            # residual encoder cfgs
            residual_patch_size: int, residual_layers: int, residual_heads: int,
            # action encoder cfgs
            action_layers: int, action_heads: int, n_bp: int
    ):
        rgb_encoder, image_resolution, vision_width, embed_dim = IFrameEncoder.from_pretrained(
            pretrained_clip_name_or_path
        )

        motion_encoder = VisionTransformer(
            input_resolution=image_resolution // 4,
            patch_size=motion_patch_size,
            width=vision_width // 4, layers=motion_layers, heads=motion_heads,
            output_dim=embed_dim,
            in_channels=4
        )
        residual_encoder = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=residual_patch_size,
            width=vision_width, layers=residual_layers, heads=residual_heads,
            output_dim=embed_dim,
            in_channels=3
        )
        action_encoder = ActionEncoder(
            width=embed_dim, layers=action_layers, heads=action_heads, n_bp=n_bp, n_bp_type=2
        )
        return cls(
            rgb_encoder=rgb_encoder,
            motion_encoder=motion_encoder,
            residual_encoder=residual_encoder,
            action_encoder=action_encoder,
            output_dim=embed_dim
        )

# -*- coding: utf-8 -*-
# @Time    : 2022/11/15 22:01
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : cocap_trainer.py

import os
import logging
import einops
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.optim import lr_scheduler
from collections import defaultdict

from .build import TRAINER_REGISTRY, TrainerBase
from ..data.build import build_loader
from ..modeling.model import build_model
from ..modeling.optimizer import build_optimizer
from ..modeling.loss import build_loss
from ..modeling.meter import build_meter
from mm_video.layers.bert import BertLayerNorm

from ..utils.train_utils import CudaPreFetcher, gather_object_multiple_gpu, get_timestamp
from ..utils.checkpoint import save_checkpoint, load_checkpoint, auto_resume, load_model
from ..utils.writer import get_writer
from ..utils.profile import Timer
from ..utils.json import save_json

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

from typing import List, AnyStr

logger = logging.getLogger(__name__)


@TRAINER_REGISTRY.register()
class CoCapTrainer(TrainerBase):

    def __init__(self, cfg):
        self.task_type = cfg.TRAINER.CAPTION_TRAINER.TASK_TYPE
        super(CoCapTrainer, self).__init__(cfg)

    def build(self):
        cfg = self.cfg
        self.model = build_model(cfg)
        self.dataloader = build_loader(cfg, mode=("train", "test"))
        # update total step for optimizer
        gradient_accumulation_steps = cfg.TRAINER.TRAINER_BASE.GRADIENT_ACCUMULATION_STEPS
        epoch = cfg.TRAINER.TRAINER_BASE.EPOCH
        num_train_optimization_steps = (int(len(self.dataloader["train"]) + gradient_accumulation_steps - 1)
                                        / gradient_accumulation_steps) * epoch
        cfg.defrost()
        cfg.OPTIMIZER.PARAMETER.t_total = num_train_optimization_steps
        cfg.freeze()
        self.optimizer, self.scheduler = self.prep_optimizer(cfg, self.model)
        self.loss_func = build_loss(self.cfg)
        self.scaler = GradScaler() if self.enable_amp else None
        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.debug("Model:\n%s", self.model)

    @staticmethod
    def prep_optimizer(cfg, model):
        # based on:
        # https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136
        if hasattr(model, 'module'):
            model = model.module

        decay = set()
        no_decay = set()

        pretrained_modules = [
            "compressed_video_transformer.rgb_encoder.conv1",
            "compressed_video_transformer.rgb_encoder.class_embedding",
            "compressed_video_transformer.rgb_encoder.positional_embedding",
            "compressed_video_transformer.rgb_encoder.ln_pre",
            "compressed_video_transformer.rgb_encoder.transformer",
            "compressed_video_transformer.rgb_encoder.ln_post",
            "compressed_video_transformer.rgb_encoder.proj",
            "caption_head.cap_sa_decoder.word_embeddings",
            "caption_head.prediction_head.decoder",
        ]
        whitelist_weight_modules = (nn.Linear, nn.MultiheadAttention, nn.Conv2d)
        blacklist_weight_modules = (nn.LayerNorm, nn.BatchNorm2d, nn.Embedding, BertLayerNorm)
        for mn, m in model.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if any(fpn.startswith(p_fpn) for p_fpn in pretrained_modules):  # pretrained
                    no_decay.add(fpn)
                elif pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("proj") or pn.endswith("projection"):
                    decay.add(fpn)
                elif fpn.endswith("embedding"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, \
            "parameters %s were not separated into either decay/no_decay set!" % (
                str(param_dict.keys() - union_params),)

        pretrained_no_decay = [pn for pn in sorted(list(no_decay)) if
                               any(pn.startswith(p_pn) for p_pn in pretrained_modules)]
        not_pretrained_no_decay = [pn for pn in sorted(list(no_decay)) if
                                   not any(pn.startswith(p_pn) for p_pn in pretrained_modules)]

        logger.debug("Parameter group decay_param: %s",
                     "\n   " + "\n   ".join([pn for pn in sorted(list(decay))]))
        logger.debug("Parameter group no_decay_pretrained_param: %s",
                     "\n   " + "\n   ".join([pn for pn in sorted(list(pretrained_no_decay))]))
        logger.debug("Parameter group no_decay_not_pretrained_param: %s",
                     "\n   " + "\n   ".join([pn for pn in sorted(list(not_pretrained_no_decay))]))

        decay_param = [param_dict[pn] for pn in sorted(list(decay))]
        no_decay_pretrained_param = [param_dict[pn] for pn in sorted(list(pretrained_no_decay))]
        no_decay_not_pretrained_param = [param_dict[pn] for pn in sorted(list(not_pretrained_no_decay))]

        optimizer_grouped_parameters = [
            {"params": decay_param},
            {"params": no_decay_pretrained_param, "weight_decay": 0.0, "lr": cfg.TRAINER.CAPTION_TRAINER.CLIP_LR},
            {"params": no_decay_not_pretrained_param, "weight_decay": 0.0}
        ]

        warmup_epoch = int(cfg.OPTIMIZER.PARAMETER.warmup * cfg.TRAINER.TRAINER_BASE.EPOCH)
        optimizer = build_optimizer(cfg, optimizer_grouped_parameters)
        scheduler = lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1 if epoch < warmup_epoch
            else cfg.TRAINER.CAPTION_TRAINER.LR_DECAY_GAMMA ** (epoch - warmup_epoch)
        )

        return optimizer, scheduler

    @torch.no_grad()
    def _on_test_epoch(self):
        # switch based on (downstream) task type
        if self.task_type == "captioning":
            self._on_test_epoch_captioning()
        else:
            raise ValueError("Task type is not supported.")

    @torch.no_grad()
    def _on_test_epoch_captioning(self):
        self.model.eval()

        test_dataloader = self.dataloader["test"]
        checkpoint = {
            "epoch": self.epoch,
            "cap_config": self.model.module.caption_head.cap_config if
            hasattr(self.model, 'module') else self.model.caption_head.cap_config
        }
        metrics = eval_language_metrics(checkpoint, test_dataloader, self.cfg, model=self.model)

        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info('\t>>>  Bleu_4: {:.2f} - METEOR: {:.2f} - ROUGE_L: {:.2f} - CIDEr: {:.2f}'.
                        format(metrics['Bleu_4'] * 100, metrics['METEOR'] * 100, metrics['ROUGE_L'] * 100,
                               metrics['CIDEr'] * 100))

            for metric, value in metrics.items():
                self.writer.add_scalar(f"test/{metric}", value * 100, global_step=self.epoch)


class Translator(object):
    """Load with trained model and handle the beam search"""

    def __init__(self, checkpoint, model=None):
        self.max_v_len = checkpoint['cap_config'].max_v_len
        self.max_t_len = checkpoint['cap_config'].max_t_len
        self.PAD = checkpoint['cap_config'].PAD_id
        self.BOS = checkpoint['cap_config'].BOS_id

        self.model = model
        self.model.eval()

        self.timer = Timer(synchronize=True, history_size=500, precision=3)

    def translate_batch_single_sentence_greedy(self, inputs, model):
        inputs_ids = inputs["input_ids"]
        input_masks = inputs["input_mask"]
        max_t_len = 77  # hard-code sentence length, for speed test, set it to 21
        inputs_ids[:, :] = 0.
        input_masks[:, :] = 0.
        assert torch.sum(input_masks[:, :]) == 0, "Initially, all text tokens should be masked"
        bsz = len(inputs_ids)
        next_symbols = torch.IntTensor([self.BOS] * bsz)  # (N, )

        self.timer.reset()
        warn_visual_output = False
        for dec_idx in range(max_t_len):
            inputs_ids[:, dec_idx] = next_symbols.clone()
            input_masks[:, dec_idx] = 1
            outputs = model(inputs)
            pred_scores = outputs["prediction_scores"]
            next_words = pred_scores[:, dec_idx].max(1)[1]
            next_symbols = next_words.cpu()
            if "visual_output" in outputs:
                inputs["visual_output"] = outputs["visual_output"]
            elif not warn_visual_output:
                logger.warning("visual_output is not in the output of model, this may slow down the caption test")
                warn_visual_output = True
        self.timer(stage_name="inference")
        return inputs_ids

    def translate_batch(self, model_inputs):
        """while we used *_list as the input names, they could be non-list for single sentence decoding case"""
        return self.translate_batch_single_sentence_greedy(model_inputs, self.model)


def convert_ids_to_sentence(tokens):
    from mm_video.layers.clip.clip import _tokenizer
    text = _tokenizer.decode(tokens)
    text_list = text.split(" ")
    new = []
    for i in range(len(text_list)):
        if i == 0:
            new.append(text_list[i].split(">")[-1])
        elif "<|endoftext|>" in text_list[i]:
            break
        else:
            new.append(text_list[i])
    return " ".join(new)


def run_translate(data_loader, translator, epoch, opt):
    # submission template
    batch_res = {"version": "VERSION 1.0",
                 "results": defaultdict(list),
                 "external_data": {"used": "true", "details": "ay"}}
    for bid, batch in enumerate(tqdm(data_loader,
                                     dynamic_ncols=True,
                                     disable=dist.is_initialized() and dist.get_rank() != 0,
                                     desc=f"Test: {epoch + 1}/{opt.TRAINER.TRAINER_BASE.EPOCH}")):
        if torch.cuda.is_available():
            batch = CudaPreFetcher.cuda(batch)
        dec_seq = translator.translate_batch(batch)

        # example_idx indicates which example is in the batch
        for example_idx, (cur_gen_sen, cur_meta) in enumerate(zip(dec_seq, batch['metadata'][1])):
            cur_data = {
                "sentence": convert_ids_to_sentence(cur_gen_sen.tolist()),
                "gt_sentence": cur_meta
            }
            batch_res["results"][batch['metadata'][0][example_idx].split("video")[-1]].append(cur_data)
    logger.debug(translator.timer.print())
    return batch_res


class EvalCap:
    def __init__(self, annos, rests, cls_tokenizer=PTBTokenizer,
                 use_scorers=('Bleu', 'METEOR', 'ROUGE_L', 'CIDEr')):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.annos = annos
        self.rests = rests
        self.Tokenizer = cls_tokenizer
        self.use_scorers = use_scorers

    def evaluate(self):
        res = {}
        for r in self.rests:
            res[str(r['image_id'])] = [{'caption': r['caption']}]

        gts = {}
        for imgId in self.annos:
            gts[str(imgId)] = [{'caption': c} for c in self.annos[imgId]]

        # =================================================
        # Set up scorers
        # =================================================
        # print('tokenization...')
        tokenizer = self.Tokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)
        # =================================================
        # Set up scorers
        # =================================================
        # print('setting up scorers...')
        use_scorers = self.use_scorers
        scorers = []
        if 'Bleu' in use_scorers:
            scorers.append((Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]))
        if 'METEOR' in use_scorers:
            scorers.append((Meteor(), "METEOR"))
        if 'ROUGE_L' in use_scorers:
            scorers.append((Rouge(), "ROUGE_L"))
        if 'CIDEr' in use_scorers:
            scorers.append((Cider(), "CIDEr"))
        if 'SPICE' in use_scorers:
            scorers.append((Spice(), "SPICE"))

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]


def evaluate(submission, reference):
    tokenizer = PTBTokenizer  # for English
    annos = reference
    data = submission['results']
    rests = []
    for name, value in data.items():
        rests.append({'image_id': str(name), 'caption': value[0]['sentence']})
    eval_cap = EvalCap(annos, rests, tokenizer)

    eval_cap.evaluate()

    all_score = {}
    for metric, score in eval_cap.eval.items():
        all_score[metric] = score
    return all_score


def eval_language_metrics(checkpoint, eval_data_loader, opt, model=None, eval_mode="test"):
    """eval_mode can only be set to `val` here, as setting to `test` is cheating
    0, run inference
    1, Get METEOR, BLEU1-4, CIDEr scores
    2, Get vocab size, sentence length
    """
    translator = Translator(checkpoint, model=model)
    json_res = run_translate(eval_data_loader, translator, checkpoint["epoch"], opt=opt)
    if dist.is_initialized():
        all_results = gather_object_multiple_gpu(list(json_res["results"].items()))
        json_res['results'] = {k: v for k, v in all_results}
        logger.debug("Caption test length: %s", len(json_res["results"].items()))

    # save result tp log for debug
    if not dist.is_initialized() or dist.get_rank() == 0:
        res_filepath = os.path.join(opt.LOG.DIR, "caption_greedy_pred_{}_{}.json".format(eval_mode, get_timestamp()))
        os.makedirs(os.path.dirname(res_filepath), exist_ok=True)
        save_json(json_res, res_filepath, save_pretty=True)

    if not dist.is_initialized() or dist.get_rank() == 0:
        json_ref = eval_data_loader.dataset.json_ref
        return evaluate(json_res, json_ref)
    else:
        return None

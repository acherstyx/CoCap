# -*- coding: utf-8 -*-
# @Time    : 2023/3/13 23:35
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : compute_cider.py

import argparse

from mm_video.utils.json import load_json
from mm_video.trainer.cocap_trainer import evaluate


def main(opt):
    pred = load_json(opt.pred_json)
    ref = load_json(opt.ref_json)

    if opt.lower:
        pred["results"] = {
            k: [{kk: vv.lower()[:-2] + "." if vv.endswith(" .") else vv.lower() for kk, vv in x.items()} for x in v]
            for k, v in pred["results"].items()
        }
        ref = {k: [vv.lower() for vv in v] for k, v in ref.items()}

    metrics = evaluate(submission=pred, reference=ref)

    print('>>>  Bleu_4: {:.2f} - METEOR: {:.2f} - ROUGE_L: {:.2f} - CIDEr: {:.2f}'.
          format(metrics['Bleu_4'] * 100, metrics['METEOR'] * 100, metrics['ROUGE_L'] * 100,
                 metrics['CIDEr'] * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Compute CIDEr")
    parser.add_argument("pred_json", type=str)
    parser.add_argument("ref_json", type=str)
    parser.add_argument("--lower", action="store_true")

    main(parser.parse_args())

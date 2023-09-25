# -*- coding: utf-8 -*-
# @Time    : 2022/11/25 15:16
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : checkpoint_tweak.py

import argparse
import os
import torch
from argparse import ArgumentParser


def extract(args):
    if not args.is_model:
        state_dict = torch.load(args.input, map_location="cpu")
        assert isinstance(state_dict, dict) and "model" in state_dict.keys()
        assert not os.path.exists(args.output), "Output file already exist at {}".format(args.output)
        torch.save(state_dict["model"], args.output)
    else:
        print("Extract only work on checkpoint.")


def remove(args):
    if args.is_model:
        state_dict = torch.load(args.input, map_location="cpu")
        state_dict = {k: v for k, v in state_dict.items() if not any(k.startswith(rm_k) for rm_k in args.rm_keys)}
        torch.save(state_dict, args.output)
    else:
        print("Remove only work on model.")


def info(args):
    state_dict = torch.load(args.input, map_location="cpu")
    for k in state_dict.keys():
        print(k)


def main():
    parser = ArgumentParser(description="Checkpoint tweaks")
    parser.add_argument("--is_model", action="store_true")

    subparser = parser.add_subparsers(title="function", required=True)

    extract_parser = subparser.add_parser("extract", help="Extract model parameter from checkpoint file")
    extract_parser.set_defaults(function=extract)
    extract_parser.add_argument("--input", "-i", required=True)
    extract_parser.add_argument("--output", "-o", required=True)

    remove_parser = subparser.add_parser("remove", help="Remove parameters with name start with the given keys")
    remove_parser.set_defaults(function=remove)
    remove_parser.add_argument("--input", "-i", required=True)
    remove_parser.add_argument("--output", "-o", required=True)
    remove_parser.add_argument("rm_keys", help="Keys (start with) to remove", default=[], nargs=argparse.ONE_OR_MORE)

    show_parser = subparser.add_parser("info", help="Show the information of state dict")
    show_parser.set_defaults(function=info)
    show_parser.add_argument("--input", "-i", required=True)

    args = parser.parse_args()
    args.function(args)


if __name__ == '__main__':
    main()

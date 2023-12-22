# -*- coding: utf-8 -*-
# @Time    : 2022/11/25 15:16
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : checkpoint_tweak.py

import os
import copy
import re
import fire
import torch
from typing import Dict


class ModelStates:
    def __init__(self, model_file: str, no_confirm: bool = False):
        state_dict = torch.load(model_file, map_location="cpu")
        self.history = []
        self.state_dict: Dict[str, torch.Tensor] = state_dict
        self.no_confirm: bool = no_confirm

    def _confirm(self):
        if not self.no_confirm:
            input("Press Enter to confirm...")

    def info(self):
        for k in self.state_dict.keys():
            print(k)
        return self

    def remove(self, pattern: str):
        print("Removing keys:")
        pattern = re.compile(pattern)
        for k in self.state_dict.keys():
            if pattern.match(k):
                print(f"\t{k}")
        self._confirm()

        self.history.append(copy.deepcopy(self.state_dict))
        self.state_dict = {k: v for k, v in self.state_dict.items() if not pattern.match(k)}
        return self

    def replace(self, pattern: str, repl: str):
        print("Replacing keys:")
        pattern = re.compile(pattern)
        for k in self.state_dict.keys():
            if pattern.match(k):
                print(f"\t{k} \t-> {pattern.sub(repl, k)}")
        self._confirm()

        self.history.append(copy.deepcopy(self.state_dict))
        self.state_dict = {pattern.sub(repl, k) if pattern.match(k) else k: v for k, v in self.state_dict.items()}

    def save(self, output_file: str, overwrite: bool = False):
        if not overwrite:
            assert not os.path.exists(output_file), "Output file already exist at {}".format(output_file)
        torch.save(self.state_dict, output_file)
        print(f"Model is saved to {output_file}")
        return self

    def undo(self):
        if self.history:
            self.state_dict = self.history[-1]
            self.history = self.history[:-1]
        else:
            print("No history.")
        return self

    def reset(self):
        if self.history:
            self.state_dict = self.history[0]
            self.history = []
        else:
            print("No changes is made to the state dict.")
        return self


if __name__ == '__main__':
    fire.Fire(ModelStates, name="Checkpoint tweaks")

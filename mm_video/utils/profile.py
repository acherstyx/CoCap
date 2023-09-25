# -*- coding: utf-8 -*-
# @Time    : 7/12/23
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : profile.py

import time
import logging
import numpy as np
import torch
from collections import defaultdict, deque
import functools
from tabulate import tabulate

logger = logging.getLogger(__name__)


def format_time(s: float) -> str:
    """Return a nice string representation of `s` seconds."""
    m = int(s / 60)
    s -= m * 60
    h = int(m / 60)
    m -= h * 60
    ms = int(s * 100000) / 100
    s = int(s * 100) / 100.0
    return ("" if h == 0 else str(h) + "h") + ("" if m == 0 else str(m) + "m") + ("" if s == 0 else str(s) + "s") + \
        (str(ms) + "ms" if s == 0 else "")


class Timer(object):
    def __init__(self, msg="", synchronize: bool = False, history_size: int = 1000, precision: int = 3):
        """

        :param msg:
        :param synchronize: Call `torch.cuda.synchronize()` when getting time
        :param history_size:
        :param precision: round seconds to a given precision in decimal digits to avoid verbose
        """
        self.msg = msg
        self.synchronize = synchronize
        self.precision = precision

        if self.msg:
            logger.info("%s", msg)

        self.start = self.get_time()
        self.last_checkpoint = self.start

        self.time_history = defaultdict(functools.partial(deque, maxlen=history_size))
        self.history_size = history_size

    def get_time(self):
        if self.synchronize and torch.cuda.is_available():
            torch.cuda.synchronize()
        return round(time.time(), self.precision)

    def reset(self):
        self.last_checkpoint = self.get_time()

    def __enter__(self):
        self.start = self.get_time()
        return self

    def __exit__(self, typ, value, traceback):
        self._duration = self.get_time() - self.start
        if self.msg:
            logger.info("%s [took %s]", self.msg, format_time(self._duration))

    def __call__(self, stage_name: str):
        current_time = self.get_time()
        duration = (current_time - self.last_checkpoint)
        self.last_checkpoint = current_time
        self.time_history[stage_name].append(duration)
        return duration

    def get_info(self, averaged=True):
        return {
            k: round(float(np.mean(v)), self.precision) if averaged else round(v[-1], self.precision)
            for k, v in self.time_history.items()
        }

    def __str__(self):
        return str(self.get_info())

    def print(self):
        data = [[k, format_time(np.mean(v).item())] for k, v in self.time_history.items()]
        return tabulate(data, headers=["Stage", "Time (ms)"], tablefmt="simple")


if __name__ == '__main__':
    with Timer("Running...") as f:
        time.sleep(1.12)

    timer = Timer()
    time.sleep(0.5)
    timer("s1")
    time.sleep(0.21)
    timer("s2")
    timer.print()
    print(timer)
    print(timer.get_info())

# -*- coding: utf-8 -*-
# @Time    : 8/2/23
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : test_train_utils.py

import unittest
import tempfile

import torch.cuda

from mm_video.utils.train_utils import gather_object_multiple_gpu


def run_gather_filesystem(proc):
    # require at least 2 GPU to run this test
    import torch.distributed as dist

    dist.init_process_group(backend="nccl", init_method="tcp://localhost:2222", world_size=2, rank=proc)
    torch.cuda.set_device(proc)
    # run
    temp_dir = "./test_data/temp"
    gathered = gather_object_multiple_gpu(
        [{"proc": proc, "data": f"some data from {proc}."}],
        backend="filesystem",
        shared_folder=temp_dir
    )
    if dist.get_rank() == 0:
        print(gathered)


class TestTrainUtils(unittest.TestCase):
    def test_gather_object_multiple_gpu(self):
        import torch.multiprocessing as mp
        mp.spawn(run_gather_filesystem, nprocs=2)


if __name__ == '__main__':
    unittest.main()

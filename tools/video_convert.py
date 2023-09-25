# -*- coding: utf-8 -*-
# @Time    : 2022/9/27 14:15
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : video_convert.py

"""convert video to h264/h265 format"""

import os
import joblib
import tqdm
import argparse
from mm_video.utils.video import convert_video


def main():
    parser = argparse.ArgumentParser(description="Convert videos to h264/h265 format in parallel")
    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("-n", "--num_process", type=int, default=os.cpu_count())
    parser.add_argument("--shard_id", type=int, default=None)
    # encode opt
    parser.add_argument("--codec", type=str, default="libx264", choices=["libx264", "libx265"])
    parser.add_argument("--keyint", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--resize", type=int, default=None)
    args = parser.parse_args()

    if args.shard_id is not None:
        shard_folders = sorted(list(os.listdir(args.input_dir)))
        args.input_dir = os.path.join(args.input_dir, shard_folders[args.shard_id])
        args.output_dir = os.path.join(args.output_dir, shard_folders[args.shard_id])

    video_files = []
    for f in os.listdir(os.path.join(args.input_dir)):
        video_files.append((os.path.join(args.input_dir, f), os.path.join(args.output_dir, f)))

    runner = joblib.Parallel(n_jobs=args.num_process, return_as="generator")(
        joblib.delayed(convert_video)(f_in, f_out,
                                      codec=args.codec,
                                      keyint=args.keyint,
                                      overwrite=args.overwrite,
                                      verbose=args.verbose,
                                      resize=args.resize)
        for f_in, f_out in video_files
    )

    for _ in tqdm.tqdm(
            runner,
            total=len(video_files),
            dynamic_ncols=True,
            desc=f"Shard {args.shard_id + 1}/{len(shard_folders)}" if args.shard_id is not None else ""
    ):
        pass


if __name__ == '__main__':
    main()

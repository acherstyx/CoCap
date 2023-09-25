# -*- coding: utf-8 -*-
# @Time    : 2023/2/21 21:48
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : check_video_integrity.py

import argparse
import json
import os.path
import joblib
import cv2
import tqdm


def check_video(video_file):
    try:
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            return False, video_file, "cannot open"
    except cv2.error:
        return False, video_file, "cv2 error"
    except Exception as e:
        return False, video_file, "other exception: {}".format(e)
    else:
        return True, video_file, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Check the integrity of video files.")
    parser.add_argument("--video_root", type=str, help="Dir", required=True)
    parser.add_argument("--output", "-o", type=str, help="Output check result in json format",
                        default="video_corrupt_check.json")
    parser.add_argument("--failed_only", action="store_true")
    args = parser.parse_args()

    # TODO: walk entire dir and filter by extension, support multiple video_root
    video_list = [os.path.join(args.video_root, f) for f in os.listdir(args.video_root)]

    runner = joblib.Parallel(n_jobs=os.cpu_count(), return_as="generator")(
        joblib.delayed(check_video)(f) for f in video_list
    )
    ret = list(tqdm.tqdm(runner, total=len(video_list)))

    with open(args.output, "w") as f:
        # TODO: better output format
        ret = {
            os.path.relpath(file, start=args.video_root): {
                "success": success,
                "error_msg": error_msg,
                "full_path": os.path.realpath(file)
            } for success, file, error_msg in ret if not args.failed_only or not success
        }
        json.dump(ret, f, indent=4)

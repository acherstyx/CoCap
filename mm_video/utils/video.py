# -*- coding: utf-8 -*-
# @Time    : 8/7/23
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : video.py


from typing import *

import cv2
import os
import subprocess
from joblib import Parallel, delayed

__all__ = ["get_duration_info", "convert_video"]


def _get_single_video_duration_info(video_path) -> (float, float, int):
    """
    return video duration in seconds
    :param video_path: video path
    :return: video duration, fps, frame count
    """
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    return frame_count / fps, fps, int(frame_count)


def get_duration_info(video_paths: Union[str, Iterable]) -> (float, float, int):
    """

    :param video_paths: video path or a list of video path
    :return: video duration, fps, frame count
    """
    if isinstance(video_paths, str):
        return _get_single_video_duration_info(video_paths)
    else:
        return Parallel(n_jobs=os.cpu_count())(
            delayed(_get_single_video_duration_info)(path) for path in video_paths
        )


def convert_video(input_file: AnyStr, output_file: AnyStr,
                  ffmpeg_exec: AnyStr = "/usr/bin/ffmpeg",
                  codec="libx264",
                  keyint: int = None,
                  overwrite: bool = False,
                  verbose: bool = False,
                  resize: tuple = None) -> None:
    """
    :param input_file:
    :param output_file:
    :param ffmpeg_exec:
    :param codec: supported video codec, e.g., libx264 and libx265
    :param keyint:
    :param overwrite:
    :param verbose:
    :param resize:
    """
    assert codec is None or codec in ["libx264", "libx265"], "Video codec {} is not supported.".format(codec)
    assert keyint is None or codec is not None, "Codec must be specified if keyint is not None."

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    command = [ffmpeg_exec, "-i", f"{input_file}", "-max_muxing_queue_size", "9999"]
    if codec is not None:
        # use specified codec
        command += ["-c:v", codec]
        if codec == "libx265":
            command += ["-vtag", "hvc1"]
        if keyint is not None and codec == "libx264":
            command += ["-x264-params", f"keyint={keyint}"]
        elif keyint is not None and codec == "libx265":
            command += ["-x265-params", f"keyint={keyint}"]
    if resize is not None:
        if isinstance(resize, int):  # resize height
            assert resize % 2 == 0, "size is not divisible by 2"
            command += [
                "-vf",
                f"scale='if(gt(ih,iw),{resize},trunc(oh*a/2)*2)':'if(gt(ih, iw),trunc(ow/a/2)*2,{resize})'"
            ]
        elif (isinstance(resize, type) or isinstance(resize, list)) and len(resize) == 2:
            assert isinstance(resize[0], int) and isinstance(resize[1], int), "size should be int"
            assert resize[0] % 2 == 0 and resize[1] % 2 == 0, "size is not divisible by 2"
            command += ["-vf", f"scale={resize[0]}:{resize[1]}"]
        else:
            raise ValueError("size is not supported: {}".format(resize))
    command += ["-c:a", "copy", "-movflags", "faststart", f"{output_file}"]

    if overwrite:
        command += ["-y"]
    else:
        command += ["-n"]
    subprocess.run(command,
                   stderr=subprocess.DEVNULL if not verbose else None,
                   stdout=subprocess.DEVNULL if not verbose else None)
    # TODO: return

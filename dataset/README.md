# Data Preparation

This readme provides step-by-step instructions on how to prepare the data for the experiments.
By following the steps outlined below, you will obtain a file structure similar to the one shown below:

```text
dataset
├── README.md
├── caption_annotations.tar.gz
├── msrvtt
│   ├── MSRVTT_data.json
│   ├── videos
│   │   ├── video0.mp4
│   │   ├── ...
│   │   └── video9.mp4
│   └── videos_h264_keyint_60
│       ├── video0.mp4
│       ├── ...
│       └── video9.mp4
├── msvd
│   ├── MSVD_caption.json
│   ├── videos
│   │   ├── 00jrXRMlZOY_0_10.mp4
│   │   ├── ...
│   │   └── zzit5b_-ukg_5_20.mp4
│   ├── videos_240_h264_keyint_60
│   │   ├── 00jrXRMlZOY_0_10.mp4
│   │   ├── ...
│   │   └── zzit5b_-ukg_5_20.mp4
│   └── videos_h264
└── vatex
    ├── VATEX_caption.json
    ├── videos
    │   ├── 009PvqRCVII_000291_000301.mp4
    │   ├── ...
    │   └── zZZohOGP3JA_000002_000012.mp4
    └── videos_240_h264_keyint_60
        ├── 009PvqRCVII_000291_000301.mp4
        ├── ...
        └── zZZohOGP3JA_000002_000012.mp4
```

## 1. Download Annotations

We provide download links for the annotations required for the caption task. 

[\[Google Drive\]](https://drive.google.com/file/d/1uBT6tP_TZIxZwQo0lnkybURXpL8nLaNk/view?usp=sharing)

Once downloaded, move the tarball to the current directory and execute the following command:

```shell
cd dataset # make sure you are in the correct directory
tar -xvf caption_annotations.tar.gz
```

## 2. Download Videos

We do not host the video files for download. You will need to download the videos for each caption dataset from their official homepage or other sources on the internet. Here we provide some sources we could find for downloading.

After downloading, place all the videos in the `./dataset/{msrvtt,msvd,vatex}/videos` folder in a **FLAT** file structure.

- **MSRVTT**: As discussed in the [issue](https://github.com/VisionLearningGroup/caption-guided-saliency/issues/6#issuecomment-342368239), you can download the videos from https://www.mediafire.com/folder/h14iarbs62e7p/shared.
- **MSVD**: You can download the videos from https://www.cs.utexas.edu/users/ml/clamp/videoDescription/.
- **VATEX**: Download the videos from the official page of [VATEX](https://eric-xw.github.io/vatex-website/download.html). 

## 3. Convert and Crop the Videos

Next, you need to convert the videos to the H.264 format and crop them based on the short edge. We provide a script that allows you to quickly convert the videos in parallel.

Usage:

```text
usage: video_convert.py [-h] -i INPUT_DIR -o OUTPUT_DIR [-n NUM_PROCESS]
                        [--shard_id SHARD_ID] [--codec {libx264,libx265}] [--keyint KEYINT]
                        [--overwrite] [--verbose] [--resize RESIZE]

Convert videos to h264/h265 format in parallel

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_DIR, --input_dir INPUT_DIR
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
  -n NUM_PROCESS, --num_process NUM_PROCESS
  --shard_id SHARD_ID
  --codec {libx264,libx265}
  --keyint KEYINT
  --overwrite
  --verbose
  --resize RESIZE
```

To convert the videos for each dataset, run the following commands in the project root:

```bash
# msrvtt
python3 tools/video_convert.py --codec=libx264 --keyint=60 --resize=240 -i dataset/msrvtt/videos -o dataset/msrvtt/videos_h264_keyint_60
# msvd
python3 tools/video_convert.py --codec=libx264 --keyint=60 --resize=240 -i dataset/msvd/videos -o dataset/msvd/videos_240_h264_keyint_60
# vatex
python3 tools/video_convert.py --codec=libx264 --keyint=60 --resize=240 -i dataset/vatex/videos -o dataset/vatex/videos_240_h264_keyint_60
```

This requires ffmpeg to be installed. We recommend installing the latest version (>=5.0) to avoid failures during the conversion process.

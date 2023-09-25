# Accurate and Fast Compressed Video Captioning

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/accurate-and-fast-compressed-video-captioning/video-captioning-on-msr-vtt-1)](https://paperswithcode.com/sota/video-captioning-on-msr-vtt-1?p=accurate-and-fast-compressed-video-captioning)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/accurate-and-fast-compressed-video-captioning/video-captioning-on-msvd-1)](https://paperswithcode.com/sota/video-captioning-on-msvd-1?p=accurate-and-fast-compressed-video-captioning)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/accurate-and-fast-compressed-video-captioning/video-captioning-on-vatex-1)](https://paperswithcode.com/sota/video-captioning-on-vatex-1?p=accurate-and-fast-compressed-video-captioning)


✨This is the official implementation of ICCV 2023 paper *[Accurate and Fast Compressed Video Captioning](https://arxiv.org/abs/2309.12867)*.

## Introduction

In this work, we propose an end-to-end video captioning method based on compressed domain information from the encoded H.264 videos. Our approach aims to accurately generate captions for compressed videos in a fast and efficient manner.

![Framework](./assets/framework.svg)

By releasing this code, we hope to facilitate further research and development in the field of compressed video processing. If you find this work useful in your own research, please consider citing our paper as a reference.

## Preparation

### 1. Install the Requirements

To run the code, please install the dependency libraries by using the following command:

```shell
sudo apt update && sudo apt install default-jre -y  # required by pycocoevalcap
pip3 install -r requirements.txt
```

Additionally, you will need to install the compressed video reader as described in the README.md of [AcherStyx/Compressed-Video-Reader](https://github.com/AcherStyx/Compressed-Video-Reader).


### 2. Prepare the Pretrained Models

Our model is based on the pretrained CLIP. You can run the following script to download the weights before training to avoid any network issues:

```bash
sudo apt update && sudo apt install aria2 -y  # install aria2
bash model_zoo/download_model.sh
```

This will download the CLIP model to `model_zoo/clip_model`. Note that this directory is hard-coded in our code.

### 3. Prepare the Data

We have conducted experiments on three video caption datasets: MSRVTT, MSVD, and VATEX. The datasets are stored in the `dataset` folder under the project root. For detailed instructions on downloading and preparing the training data, please refer to [dataset/README.md](./dataset/README.md).

## Training & Evaluation

The training is configured using YAML, and all the configurations are listed in [`configs/compressed_video`](./configs/compressed_video). You can use the following commands to run the experiments:

```shell
# msrvtt
python3 mm_video/run_net.py --cfg configs/compressed_video/msrvtt_captioning.yaml
# msvd
python3 mm_video/run_net.py --cfg configs/compressed_video/msvd_captioning.yaml
# vatex
python3 mm_video/run_net.py --cfg configs/compressed_video/vatex_captioning.yaml
```

By default, the logs and results will be saved to `./log/<experiment_name>/`. The loss and metrics are visualized using tensorboard.

## Citation

```text
@inproceedings{shen2023accurate,
      title={Accurate and Fast Compressed Video Captioning}, 
      author={Yaojie Shen and Xin Gu and Kai Xu and Heng Fan and Longyin Wen and Libo Zhang},
      booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
      year={2023}
}
```

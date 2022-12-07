# DeepInfrared

**This page is under construction, not finished yet.**

DeepInfrared aims to be an open benchmark for infrared small target detection, currently consisting of:

1. Public infrared small target dataset (SIRST-V2);
2. Evaluation metrics specially designed (mNoCoAP);
3. An open source toolbox based on PyTorch (DeepInfrared).

## SIRST-V2 Dataset

As a part of the DeepInfrared Eco-system, we provide the SIRST-V2 dataset as a benchmark.
SIRST-V2 is a dataset specially constructed for single-frame infrared small target detection, in which the images are selected from thousands of infrared sequences for different scenarios.

![](https://github.com/YimianDai/open-sirst-v2/blob/master/gallery.jpg)

Annotation formats available:

- bounding box;
- semantic segmentation;
- normalized contrast (produced when data loading).

The dataset can be downloaded [here](https://github.com/YimianDai/open-sirst-v2).

## The DeepInfrared Toolkit

### Installation

Please refer to [Installation](https://github.com/YimianDai/open-deepinfrared/blob/master/docs/INSTALL.md) for installation instructions.

### Getting Started

#### Train

```shell
# assume that you are under the root directory of this project,
# and you have activated your virtual environment if needed.
# and with SIRST-V2 dataset in 'data/sirst/'

python tools/train_det.py \
    configs/oscar/sota/oscar_w_noco_head_r18_caffe_fpn_p2_gn-head_1x_sirst_det2noco.py \
    --gpu-id 0 \
    --work-dir work_dirs/oscar_w_noco_head_r18_caffe_fpn_p2_gn-head_1x_sirst_det2noco

```

#### Inference

```shell
python tools/test_det.py \
    configs/oscar/sota/oscar_w_noco_head_r18_caffe_fpn_p2_gn-head_1x_sirst_det2noco.py \
    work_dirs/oscar_w_noco_head_r18_caffe_fpn_p2_gn-head_1x_sirst_det2noco/best.pth --eval "mNoCoAP"
```

### Overview of Benchmark and Model Zoo

For your convenience, we provide the following trained models.

Model | mNoCoAP | Config | Log | GFLOPS | Download
--- |:---:|:---:|:---:|:---:|:---:
faster_rcnn_r50_fpn_1x | 0.7141 | [config](https://raw.githubusercontent.com/YimianDai/deepinfrared-files/master/faster_rcnn_r50_fpn_1x_sirst_0_7141/faster_rcnn_r50_fpn_1x_sirst.py) | [log](https://raw.githubusercontent.com/YimianDai/deepinfrared-files/master/faster_rcnn_r50_fpn_1x_sirst_0_7141/20221201_041954.log) | | [baidu](https://pan.baidu.com/s/1fzgl2kJbcve4LC6tklGMYA?pwd=dv7b)
fcos_rfla_r50_kld_1x | 0.7882 | [config](https://raw.githubusercontent.com/YimianDai/deepinfrared-files/master/fcos_rfla_r50_kld_1x_0_7882/sirstv2_fcos_rfla_r50_kld_1x.py) | [log](https://raw.githubusercontent.com/YimianDai/deepinfrared-files/master/fcos_rfla_r50_kld_1x_0_7882/20221126_152729.log) | | [baidu](https://pan.baidu.com/s/1-JU-CA5a7FmEr0TRvXgh5Q?pwd=7gu6)
oscar_r18_fpn_p2_128_1x | 0.8352 | [config](https://raw.githubusercontent.com/YimianDai/deepinfrared-files/master/oscar_w_noco_head_r18_caffe_fpn_p2_128_gn-head_1x_sirst_det2noco_0_8352/oscar_w_noco_head_r18_caffe_fpn_p2_128_gn-head_1x_sirst_det2noco.py) | [log](https://raw.githubusercontent.com/YimianDai/deepinfrared-files/master/oscar_w_noco_head_r18_caffe_fpn_p2_128_gn-head_1x_sirst_det2noco_0_8352/20221203_034804.log) | 25.36 | [baidu](https://pan.baidu.com/s/1y5jQGZbPiPFm-FPvBydSCQ?pwd=rwyk)
oscar_r18_fpn_p2_256_1x | 0.8502 | [config](https://raw.githubusercontent.com/YimianDai/deepinfrared-files/master/oscar_w_noco_head_r18_caffe_fpn_p2_gn-head_1x_sirst_det2noco_0_8502/oscar_w_noco_head_r18_caffe_fpn_p2_gn-head_1x_sirst_det2noco.py) | [log](https://raw.githubusercontent.com/YimianDai/deepinfrared-files/master/oscar_w_noco_head_r18_caffe_fpn_p2_gn-head_1x_sirst_det2noco_0_8502/20221201_145722.log) | 68.32 | [baidu](https://pan.baidu.com/s/1JD5-6sb8Y-0tzOsGyzxebQ?pwd=pdj5)

## FAQ

## Acknowledgement

## Citation


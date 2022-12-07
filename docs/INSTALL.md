## Installation

### Requirements

- Python 3.7
- PyTorch 1.12.1
- MMCV 1.7.0
- MMDetection 2.25.2
- MMSegmentation 0.29.1

### Install deepinfrared

```shell
git clone git@github.com:YimianDai/open-deepinfrared.git
cd open-deepinfrared
python setup.py develop
```

### A from-scratch setup script

```shell
conda create -n deepir python=3.7 -y
conda activate deepir

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install openmim
mim install mmcv-full
mim install mmdet
mim install mmsegmentation

git clone git@github.com:YimianDai/open-deepinfrared.git
cd open-deepinfrared
python setup.py develop
```
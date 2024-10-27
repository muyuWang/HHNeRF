# High-fidelity and High-efficiency Talking Portrait Synthesis with Detail-aware Neural Radiance Fields

This repository provides a PyTorch implementation for the paper: [High-fidelity and High-efficiency Talking Portrait Synthesis with Detail-aware Neural Radiance Fields](https://arxiv.org/abs/2211.12368).


A **self-driven** generated video of our method:
[here](./results/Cameron.mp4)

A **cross-driven** generated video of our method:
[here](./results/Sunak.mp4)

# Installation

Tested on Ubuntu 22.04, Pytorch 2.0.1 and CUDA 11.6.

```bash
git clone https://github.com/muyuWang/HHNeRF.git
cd HHNeRF
```

### Install dependency
```bash
# for ubuntu, portaudio is needed for pyaudio to work.
sudo apt install portaudio19-dev

pip install -r requirements.txt
```

# Data pre-processing
Our data preprocessing method follows previous work [AD-NeRF](https://github.com/YudongGuo/AD-NeRF), [SSP-NeRF](https://github.com/alvinliu0/SSP-NeRF) and [RAD-NeRF](https://github.com/ashawkey/RAD-NeRF/tree/main).
We provide some HR videos in 900 * 900 resolution. In data preprocessing, please downsample them to 450 * 450. Then use the downsampled frames to perform data preprocessing in [RAD-NeRF](https://github.com/ashawkey/RAD-NeRF/tree/main) (extract images, detect lands,  face parsing, extract background, estimate head pose ...). With the extracted landmarks, extract patches from the eye region and then utilize a ResNet model to extract their features. 

* Finally, file structure after finishing all steps:
    ```bash
    ./data/<ID>
    ├──<ID>.mp4 # original video
    ├──ori_imgs # original images from video
    │  ├──0.png
    │  ├──0.lms # 2D landmarks
    │  ├──...
    ├──hr_imgs # HR ground truth frames (static background)
    │  ├──0.jpg
    │  ├──...
    ├──eye_features # eye pathes and features
    │  ├──0_l.png # left eye
    │  ├──0_r.png # right eye
    │  ├──0.pt # eye feature
    │  ├──...
    ├──gt_imgs # ground truth images (static background)
    │  ├──0.jpg
    │  ├──...
    ├──parsing # semantic segmentation
    │  ├──0.png
    │  ├──...
    ├──torso_imgs # inpainted torso images
    │  ├──0.png
    │  ├──...
    ├──aud.wav # original audio 
    ├──aud.npy # audio features (deepspeech)
    ├──bc.jpg # default background
    ├──track_params.pt # raw head tracking results
    ├──transforms_train.json # head poses (train split)
    ├──transforms_val.json # head poses (test split)
    ```

Some HR talking videos and processed data can be downloaded at [baidudisk](https://pan.baidu.com/s/1iR5Q3xJ2n3KYfKS9XPoA8Q?pwd=hy7i). 

# Usage

The training script is in `train.sh.`. Here is an example.

Training DaNeRF module:
```bash
python main.py data/Sunak/ --workspace trial/Sunak/ -O --iters 70000 --data_range 0 -1 --dim_eye 6 --lr 0.005 --lr_net 0.0005 --num_rays 65536 --patch_size 32
```

Training DaNeRF and ECSR jointly:
```bash
python main_sr.py data/Sunak/ --workspace trial/Sunak/ -O --iters 150000 --data_range 0 -1 --dim_eye 6 --patch_size 32 --srtask --num_rays 16384 --lr 0.005 --lr_net 0.0005 --weight_pcp 0.05 --weight_style 0.01 --weight_gan 0.01 --test_tile 450
```
with ckpt use ` --ftsr_path  'trial/Sunak/modelsr_ckpt/sresrnet_17.pth' `.


# Acknowledgement

This project is developed based on [RAD-NeRF](https://github.com/ashawkey/RAD-NeRF/tree/main) of Tang et al and [4K-NeRF](https://github.com/frozoul/4K-NeRF) of Wang et al. Thanks for these great works.



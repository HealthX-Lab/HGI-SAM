# HGI-SAM
# Weakly Supervised Intracranial Hemorrhage Segmentation using Head-Wise Gradient-Infused Self-Attention Maps from a Swin Transformer in Categorical Learning

<p>
    <img src="https://github.com/ah-rasoulian/HGI-SAM/blob/master/extra/images/model.png"  alt="model"/>
    <img src="https://github.com/ah-rasoulian/HGI-SAM/blob/master/extra/images/attention_map_generation.png" alt="attention_map_generation"/>
</p>

This repository contains the code used for the paper: [Weakly Supervised Intracranial Hemorrhage Segmentation using Head-Wise Gradient-Infused Self-Attention Maps from a Swin Transformer in Categorical Learning.](https://arxiv.org/abs/2304.04902)

Please cite the paper if you are using our model.

## Installation

### Data
You can download the two datasets used in this study from the following links:
1) [RSNA Intracranial Hemorrhage Dataset](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/data)
2) [PhysioNet Intracranial Hemorrhage Dataset](https://physionet.org/content/ct-ich/1.3.1/)

For the codes using PhysioNet dataset (train fully supervised UNet and inference code) to be able to run, brain-masks of scans are needed. A zip file of such directory is provided in `install` folder.
It has to be unzipped and put in the PhysioNet dataset root directory.

### Setup environment
1) Clone the repository. Note that in order to download the weights of trained models, Git LFS must be installed.
2) Install required python libraries. `environment.yml` file located in `install` directory contains the name of all dependencies.

## Usage of the code
if only interested to do the inference, please skip to [Section 2 - Inference](#2-inference). However, if you intend to execute the complete pipeline, you should start from [Section 1 - Train Fully Supervised UNet](#1-train-fully-supervised-unet).

### 1) Train Fully Supervised UNet

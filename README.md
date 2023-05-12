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

For the codes using PhysioNet dataset (fully supervised UNet and inference code) to be able to run, a directory containing brain-masks of scans must be put in the PhysioNet dataset root directory.
# HGI-SAM

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
if only interested to do the inference, please skip to [Section 2 - Inference](#2-inference). However, if you intend to execute the complete pipeline, you should start from [Section 1 - Train Models](#1-train-models).

### 1) Train models
The first step of the pipeline is training the models.

#### a) Weakly Supervised Swin Transformer (our proposed technique)
The script to train the Swin-Transformer model for Intracranial Hemorrhage detection is `train_rsna.py`, and is associated with [configs/train_rsna_config.json](https://github.com/ah-rasoulian/HGI-SAM/blob/master/configs/train_rsna_config.json).
In the config file, if the field `"do_finetune"` is set to `"True"`, the weights of the trained model in our [MLCN paper](https://link.springer.com/chapter/10.1007/978-3-031-17899-3_7) will be used for model initialization.
Otherwise, the model is trained from scratch. Also, make sure to change the `"data_path"` field to the RSNA-ICH dataset root directory.

Then, run the script with:

    python train_rsna.py configs\train_rsna_config.json

#### b) Fully Supervised UNet
The script to train the UNet model is `train_physionet.py`, and is associated with [configs/train_physionet_config.json](https://github.com/ah-rasoulian/HGI-SAM/blob/master/configs/train_physionet_config.json).
It divides the PhysioNet dataset into 5 balanced folds, and performs a 5-fold cross-validation to train the UNet model fully supervised.

Before running the script, make sure to change the `"data_path"` field in config file to the PhysioNet dataset root directory. Then, run the script with:

    python train_physionet.py configs\train_physionet_config.json

### 2) Inference
The second step of the pipeline is doing the inference. The script used to carry out the inference is `inference.py`, and the required configs should be specified in [configs/inference_config.json](https://github.com/ah-rasoulian/HGI-SAM/blob/master/configs/inference_config.json).
It does the detection and segmentation of all models on different folds defined in [extra/folds_division](https://github.com/ah-rasoulian/HGI-SAM/tree/master/extra/folds_division), and prints the results.

Before running the script, you should modify the paths to PhysioNet dataset directory, and trained models weights.
Then, the script can be run with:

    python inference.py configs\inference_config.json

In the config file, if you define the `"save_visualization": "True"`, segmentation prediction images of different models along with their ground-truth will be stored in the specified directory.
Also, segmentation results of all samples will be saved in the csv file defined in `"csv_seg_results_path":?` argument.

If you are unable to run the code, feel free to reach out [ah.rasoulian@gmail.com](mailto:ah.rasoulian@gmail.com) 
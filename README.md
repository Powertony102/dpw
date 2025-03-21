# Data Programming Workshop Project

## Introduction

### Model structure

![](/figure/net.png)

## Project Structure

```
.
├── datasets/
│   └── UCI_HAR_Dataset/     # UCI HAR Dataset files
├── models/                  # Saved model checkpoints
├── train_mhattention.py    # Train
├── test.py                 # Test
├── model/                  # Saved model
│   └── /exp_mh_attention_epoch_200/  # Model implementations
├── requirements.txt        # Dependencies
└── README.md              # Documentation
```

## Usage

### Environment
This repository is implemented in CUDA 12.1. To install the requirment, you can do following steps: 

```
conda create -n dpw python=3.11
conda activate dpw
pip install -r requirements.txt
```

### Train the model
```
# Use default parameters
python train_mhattention.py

# Examples if you use personal parameters
python train_mhattention.py --dataset_name UCI_HAR --exp mh_attention --gpu 0 --epoch 200 --deterministic 1 --seed 42
```

### Test the model
```
python test.py
```
# Data Programming Workshop Project

## Introduction

### Method
We design our **SIGMA-Net (Squeeze-Integrated GRU Multihead Attention Network)** for Human Activity Recognition task.

The model architecture combines GRU (Gated Recurrent Unit), SE (Squeeze-and-Excitation) Block, and Multi-Head Attention mechanisms:

1. **GRU Layer**: Processes sequential sensor data and captures temporal dependencies
2. **SE Block**: Adaptively recalibrates channel-wise feature responses
3. **Multi-Head Attention**: Enables the model to focus on different parts of the input sequence simultaneously

This hybrid design enhances feature extraction and temporal relationship learning for human activity recognition.

### Model structure

![](/figure/net.png)

## Dataset and Exploratory Data Analysis

We use the UCI HAR (Human Activity Recognition) Dataset, which contains smartphone sensor data collected from 30 volunteers performing six activities:
- WALKING
- WALKING_UPSTAIRS
- WALKING_DOWNSTAIRS
- SITTING
- STANDING
- LAYING

The dataset includes tri-axial acceleration and angular velocity measurements captured by the smartphone's accelerometer and gyroscope.

### Data Visualization

We have performed comprehensive exploratory data analysis on the UCI HAR Dataset to understand the patterns and characteristics of different activities. Our analysis includes:

- Signal distributions across different activities
- Feature importance analysis
- Temporal patterns in sensor data
- Class distribution visualization

For detailed visualizations and insights, please refer to the Jupyter notebooks in the `data_visualization` directory:
- `data_visualization.ipynb`: Contains visualizations of sensor data patterns
- `EDA.ipynb`: Provides exploratory data analysis and statistical insights

## Project Structure

```
.
├── datasets/
│   └── UCI_HAR_Dataset/     # UCI HAR Dataset files
├── models/                  # Saved model checkpoints
├── train_mhattention.py     # Training script
├── test.py                  # Test script
├── model/                   # Saved model
│   └── exp_mh_attention_epoch_200/  # Model implementations
├── data_visualization/      # Data analysis and visualization
│   ├── data_visualization.ipynb  # Sensor data visualizations
│   ├── data_visualization.pdf    # Exported visualizations
│   └── EDA.ipynb                 # Exploratory data analysis
├── requirements.txt         # Dependencies
└── README.md                # Documentation
```

## Usage

### Environment
This repository is implemented in CUDA 12.1. To install the requirements, you can follow these steps: 

```
conda create -n dpw python=3.11
conda activate dpw
pip install -r requirements.txt
```

### Data Exploration
To explore the dataset and visualizations:
```
jupyter notebook data_visualization/EDA.ipynb
# or
jupyter notebook data_visualization/data_visualization.ipynb
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

## Model Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `--dataset_name` | Name of the dataset | UCI_HAR |
| `--exp` | Experiment name | mh_attention |
| `--gpu` | GPU ID to use | 0 |
| `--epoch` | Number of training epochs | 200 |
| `--deterministic` | Use deterministic training | 1 |
| `--seed` | Random seed | 42 |

## Results

Our SIGMA-Net achieves state-of-the-art performance on the UCI HAR dataset. The model demonstrates strong ability to distinguish between similar activities like walking, walking upstairs, and walking downstairs, which are traditionally challenging to differentiate.

Key performance metrics:
- High accuracy across all activity classes
- Robust performance with minimal fine-tuning
- Effective feature extraction from raw sensor data
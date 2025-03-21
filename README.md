# Data Programming Workshop Project

## Project Structure

```
.
├── src/
│   ├── controllers/     # Request handlers
│   ├── models/         # Data models
│   └── services/       # Business logic
├── tests/             # Test files
└── config/            # Configuration files
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
import os
import torch
import argparse
import logging
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch.nn.functional as F

# Import model and dataset classes
from train_mhattention import EnhancedAttentionGRUModel, HumanActivityDataset, __load_X, LABELS, SIGNALS

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='UCI_HAR', help='dataset_name')
parser.add_argument('--exp', type=str, default='mh_attention', help='exp_name')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--epoch', type=int, default='180', help='epoch number')
args = parser.parse_args()

snapshot_path = "./model/exp_{}_epoch_{}/{}".format(args.exp, args.epoch, args.dataset_name)
model_save_path = "./model/exp_{}_epoch_{}/{}".format(args.exp, args.epoch, args.dataset_name)
model_path = model_save_path + "/best_model.pth"

# Ensure the directory exists
if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cuda:" + str(1 - int(args.gpu)))
print(f"Using device: {device}")

# Load data
train_paths = ['/home/jovyan/work/dpw/data/UCI_HAR_Dataset/train/Inertial Signals/' + signal + 'train.txt' for signal in SIGNALS]
test_paths = ['/home/jovyan/work/dpw/data/UCI_HAR_Dataset/test/Inertial Signals/' + signal + 'test.txt' for signal in SIGNALS]

X_train = __load_X(train_paths)
X_test = __load_X(test_paths)
y_train = np.loadtxt('/home/jovyan/work/dpw/data/UCI_HAR_Dataset/train/y_train.txt', dtype=np.int32)
y_test = np.loadtxt('/home/jovyan/work/dpw/data/UCI_HAR_Dataset/test/y_test.txt', dtype=np.int32)

# Ensure label values are in the range [0, num_classes-1]
y_train = y_train - 1
y_test = y_test - 1

# Create datasets and data loaders
train_dataset = HumanActivityDataset(X_train, y_train)
test_dataset = HumanActivityDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model
input_size = X_train.shape[2]
hidden_size = 128
num_layers = 2
num_classes = len(np.unique(y_train))
num_heads = 4  # Number of heads in multi-head attention

model = EnhancedAttentionGRUModel(input_size, hidden_size, num_layers, num_classes, num_heads)
model.to(device)

# Load the trained model weights
model.load_state_dict(torch.load(model_path))
model.eval()

def test():
    # Test the model, output 6 decimal places
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    all_attention_weights = []  # Store all attention weights
    
    try:
        with torch.no_grad():
            with tqdm(test_loader, desc='Testing') as pbar:
                for inputs, labels in pbar:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs, attention_weights, _, _ = model(inputs)  # The model now returns four values
                    _, preds = torch.max(outputs, 1)
                    
                    total += labels.size(0)
                    correct += (preds == labels).sum().item()
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_attention_weights.append(attention_weights.cpu().numpy())  # Store attention weights
                    
                    # Update progress bar
                    pbar.set_postfix({'acc': f'{100 * correct / total:.2f}%'})
        
        # Calculate and print model performance metrics
        accuracy = 100 * correct / total
        report = classification_report(all_labels, all_preds, digits=6)
        conf_mat = confusion_matrix(all_labels, all_preds)

        # Save results to image file /result
        df = pd.DataFrame(conf_mat, index=LABELS, columns=LABELS)
        df.to_csv(snapshot_path + '/confusion_matrix.csv')
        
        # Generate attention mechanism heatmap
        all_attention_weights = np.concatenate(all_attention_weights, axis=0)
        all_attention_weights = all_attention_weights[:len(all_labels)]
        
        # Reshape attention weights, reduce time step density
        # Group attention weight matrix by class
        attention_by_class = {}
        for i, label_val in enumerate(all_labels):
            label = int(label_val)  # Ensure the label is an integer type
            if label not in attention_by_class:
                attention_by_class[label] = []
            attention_by_class[label].append(all_attention_weights[i])
        
        # Calculate average attention for each class
        avg_attention_by_class = {}
        for label, attn_weights in attention_by_class.items():
            avg_attention_by_class[label] = np.mean(np.array(attn_weights), axis=0)
        
        # Create an average attention matrix, each row represents a class
        num_classes = len(LABELS)
        # Sample time steps from 128 to 32 (step size 4)
        time_length = all_attention_weights.shape[1]  # Get the actual time step length
        sampled_timesteps = list(range(0, time_length, 4))  # Sample every 4 steps
        
        # Initialize the matrix with the correct size
        avg_attention_matrix = np.zeros((num_classes, len(sampled_timesteps)))
        
        for i in range(num_classes):
            if i in avg_attention_by_class:
                for j, timestep in enumerate(sampled_timesteps):
                    if timestep < avg_attention_by_class[i].shape[0]:
                        # If it is a multi-dimensional array, take the first element
                        value = avg_attention_by_class[i][timestep]
                        if isinstance(value, np.ndarray):
                            if value.size == 1:  # If it is a single-element array
                                avg_attention_matrix[i, j] = float(value)
                            else:  # If it is a multi-element array, take the average
                                avg_attention_matrix[i, j] = np.mean(value)
                        else:  # If it is already a scalar
                            avg_attention_matrix[i, j] = value
        
        # Plot heatmap
        plt.figure(figsize=(14, 8))
        heat_map = sns.heatmap(avg_attention_matrix, cmap='viridis', 
                       xticklabels=[f'{t}' for t in sampled_timesteps],
                       yticklabels=LABELS)
        
        # Add title and labels
        plt.title('Average Attention Weights by Activity Class (Time Step Sampled)')
        plt.xlabel('Time Steps (sampled every 4 steps)')
        plt.ylabel('Activity')
        
        # Adjust the font size of the y-axis labels
        plt.yticks(rotation=0, fontsize=10)
        
        # Save image
        plt.tight_layout()
        plt.savefig(snapshot_path + '/attention_weights_sampled.png')
        
        # Also generate a signal-level attention visualization
        # Find the most representative sample for each activity class (attention distribution closest to the average)
        representative_samples = {}
        for label in range(num_classes):
            if label in attention_by_class and len(attention_by_class[label]) > 0:
                avg_attn = avg_attention_by_class[label]
                best_sample_idx = 0
                min_dist = float('inf')
                
                for idx, attn in enumerate(attention_by_class[label]):
                    # Calculate the distance from the average attention
                    dist = np.sum((attn - avg_attn) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        best_sample_idx = idx
                
                # Create an array of the correct size
                rep_sample = np.zeros(len(sampled_timesteps))
                for j, timestep in enumerate(sampled_timesteps):
                    if timestep < attention_by_class[label][best_sample_idx].shape[0]:
                        # Handle potentially multi-dimensional arrays as well
                        value = attention_by_class[label][best_sample_idx][timestep]
                        if isinstance(value, np.ndarray):
                            if value.size == 1:
                                rep_sample[j] = float(value)
                            else:
                                rep_sample[j] = np.mean(value)
                        else:
                            rep_sample[j] = value
                
                representative_samples[label] = rep_sample
        
        # Plot heatmap of representative samples
        if representative_samples:
            plt.figure(figsize=(14, 8))
            rep_attn_matrix = np.array([representative_samples[i] if i in representative_samples else np.zeros(len(sampled_timesteps)) 
                                      for i in range(num_classes)])
            _ = sns.heatmap(rep_attn_matrix, cmap='viridis', 
                        xticklabels=[f'{t}' for t in sampled_timesteps],
                        yticklabels=LABELS)
            plt.title('Representative Sample Attention Weights by Activity Class')
            plt.xlabel('Time Steps (sampled every 4 steps)')
            plt.ylabel('Activity')
            plt.yticks(rotation=0, fontsize=10)
            plt.tight_layout()
            plt.savefig(snapshot_path + '/representative_attention_weights.png')

        print(f"Test Accuracy: {accuracy:.2f}%")
        print("\nClassification Report:")
        print(report)
        print("\nConfusion Matrix:")
        print(conf_mat)
        
        return accuracy
    except Exception as e:
        import traceback
        print(f"Error in test function: {str(e)}")
        print(traceback.format_exc())
        return 0.0

if __name__ == "__main__":
    test()
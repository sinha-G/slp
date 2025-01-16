import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import numpy as np
import gzip
import pickle
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import gc
from torch.cuda.amp import autocast, GradScaler
import pandas

class TrainingDataset(Dataset):
    """
    Custom dataset for loading game segments from compressed numpy files.
    """
    def __init__(self, df, transform=None):
        """
        Initializes the dataset.

        Args:
            file_paths (list of str): List of paths to the numpy files.
            labels (list): List of labels corresponding to each file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.file_paths = df['player_inputs_np_sub_path'].to_numpy()
        self.encoded_labels = df['encoded_labels'].to_numpy()
        self.segment_start_index = df['segment_start_index'].to_numpy()
        # self.segment_index = df['segment_index'].to_numpy()
        self.segment_length = df['segment_length'].to_numpy()
        self.transform = transform

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.file_paths)
    

    def __getitem__(self, idx):
        """Loads and returns a sample from the dataset at the specified index."""
        with gzip.open('/workspace/melee_project_data/input_np/' + self.file_paths[idx].replace('\\','/'), 'rb') as f:
            segment = np.load(f)

        if self.transform:
            segment = self.transform(segment)
        
        # Start and end of the segment
        segment_start = self.segment_start_index[idx]
        segment_end = self.segment_start_index[idx] + self.segment_length[idx]
        
        # Convert to PyTorch tensors
        segment_tensor = torch.from_numpy(segment[:,segment_start:segment_end]).float()
        label_tensor = torch.tensor(self.encoded_labels[idx], dtype=torch.long)
        return segment_tensor, label_tensor
    
def prepare_data_loaders_no_val(train_df, test_df, batch_size, num_workers):
    """
    Prepares training, validation, and test data loaders.

    Args:
        file_paths (np.array): Array of file paths to the data files.
        labels (np.array): Array of labels corresponding to the data files.
        batch_size (int): Batch size for the data loaders.
        num_workers (int): Number of worker processes to use for data loading.

    Returns:
        dict: A dictionary containing 'train', 'val', and 'test' DataLoaders.
    """

    # Initialize datasets
    train_dataset = TrainingDataset(train_df)
    # val_dataset = TrainingDataset(file_paths_val, labels_val)
    test_dataset = TrainingDataset(test_df)

    # Initialize data loaders
    loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True,persistent_workers=True),
        'test': DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True,persistent_workers=True),
        # 'val': DataLoader(val_dataset, batch_size=2**9, num_workers=num_workers, shuffle=False, pin_memory=True,persistent_workers=True)
    }
    return loaders

def prepare_data_loaders_val(train_df, test_df, val_df, batch_size, num_workers):
    """
    Prepares training, validation, and test data loaders.

    Args:
        file_paths (np.array): Array of file paths to the data files.
        labels (np.array): Array of labels corresponding to the data files.
        batch_size (int): Batch size for the data loaders.
        num_workers (int): Number of worker processes to use for data loading.

    Returns:
        dict: A dictionary containing 'train', 'val', and 'test' DataLoaders.
    """

    # Initialize datasets
    train_dataset = TrainingDataset(train_df)
    val_dataset = TrainingDataset(val_df)
    test_dataset = TrainingDataset(test_df)

    # Initialize data loaders
    loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True,persistent_workers=True),
        'test': DataLoader(test_dataset, batch_size=2**9, num_workers=num_workers, shuffle=False, pin_memory=True,persistent_workers=True),
        'val': DataLoader(val_dataset, batch_size=2**9, num_workers=num_workers, shuffle=False, pin_memory=True,persistent_workers=True)
    }
    return loaders

def train_model(model, criterion, optimizer, loaders, device, num_epochs=1):
    scaler = GradScaler()  # Initialize the gradient scaler

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total = 0
        train_loader_tqdm = tqdm(loaders['train'], desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')
        
        for inputs, labels in train_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Resets the optimizer
            optimizer.zero_grad()
            
            # Runs the forward pass with autocasting.
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)  # Compute loss

            
            # Scales loss and calls backward() to create scaled gradients
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            
            # Updates the scale for next iteration.
            scaler.update()

            # Update progress
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loader_tqdm.set_postfix(loss=(train_loss / total), accuracy=(100.0 * train_correct / total))

            
def evaluate_model(model, test_loader, device):
    """
    Evaluates the model on the test dataset.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): The device to evaluate on.
    """
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    print(f'Test Accuracy: {100.0 * test_correct / test_total}%')
    

def predict(model, test_loader, device):
    """
    Evaluates the model on the test dataset and returns predictions as a numpy array,
    displaying a progress bar during the evaluation.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): The device to evaluate on.

    Returns:
        numpy.ndarray: Numpy array of predictions.
    """
    model.eval()
    predictions = []
    # Initialize tqdm progress bar
    pbar = tqdm(total=len(test_loader), desc="Evaluating", leave=False)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())  # Collecting predictions

            # Update the progress bar
            pbar.update(1)

    pbar.close()  # Ensure the progress bar is closed after the loop
    return predictions

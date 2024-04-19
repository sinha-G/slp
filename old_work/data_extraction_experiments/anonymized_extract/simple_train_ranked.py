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

class GameSegmentDataset(Dataset):
    """
    Custom dataset for loading game segments from compressed numpy files.
    """
    def __init__(self, file_paths, labels, transform=None):
        """
        Initializes the dataset.

        Args:
            file_paths (list of str): List of paths to the numpy files.
            labels (list): List of labels corresponding to each file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """Loads and returns a sample from the dataset at the specified index."""
        with gzip.open(self.file_paths[idx], 'rb') as f:
            segment = np.load(f)

        if self.transform:
            segment = self.transform(segment)
        
        # Convert to PyTorch tensors
        segment_tensor = torch.from_numpy(segment).float()
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return segment_tensor, label_tensor

class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network model for processing sequential data.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.LazyConv1d(32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.fc1 = nn.LazyLinear(128)  # LazyLinear allows deferring the determination of in_features
        self.fc2 = nn.Linear(128,17)  # Assuming 5 classes for classification

    def forward(self, x):
        """Defines the forward pass of the model."""
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)  # Flatten for the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_data(save_path):
    """
    Loads file paths and labels from the specified save path.

    Args:
        save_path (str): Directory where the file_paths.pkl and label_list.pkl are stored.

    Returns:
        tuple: A tuple containing arrays of file paths and labels.
    """
    with gzip.open("C:\\Users\\jaspa\\Grant ML\\slp\\data\\ranked_file_paths.npy.gz", 'rb') as f:
        file_paths = np.load(f)

    with gzip.open("C:\\Users\\jaspa\\Grant ML\\slp\\data\\ranked_label_list.npy.gz", 'rb') as f:
        labels  = np.load(f)

    return file_paths, labels

def prepare_data_loaders(file_paths, labels, batch_size=64, num_workers=15):
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
    # Split the dataset into training, validation, and test sets
    file_paths_train, file_paths_temp, labels_train, labels_temp = train_test_split(
        file_paths, labels, test_size=0.3, stratify=labels)
    file_paths_val, file_paths_test, labels_val, labels_test = train_test_split(
        file_paths_temp, labels_temp, test_size=0.5, stratify=labels_temp)

    # Initialize datasets
    train_dataset = GameSegmentDataset(file_paths_train, labels_train)
    val_dataset = GameSegmentDataset(file_paths_val, labels_val)
    test_dataset = GameSegmentDataset(file_paths_test, labels_test)

    # Initialize data loaders
    loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True,persistent_workers=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True,persistent_workers=True),
        'test': DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True,persistent_workers=True)
    }
    return loaders

def train_model(model, criterion, optimizer, loaders, device, num_epochs=10):
    """
    Trains the model.

    Args:
        model (nn.Module): The neural network model to train.
        criterion (callable): The loss function.
        optimizer (Optimizer): The optimizer for updating model parameters.
        loaders (dict): Dictionary containing 'train' and 'test' DataLoaders.
        device (torch.device): The device to train on.
        num_epochs (int): Number of epochs to train for.
    """
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total = 0
        train_loader_tqdm = tqdm(loaders['train'], desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')
        
        for inputs, labels in train_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
                    
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loader_tqdm.set_postfix(loss=(train_loss / total), accuracy=(100.0 * train_correct / total))

        # Evaluate on the test set
        evaluate_model(model, loaders['test'], device)

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

def main():
    # Example usage
    save_path = 'C:/Users/jaspa/Grant ML/slp/data'
    file_paths, labels = load_data(save_path)
    loaders = prepare_data_loaders(file_paths, labels,batch_size=64,num_workers=16)

    model = SimpleCNN().to('cuda')  # Assuming the use of a GPU
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    train_model(model, criterion, optimizer, loaders, 'cuda')

if __name__ == '__main__':
    main()
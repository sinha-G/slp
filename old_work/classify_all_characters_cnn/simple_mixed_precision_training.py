import os
import gzip
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class GameSegmentDataset(Dataset):
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
        
        # Extract 'file_paths' + 'file' and 'labels' columns
        # file_paths = (df['path'] + '\\' + df['file']).tolist()
        # labels = df['labels'].tolist()
        # segment_shifts = df['segment_shift'].tolist()
        # segment_indices = df['segment_index'].tolist()
    
        self.file_paths = (df['path'] + '\\' + df['file']).tolist()
        self.labels = df['labels'].tolist()
        self.segment_shifts = df['segment_shift'].tolist()
        self.segment_indices = df['segment_index'].tolist()
        self.transform = transform

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """Loads and returns a sample from the dataset at the specified index."""
        with gzip.open(self.file_paths[idx], 'rb') as f:
            game = np.load(f)

        # segment = np.zeros([9, 1024])
        segment_shift = self.segment_shifts[idx]
        segment_index = self.segment_indices[idx]

        segment = game[:, segment_shift * segment_index: segment_shift * segment_index + 1024]
        
        if self.transform:
            segment = self.transform(segment)
        
        # Convert to PyTorch tensors
        segment_tensor = torch.from_numpy(segment.copy()).float()
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return segment_tensor, label_tensor

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(9, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.fc1 = nn.LazyLinear(128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
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
    # Load the feather file
    test_df = pd.read_feather('C:/Users/jaspa/Grant ML/slp/data/sample_test_df.feather')
    train_df = pd.read_feather('C:/Users/jaspa/Grant ML/slp/data/sample_train_df.feather')
    val_df = pd.read_feather('C:/Users/jaspa/Grant ML/slp/data/sample_val_df.feather')
    

    # Specify what characters to use in classification here
    characters_to_keep = [
                'FOX', 
                # 'FALCO', 
                'MARTH', 
                'SHEIK', 
                # 'CAPTAIN_FALCON', 
                # 'PEACH', 
                # 'JIGGLYPUFF', 
                # 'SAMUS', 
                # 'ICE_CLIMBERS', 
                # 'GANONDORF', 
                # 'YOSHI', 
                # 'LUIGI', 
                # 'PIKACHU', 
                # 'DR_MARIO', 
                # 'NESS', 
                # 'LINK', 
                # 'MEWTWO', 
                # 'GAME_AND_WATCH', 
                # 'DONKEY_KONG', 
                # 'YOUNG_LINK', 
                # 'MARIO', 
                # 'ROY', 
                # 'BOWSER', 
                # 'ZELDA', 
                # 'KIRBY', 
                # 'PICHU'
                ]
    
    test_df = test_df[test_df['character'].isin(characters_to_keep)]
    train_df = train_df[train_df['character'].isin(characters_to_keep)]
    val_df = val_df[val_df['character'].isin(characters_to_keep)]

    # Extract 'file_paths' + 'file' and 'labels' columns
    # file_paths = (df['path'] + '\\' + df['file']).tolist()
    # labels = df['labels'].tolist()
    # segment_shifts = df['segment_shift'].tolist()
    # segment_indices = df['segment_index'].tolist()

    # Reduce the labels to be [0, 1, ..., len(characters_to_keep) - 1]
    label_encoder = LabelEncoder()
    label_encoder.fit(test_df['labels'])
    test_df['labels'] = label_encoder.transform(test_df['labels'])
    train_df['labels'] = label_encoder.transform(train_df['labels'])
    val_df['labels'] = label_encoder.transform(val_df['labels'])

    # logging.info(df)
    # logging.info(labels)

    return train_df, val_df, test_df

def prepare_data_loaders(train_df, val_df, test_df, batch_size, num_workers = 15):
    """
    Prepares training, validation, and test data loaders.

    Args:
        file_paths (np.array): Array of file paths to the data files.
        labels (np.array): Array of labels corresponding to the data files.
        segment_shifts (np.array): Array of segment shifts corresponding to the data files.
        segment_indices (np.array): Array of segment indices corresponding to the data files.
        batch_size (int): Batch size for the data loaders.
        num_workers (int): Number of worker processes to use for data loading.

    Returns:
        dict: A dictionary containing 'train', 'val', and 'test' DataLoaders.
    """
    # Split the dataset into training, validation, and test sets
    # file_paths_train, file_paths_temp, labels_train, labels_temp, segment_shifts_train, segment_shifts_temp, segment_indices_train, segment_indices_temp = train_test_split(
    #     file_paths, labels, segment_shifts, segment_indices, test_size=0.3, stratify=labels)
    # file_paths_val, file_paths_test, labels_val, labels_test, segment_shifts_val, segment_shifts_test, segment_indices_val, segment_indices_test = train_test_split(
    #     file_paths_temp, labels_temp, segment_shifts_temp, segment_indices_temp, test_size=0.5, stratify=labels_temp)

    # Initialize datasets
    train_dataset = GameSegmentDataset(train_df)
    val_dataset = GameSegmentDataset(val_df)
    test_dataset = GameSegmentDataset(test_df)

    # Initialize data loaders
    loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True, persistent_workers=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True, persistent_workers=True),
        'test': DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True, persistent_workers=True)
    }
    return loaders
    # return train_loader, val_loader, test_loader

def train_and_evaluate(model, train_loader, test_loader, device, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scaler = GradScaler()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss, total_correct, total = 0, 0, 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # outputs = model(inputs)
            # loss = criterion(outputs, labels) 
            # # Backward pass and optimization
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=(total_loss / total), accuracy=(100.0 * total_correct / total))
        
        evaluate(model, test_loader, device)

def evaluate(model, test_loader, device):
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    print(f'\nTest Accuracy: {test_correct / test_total:.2f}%')

def main():
    save_path = 'C:/Users/jaspa/Grant ML/slp/data/'
    train_df, val_df, test_df = load_data(save_path)
    batch_size, num_workers = 64, 15
    loaders = prepare_data_loaders(train_df, val_df, test_df, batch_size, num_workers)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    num_epochs = 10
    train_and_evaluate(model, loaders['train'], loaders['test'], device, num_epochs)

if __name__ == '__main__':
    main()

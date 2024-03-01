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

class GameSegmentDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with gzip.open(self.file_paths[idx], 'rb') as f:
            segment = np.load(f)
        if self.transform:
            segment = self.transform(segment)
        return torch.from_numpy(segment).float(), torch.tensor(self.labels[idx], dtype=torch.long)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(9, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.fc1 = nn.LazyLinear(128)
        self.fc2 = nn.Linear(128, 5)

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
    with open(os.path.join(save_path, 'file_paths.pkl'), 'rb') as f:
        file_paths = np.array(pickle.load(f))
    with open(os.path.join(save_path, 'label_list.pkl'), 'rb') as f:
        labels = np.array(pickle.load(f))
    return file_paths, labels

def prepare_data_loaders(file_paths, labels, batch_size, num_workers):
    file_paths_train, file_paths_temp, labels_train, labels_temp = train_test_split(file_paths, labels, test_size=0.3, stratify=labels)
    file_paths_val, file_paths_test, labels_val, labels_test = train_test_split(file_paths_temp, labels_temp, test_size=0.5, stratify=labels_temp)
    train_dataset = GameSegmentDataset(file_paths_train, labels_train)
    val_dataset = GameSegmentDataset(file_paths_val, labels_val)
    test_dataset = GameSegmentDataset(file_paths_test, labels_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True, persistent_workers=True)
    return train_loader, val_loader, test_loader

def train_and_evaluate(model, train_loader, test_loader, device, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss, total_correct, total = 0, 0, 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
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
    file_paths, labels = load_data(save_path)
    batch_size, num_workers = 64, 15
    train_loader, val_loader, test_loader = prepare_data_loaders(file_paths, labels, batch_size, num_workers)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    num_epochs = 4
    train_and_evaluate(model, train_loader, test_loader, device, num_epochs)

if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import torch.nn.functional as F




import torch
from torch.utils.data import Dataset
import numpy as np
import gzip

class GameSegmentDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        """
        file_paths: List of paths to the numpy files
        labels: List of labels corresponding to each file
        transform: Optional transform to be applied on a sample
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        
        # segment = np.load(self.file_paths[idx])
        with gzip.open(self.file_paths[idx], 'rb') as f:
            segment = np.load(f)

        if self.transform:
            segment = self.transform(segment)
            
        # return segment
        return torch.from_numpy(segment).float(), torch.tensor(self.labels[idx], dtype=torch.long)
            
    # def __getitem__(self, idx):
    #     try:
    #         with gzip.open(self.file_paths[idx], 'rb') as f:
    #             segment = np.load(f)
    #         # Apply transform if any
    #         if self.transform:
    #             segment = self.transform(segment)
    #         return torch.from_numpy(segment).float(), self.labels[idx]
    #     except Exception as e:
    #         print(f"Error loading gzip file at index {idx}: {e}")
    #         raise
def main():
    
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv1d(9, 32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool1d(2, 2)
            # Assuming the sequential data is being halved by the pooling layers
            # Update the following line if the number of features after pooling changes
            self.fc1 = nn.LazyLinear(128)  # Adjust the dimensions based on your input size
            self.fc2 = nn.Linear(128, 5)  # We have 5 classes

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = torch.flatten(x, 1)  # Flatten the tensor for the fully connected layer
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    
    from sklearn.model_selection import train_test_split
    import pickle
    import os

    # Path where the files were saved
    save_path = 'C:/Users/jaspa/Grant ML/slp/data/'

    # Load file_paths
    with open(os.path.join(save_path, 'file_paths.pkl'), 'rb') as f:
        file_paths = np.array(pickle.load(f))

    # print(file_paths)

    # Load label_list
    with open(os.path.join(save_path, 'label_list.pkl'), 'rb') as f:
        labels = np.array(pickle.load(f))

    print(labels)
        
    # Assuming file_paths and labels are lists containing your data and labels
    file_paths_train, file_paths_temp, labels_train, labels_temp = train_test_split(file_paths, labels, test_size=0.3, stratify=labels)
    file_paths_val, file_paths_test, labels_val, labels_test = train_test_split(file_paths_temp, labels_temp, test_size=0.5, stratify=labels_temp)



    from torch.utils.data import DataLoader

    # Initialize the datasets
    train_dataset = GameSegmentDataset(file_paths_train, labels_train)
    val_dataset = GameSegmentDataset(file_paths_val, labels_val)
    test_dataset = GameSegmentDataset(file_paths_test, labels_test)

    # Initialize the data loaders
    batch_size = 2**9 # Adjust based on your system's capabilities
    batch_size = 64  # Adjust based on your system's capabilities
    num_workers = 15
    # print('-------------', num_workers, '-------------------')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True, persistent_workers=True)#, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True, persistent_workers=True)#, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True, persistent_workers=True)#, pin_memory=True)


    from torch.optim import Adam
    from tqdm import tqdm


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize the model, loss function, and optimizer
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # Training loop with progress bar
    num_epochs = 4
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total = 0
        train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')
        
        # print('1')
        for inputs, labels in train_loader_tqdm:
            # print('2')
            # Move data to appropriate device (e.g., GPU if available)
            # print(inputs.shape)
            # inputs = [torch.from_numpy(segment).float() for segment in inputs]
            # inputs = torch.stack(inputs)  # If necessary, depends on your data shape
            inputs, labels = inputs.to(device), labels.to(device)
                    
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loader_tqdm.set_postfix(loss=(train_loss / total), accuracy=(100.0 * train_correct / total))

        # Evaluate on the test set
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

        test_accuracy = 100 * test_correct / test_total
        print(f'\nTest Accuracy: {test_accuracy:.2f}%')





if __name__ == '__main__':
    main()
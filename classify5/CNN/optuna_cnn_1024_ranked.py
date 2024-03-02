import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from sklearn.model_selection import train_test_split
import numpy as np
import optuna
import gzip
import pickle
from tqdm import tqdm
import pymysql
from prettytable import PrettyTable

from datetime import datetime
import os
import gc

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
    
class CustomNet(nn.Module):
    def __init__(self, trial):
        super(CustomNet, self).__init__()

        # Fixed dropout rate (not tuned by Optuna)
        dropout_rate = 0.35

        # Convolutional layers setup
        self.conv_layers = nn.ModuleList()
        self.poolings = []
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        num_layers = trial.suggest_int(f"num_conv_layers", 3, 7)
        in_channels = 9  # Fixed input channel size

        
        
        ######################################################################################################
        # In length is 2 ** 10
        # Padding is set up so that the out length is always reduced by 1 / 2 ** out_length_reduction_exponent
        # The length of a kernel is: kernel + (dilation - 1) * (kernel_size - 1)
        # The max lenght of a kernel is 25 which is kernel_size = 7 and dilation = 4
        # The in lenght can never be less than 25
        # Since the in lenght is always a power of 2, the in lenght can be no less than 2 ** 5 = 32,
        # we need to make sure not to reduce the in lenght too much, we keep track of
        # how much we can still reduce the length by using length_reduction_power_left which is set to 5.
        ######################################################################################################
        length_reduction_exporent_remaining = 5
        in_length_exponent = 10
        for i in range(num_layers):  # Convolutional layers
            ###########################
            # In length is a power of 2
            ###########################
            if i == 0: 
                out_channels = trial.suggest_int(f"conv_{i}_out_channels", 9, 9 * 48, step = 9)
                groups = 9
            elif i == -1:
                out_channels = trial.suggest_int(f"conv_{i}_out_channels", 1, 256)
                groups = 1
            else:
                out_channels = trial.suggest_int(f"conv_{i}_out_channels", 1, 512)
                groups = 1
            # kernel_size = trial.suggest_int(f"conv_{i}_kernel_size", 3, 7, step=2)
            k = trial.suggest_int(f"conv_{i}_kernel_size_power", 1, 5)  # can safely change 5 to be anything
            kernel_size = 2 * k + 1
            dilation = trial.suggest_int(f"conv_{i}_dilation", 1, 4)
            out_length_reduction_exponent = trial.suggest_int(f"conv_{i}_out_length_reduction_exponent", 0, min(2, length_reduction_exporent_remaining))
            # conv_stride_length_exponent = trial.suggest_int(f"conv_{i}_stride_length_exponent", 0, out_length_reduction_exponent)
            conv_stride_length_exponent = out_length_reduction_exponent
            # Keep track of how much reducing we still can do
            length_reduction_exporent_remaining -= out_length_reduction_exponent
            in_length_exponent -= out_length_reduction_exponent
            # Set stride
            stride = 2 ** conv_stride_length_exponent
            # Padding is chosen so that out length is a power of 2
            # there is a floor in the formula. If we want to use more than 2 for out_length_reduction_exponent, we neen do caluclate the cases
            if (conv_stride_length_exponent == 2) and (((dilation * k) % 2) == 1):
                padding = dilation * k - 1
            else:
                padding = dilation * k
                
            self.conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size,stride, padding, dilation, groups))
            in_channels = out_channels  # Update in_channels for the next layer

            if conv_stride_length_exponent < out_length_reduction_exponent:
                pooling_type = trial.suggest_int(f"layer_{i}_pooling_type", 0, 1)    # 1: max, 0: avg
                pool_kernal_size_exponent = out_length_reduction_exponent - conv_stride_length_exponent
                if pooling_type == 1:
                    self.poolings.append(nn.MaxPool1d(2 ** pool_kernal_size_exponent))
                else:
                    self.poolings.append(nn.AvgPool1d(2 ** pool_kernal_size_exponent))
            else:
                self.poolings.append(None)    #   No pooling in current layer

            
            # Batch Normalization
            self.bns.append(nn.BatchNorm1d(in_channels))
            
        # Max pooling layer
        # The kernel can be a power of two, up to the in lenght
        # In length of the output will be 2 ** out_length_exponent
        # and lenght can be 1, 2, 4, 8, 16, 32
        
        kernel_exponent = trial.suggest_int(f"maxpool_kernel_exponent",length_reduction_exporent_remaining , in_length_exponent)
        kernel_size = 2 ** kernel_exponent
        in_length_exponent -= kernel_exponent
        
        self.pool1 = nn.MaxPool1d(kernel_size=kernel_size)
        
        
        # The length right now should be 2 ** in_length_exponent, so we can be exact in our first lineal layer
        self.fc1 = nn.Linear(out_channels * 2 ** in_length_exponent, trial.suggest_int("fc1_out_features", 32, 256))
        # self.fc1 = nn.LazyLinear(trial.suggest_int("fc1_out_features", 64, 256))
        self.fc1_dropout = nn.Dropout(dropout_rate)  # Dropout after fc1
        self.fc2 = nn.Linear(self.fc1.out_features, trial.suggest_int("fc2_out_features", 32, 128))
        self.fc2_dropout = nn.Dropout(dropout_rate)  # Dropout after fc2
        self.fc3 = nn.Linear(self.fc2.out_features, 5)  # Output layer with 1 unit for binary classification

    def forward(self, x):
        # Apply convolutional layers with optional ReLU and fixed dropout
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            x = self.bns[i](x)
            if self.poolings[i]:
                x = self.poolings[i](x)
            x = F.relu(x)            

        # Optional max pooling after conv layers
        # if self.use_pool1:
        x = self.pool1(x)

        # Flatten for fully connected layers
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc1_dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_dropout(x)
        x = self.fc3(x)  # Output without activation for BCEWithLogitsLoss
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
        'train': DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True, persistent_workers=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True, persistent_workers=True),
        'test': DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True, persistent_workers=True)
    }
    return loaders

def train_model(model, criterion, optimizer, loaders, device, num_epochs=4):
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
            inputs, labels = inputs.to(device), labels.to(device).float()
                    
            # Forward pass
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress
            train_loss += loss.item()
            predicted = torch.where(outputs >= 0, torch.tensor(1), torch.tensor(0))
            total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loader_tqdm.set_postfix(loss=(train_loss / total), accuracy=(100.0 * train_correct / total))

        # Evaluate on the test set
        # evaluate_model(model, loaders['test'], device)

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
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs).squeeze()
            predicted = torch.where(outputs >= 0.5, torch.tensor(1), torch.tensor(0))
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    print(f'Test Accuracy: {100.0 * test_correct / test_total}%')

def objective(trial, dataloaders):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model with hyperparameters suggested by Optuna
    model = CustomNet(trial).to(device)

    # Define the optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
    # criterion = nn.BCEWithLogitsLoss(reduction = 'sum')
    criterion = nn.CrossEntropyLoss(reduction = 'sum')
    def train_epoch(model, dataloader, optimizer, criterion):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).long()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress
            train_loss += loss.item()
            _, predicted = torch.max(outputs[:, :5], dim=1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_accuracy = 100 * train_correct / train_total
        train_loss = train_loss / train_total

        return train_loss, train_accuracy

    def validate_epoch(model, dataloader, criterion):
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device).long()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Update progress
                val_loss += loss.item()
                _, predicted = torch.max(outputs[:, :5], dim=1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        val_loss = val_loss / val_total

        return val_loss, val_accuracy

    def evaluate_test(model, dataloader, criterion):
        model.eval()  # Set model to evaluation mode
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():  # No gradients needed
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device).long()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Update progress
                test_loss += loss.item()
                _, predicted = torch.max(outputs[:, :5], dim=1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
        test_accuracy = 100 * test_correct / test_total
        test_loss = test_loss / test_total

        return test_loss, test_accuracy
                
    # Training loop with early stopping and tqdm progress bar
    patience = 3
    epochs = 20
    min_delta = 0.0001
    min_overfit = .001

    best_val_loss = float('inf')
    best_val_acc = 0.0
    epochs_no_improve = 0
    epochs_overfit = 0

    # Initialize tqdm progress bar
    pbar = tqdm(total=epochs, desc="Epochs", position=0, leave=True)

    for epoch in range(epochs):
        train_loss, train_accuracy = train_epoch(model, dataloaders['train'], optimizer, criterion)
        val_loss, val_accuracy = validate_epoch(model, dataloaders['val'], criterion)
        
        # Early Stopping check and progress bar update
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
        if (val_loss + min_delta) < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if val_loss - train_loss < min_overfit:
            epochs_overfit = 0
        else:
            epochs_overfit += 1

        # Update progress bar
        pbar.set_postfix_str(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Best Validation Accuracy: {best_val_accuracy:.4f}")
        pbar.update(1)  # Move the progress bar by one epoch

        # Check early stopping condition
        if epochs_no_improve >= patience or epochs_overfit >= patience:
            pbar.write(f'Early stopping triggered at epoch {epoch + 1}')
            pbar.close()  # Close the progress bar
            break

    # Evaluate model on test set after training is complete (if necessary)
    test_loss, test_accuracy = evaluate_test(model, dataloaders['test'], criterion)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    pbar.close()  # Ensure the progress bar is closed
    gc.collect()
    torch.cuda.empty_cache()
    return best_val_loss


def main():
    # Example usage
    save_path = 'C:/Users/jaspa/Grant ML/slp/data'
    file_paths, labels = load_data(save_path)
    loaders = prepare_data_loaders(file_paths, labels)

    current_datetime = datetime.now()
    current_datetime_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S ")

    study = optuna.create_study(study_name = current_datetime_string + "Basic CNN",
                                direction="minimize",
                                storage = "mysql+pymysql://root:MomentusPigs@localhost:3306/optuna_trials"
                                )

    objective_with_loaders = lambda trial: objective(trial, loaders)

    study.optimize(objective_with_loaders, n_trials = 1000, show_progress_bar = True, timeout=3600 * 6)

    # Print the overall best hyperparameters
    print("Best trial overall:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # model = SimpleCNN().to('cuda')  # Assuming the use of a GPU
    # criterion = nn.BCEWithLogitsLoss()
    # optimizer = Adam(model.parameters(), lr=0.001)

    # train_model(model, criterion, optimizer, loaders, 'cuda')
    # evaluate_model(model, loaders['val'])

if __name__ == '__main__':
    main()

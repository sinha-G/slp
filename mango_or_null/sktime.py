import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchsummary import summary
import sktime as skt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
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
import logging

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
            segment = np.load(f)[:1024, :]
        # logging.info(f'Data shape: {segment.shape}')

        if self.transform:
            segment = self.transform(segment)
        
        # Convert to PyTorch tensors
        segment_tensor = torch.from_numpy(segment).float()
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return segment_tensor, label_tensor
    
def collate_fn(batch):
    # Each sample in the batch has shape [9, x]
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Get the maximum sequence length in the batch
    max_length = max(sample.shape[1] for sample in inputs)

    # Pad each input tensor to the maximum length
    padded_inputs = []
    for sample in inputs:
        pad_length = max_length - sample.shape[1]
        padded_sample = F.pad(sample, (0, pad_length), mode='constant', value=0)
        padded_inputs.append(padded_sample)

    # Stack the padded input tensors into a single tensor
    padded_inputs = torch.stack(padded_inputs, dim=0)

    # Stack the labels into a single tensor
    labels = torch.stack(labels, dim=0)

    return padded_inputs, labels

def load_data(save_path):
    """
    Loads file paths and labels from the specified save path.

    Args:
        save_path (str): Directory where the file_paths.pkl and label_list.pkl are stored.

    Returns:
        tuple: A tuple containing arrays of file paths and labels.
    """
    with gzip.open("C:\\Users\\jaspa\\Grant ML\\slp\\data\\mango_or_null_X.npy.gz", 'rb') as f:
        file_paths = np.load(f)

    with gzip.open("C:\\Users\\jaspa\\Grant ML\\slp\\data\\mango_or_null_y.npy.gz", 'rb') as f:
        labels  = np.load(f)

    return file_paths, labels

def prepare_data_loaders(file_paths, labels, batch_size, num_workers=15):
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
        'train': DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True, persistent_workers=True, collate_fn=collate_fn),
        'val': DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True, persistent_workers=True, collate_fn=collate_fn),
        'test': DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True, persistent_workers=True, collate_fn=collate_fn)
    }
    return loaders

def train_epoch(model, dataloader, optimizer, criterion):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)  # Apply .float() to labels

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update progress
        train_loss += loss.item()
        train_total += labels.size(0)

        # Apply softmax to get the predicted probabilities for each class
        predicted_probs = torch.softmax(outputs, dim=1)

        # Get the predicted class indices by finding the index with the maximum probability
        predicted_classes = torch.argmax(predicted_probs, dim=1)

        # Add 1 for every correct prediction
        train_correct += (predicted_classes == labels).sum().item()

    train_accuracy = 100 * train_correct / train_total
    train_loss = train_loss / train_total

    return train_loss, train_accuracy

def validate_epoch(model, dataloader, criterion):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Update progress
            val_loss += loss.item()
            val_total += labels.size(0)

            # Apply softmax to get the predicted probabilities for each class
            predicted_probs = torch.softmax(outputs, dim=1)

            # Get the predicted class indices by finding the index with the maximum probability
            predicted_classes = torch.argmax(predicted_probs, dim=1)

            # Add 1 for every correct prediction
            val_correct += (predicted_classes == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        val_loss = val_loss / val_total

        return val_loss, val_accuracy
    
def evaluate_test(model, dataloader, criterion, study_name):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()  # Set model to evaluation mode
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    y_true = []
    y_pred = []
        
    confusion_matrix_path = 'data\\mango_or_null\\confusion_matrices\\' + study_name + '\\'
    os.makedirs(confusion_matrix_path, exist_ok=True)
    with torch.no_grad():  # No gradients needed
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Update progress
            test_loss += loss.item()
            test_total += labels.size(0)

            # Apply softmax to get the predicted probabilities for each class
            predicted_probs = torch.softmax(outputs, dim=1)

            # Get the predicted class indices by finding the index with the maximum probability
            predicted_classes = torch.argmax(predicted_probs, dim=1)

            # Add 1 for every correct prediction
            test_correct += (predicted_classes == labels).sum().item()

            # Collect true and predicted labels for confusion matrix
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted_classes.cpu().numpy())

        test_accuracy = 100 * test_correct / test_total
        test_loss = test_loss / test_total

        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred, normalize='pred')

        current_datetime = datetime.now()
        current_datetime_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S ")
        current_datetime_string = current_datetime_string.replace(":", "-")

        # Plot confusion matrix
        opponents = ['MANGO',
                     'NULL',   
                    ]
        plt.figure(figsize=(1.5 * len(opponents), 1.5 * len(opponents)))
        sns.heatmap(cm, annot=True, fmt='f', cmap='Blues', xticklabels=opponents, yticklabels=opponents)
        plt.gca().invert_yaxis()
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(confusion_matrix_path + current_datetime_string + ' Confusion Matrix (pred).png')

        return test_loss, test_accuracy


def train_model(model, epochs, study_name, dataloaders, optimizer, criterion):
    # Training loop with early stopping and tqdm progress bar
    patience = 5
    min_overfit = 0.005

    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    epochs_no_improve = 0
    epochs_overfit = 0

    # Initialize tqdm progress bar
    pbar = tqdm(total=epochs, desc="Epochs", position=0, leave=True)

    for epoch in range(epochs):
        train_loss, train_accuracy = train_epoch(model, dataloaders['train'], optimizer, criterion)
        val_loss, val_accuracy = validate_epoch(model, dataloaders['val'], criterion)
        
        # Early Stopping check and progress bar update
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
        if val_loss < best_val_loss:
            best_val_loss = val_loss
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
        # Log Losses
        logging.info(f'Epoch {epoch + 1}/{epochs} - Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Best Validation Accuracy: {best_val_accuracy:.4f}')

        # Check early stopping condition
        if epochs_no_improve >= patience or epochs_overfit >= patience:
            pbar.write(f'Early stopping triggered at epoch {epoch + 1}')
            pbar.close()  # Close the progress bar
            break

    # Evaluate model on test set after training is complete (if necessary)
    test_loss, test_accuracy = evaluate_test(model, dataloaders['test'], criterion, study_name)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    logging.info(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    pbar.close()  # Ensure the progress bar is closed
    gc.collect()
    torch.cuda.empty_cache()
    return best_val_loss

def main():
     # Ensure reproducibility
    seed = 42
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    sampler = optuna.samplers.TPESampler(seed = seed)
    
    # Get starting time
    current_datetime = datetime.now()
    current_datetime_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S - ")
    current_datetime_string = current_datetime_string.replace(":", "-")
    
    # Set Some Variables
    study_name = current_datetime_string + "sktime - Classify Mang0 or Null Fox"
    batch_size = 32
    epochs = 50
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss(reduction = 'sum')

    # Set up logging file
    log_file = 'data\\mango_or_null\\logs\\' + study_name + ' Log.txt'
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f'Seed: {seed}')

    save_path = 'C:/Users/jaspa/Grant ML/slp/data'
    file_paths, labels = load_data(save_path)
    loaders = prepare_data_loaders(file_paths, labels, batch_size = batch_size)

    # skt.show_versions

    #model = CNNClassifier(n_epochs=epochs, batch_size=batch_size, optimizer=optimizer)

    train_model(model, epochs, study_name, loaders, optimizer, criterion)


if __name__ == '__main__':
    main()

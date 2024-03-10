import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchsummary import summary
from torch.cuda.amp import GradScaler, autocast

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import optuna
import gzip
import pickle
from tqdm import tqdm
import pymysql
from prettytable import PrettyTable
import pandas as pd

from datetime import datetime
import os
import gc
import logging
import time
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

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class CustomNet(nn.Module):
#     def __init__(self):
#         super(CustomNet, self).__init__()

#         self.conv_layers = nn.Sequential(
#             nn.Conv1d(9, 117, kernel_size=5, stride=1, padding=3, groups=9),
#             nn.ReLU(),
#             nn.BatchNorm1d(117),
#             nn.Dropout(p=0.4399179174743659),
#             nn.Dropout(p=0.11409509284368298),
#             nn.Conv1d(117, 196, kernel_size=9, stride=1, padding=3),
#             nn.ReLU(),
#             nn.BatchNorm1d(196),
#             nn.Dropout(p=0.4399179174743659),
#             nn.Dropout(p=0.11409509284368298),
#             nn.Conv1d(196, 196, kernel_size=7, stride=2, padding=3),
#             nn.ReLU(),
#             nn.BatchNorm1d(196),
#             nn.Dropout(p=0.4399179174743659),
#             nn.Dropout(p=0.11409509284368298),
#             nn.Conv1d(196, 196, kernel_size=5, stride=2, padding=3),
#             nn.ReLU(),
#             nn.BatchNorm1d(196),
#             nn.Dropout(p=0.4399179174743659),
#             nn.Dropout(p=0.11409509284368298),
#             nn.Conv1d(196, 196, kernel_size=9, stride=2, padding=3),
#             nn.ReLU(),
#             nn.BatchNorm1d(196),
#             nn.Dropout(p=0.4399179174743659),
#             nn.Dropout(p=0.11409509284368298),
#             nn.Conv1d(196, 196, kernel_size=9, stride=2, padding=3),
#             nn.ReLU(),
#             nn.BatchNorm1d(196),
#             nn.Dropout(p=0.4399179174743659),
#             nn.Dropout(p=0.11409509284368298),
#             nn.Conv1d(196, 106, kernel_size=9, stride=1, padding=3),
#             nn.ReLU(),
#             nn.BatchNorm1d(106),
#             nn.Dropout(p=0.4399179174743659),
#             nn.Dropout(p=0.11409509284368298)
#         )

#         self.fc_layers = nn.Sequential(
#             nn.Linear(in_features=6466, out_features=251, bias=True),
#             nn.ReLU(),
#             nn.BatchNorm1d(251),
#             nn.Dropout(p=0.37364553629019864),
#             nn.Linear(in_features=251, out_features=100, bias=True),
#             nn.ReLU(),
#             nn.BatchNorm1d(100),
#             nn.Dropout(p=0.37364553629019864),
#             nn.Linear(in_features=100, out_features=100, bias=True),
#             nn.ReLU(),
#             nn.BatchNorm1d(100),
#             nn.Dropout(p=0.37364553629019864),
#             nn.Linear(in_features=100, out_features=100, bias=True),
#             nn.ReLU(),
#             nn.BatchNorm1d(100),
#             nn.Dropout(p=0.37364553629019864),
#             nn.Linear(in_features=100, out_features=26, bias=True)
#         )

#     def forward(self, x):
#         x = self.conv_layers(x)
#         x = torch.flatten(x, 1)
#         x = self.fc_layers(x)
#         return x

# class CustomNet(nn.Module):
#     def __init__(self):
#         super(CustomNet, self).__init__()

#         self.conv_layers = nn.Sequential(
#             nn.Conv1d(9, 99, kernel_size=5, stride=1, padding=3, groups=9),
#             nn.ReLU(),
#             nn.BatchNorm1d(99),
#             nn.Dropout(p=0.1926622699555676),
#             nn.Conv1d(99, 235, kernel_size=7, stride=2, padding=3),
#             nn.ReLU(),
#             nn.BatchNorm1d(235),
#             nn.Dropout(p=0.1926622699555676),
#             nn.Conv1d(235, 235, kernel_size=9, stride=2, padding=3),
#             nn.ReLU(),
#             nn.BatchNorm1d(235),
#             nn.Dropout(p=0.1926622699555676),
#             nn.Conv1d(235, 235, kernel_size=5, stride=1, padding=3),
#             nn.ReLU(),
#             nn.BatchNorm1d(235),
#             nn.Dropout(p=0.1926622699555676),
#             nn.Conv1d(235, 235, kernel_size=7, stride=2, padding=3),
#             nn.ReLU(),
#             nn.BatchNorm1d(235),
#             nn.Dropout(p=0.1926622699555676),
#             nn.Conv1d(235, 235, kernel_size=5, stride=2, padding=3),
#             nn.ReLU(),
#             nn.BatchNorm1d(235),
#             nn.Dropout(p=0.1926622699555676),
#             nn.Conv1d(235, 100, kernel_size=3, stride=1, padding=3),
#             nn.ReLU(),
#             nn.BatchNorm1d(100),
#             nn.Dropout(p=0.1926622699555676)
#         )

#         self.fc_layers = nn.Sequential(
#             nn.Linear(in_features=7000, out_features=256),
#             nn.ReLU(),
#             nn.BatchNorm1d(256),
#             nn.Dropout(p=0.2638850409573792),
#             nn.Linear(in_features=256, out_features=61),
#             nn.ReLU(),
#             nn.BatchNorm1d(61),
#             nn.Dropout(p=0.2638850409573792),
#             nn.Linear(in_features=61, out_features=61),
#             nn.ReLU(),
#             nn.BatchNorm1d(61),
#             nn.Dropout(p=0.2638850409573792),
#             nn.Linear(in_features=61, out_features=61),
#             nn.ReLU(),
#             nn.BatchNorm1d(61),
#             nn.Dropout(p=0.2638850409573792),
#             nn.Linear(in_features=61, out_features=26)
#         )

#     def forward(self, x):
#         x = self.conv_layers(x)
#         x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
#         x = self.fc_layers(x)
#         return x
    
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()

        # Define the convolutional layers as per the structure given
        self.conv_layers = nn.Sequential(
            nn.Conv1d(9, 99, kernel_size=5, stride=1, padding=3, groups=9),
            nn.ReLU(),
            nn.BatchNorm1d(99),
            nn.Dropout(p=.4),
            nn.Dropout1d(p=0.0),  # Assuming Dropout1d is a typo and should be Dropout
            nn.Conv1d(99, 235, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(235),
            nn.Dropout(p=.4),
            nn.Dropout1d(p=0.0),
            nn.Conv1d(235, 235, kernel_size=9, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(235),
            nn.Dropout(p=.4),
            nn.Dropout1d(p=0.0),
            nn.Conv1d(235, 235, kernel_size=5, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(235),
            nn.Dropout(p=.4),
            nn.Dropout1d(p=0.0),
            nn.Conv1d(235, 235, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(235),
            nn.Dropout(p=.4),
            nn.Dropout1d(p=0.0),
            nn.Conv1d(235, 235, kernel_size=5, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(235),
            nn.Dropout(p=.4),
            nn.Dropout1d(p=0.0),
            nn.Conv1d(235, 100, kernel_size=3, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Dropout(p=.4),
            nn.Dropout1d(p=0.0)
        )

        # Define the fully connected layers as per the structure given
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=7000, out_features=256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.2638850409573792),
            nn.Linear(in_features=256, out_features=61),
            nn.ReLU(),
            nn.BatchNorm1d(61),
            nn.Dropout(p=0.2638850409573792),
            nn.Linear(in_features=61, out_features=61),
            nn.ReLU(),
            nn.BatchNorm1d(61),
            nn.Dropout(p=0.2638850409573792),
            nn.Linear(in_features=61, out_features=61),
            nn.ReLU(),
            nn.BatchNorm1d(61),
            nn.Dropout(p=0.2638850409573792),
            nn.Linear(in_features=61, out_features=26)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layers
        x = self.fc_layers(x)
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
    

    # # Specify what characters to use in classification here
    # characters_to_keep = [
    #             'FOX', 
    #             # 'FALCO', 
    #             # 'MARTH', 
    #             'SHEIK', 
    #             # 'CAPTAIN_FALCON', 
    #             # 'PEACH', 
    #             # 'JIGGLYPUFF', 
    #             # 'SAMUS', 
    #             # 'ICE_CLIMBERS', 
    #             # 'GANONDORF', 
    #             # 'YOSHI', 
    #             # 'LUIGI', 
    #             # 'PIKACHU', 
    #             # 'DR_MARIO', 
    #             # 'NESS', 
    #             # 'LINK', 
    #             # 'MEWTWO', 
    #             # 'GAME_AND_WATCH', 
    #             # 'DONKEY_KONG', 
    #             # 'YOUNG_LINK', 
    #             # 'MARIO', 
    #             # 'ROY', 
    #             # 'BOWSER', 
    #             # 'ZELDA', 
    #             # 'KIRBY', 
    #             # 'PICHU'
    #             ]
    
    # test_df = test_df[test_df['character'].isin(characters_to_keep)]
    # train_df = train_df[train_df['character'].isin(characters_to_keep)]
    # val_df = val_df[val_df['character'].isin(characters_to_keep)]

    # # Extract 'file_paths' + 'file' and 'labels' columns
    # # file_paths = (df['path'] + '\\' + df['file']).tolist()
    # # labels = df['labels'].tolist()
    # # segment_shifts = df['segment_shift'].tolist()
    # # segment_indices = df['segment_index'].tolist()

    # # Reduce the labels to be [0, 1, ..., len(characters_to_keep) - 1]
    # label_encoder = LabelEncoder()
    # label_encoder.fit(test_df['labels'])
    # test_df['labels'] = label_encoder.transform(test_df['labels'])
    # train_df['labels'] = label_encoder.transform(train_df['labels'])
    # val_df['labels'] = label_encoder.transform(val_df['labels'])

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


def train_epoch(model, dataloader, optimizer, criterion, scaler, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    start_time = time.time()

    with tqdm(dataloader, desc="Training", unit="batch") as tepoch:
        for batch_idx, (inputs, labels) in enumerate(tepoch):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward and backward passes
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            running_total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum().item()

            # Calculate running loss and accuracy
            running_accuracy = 100. * running_correct / running_total
            running_avg_loss = running_loss / running_total

            # Update tqdm postfixes to show running loss and accuracy
            tepoch.set_postfix(Loss=running_avg_loss, Accuracy=f"{running_accuracy:.2f}%")

    elapsed_time = time.time() - start_time
    train_loss = running_loss / running_total
    train_accuracy = running_accuracy
    print(f"Training time: {elapsed_time:.2f}s")

    return train_loss, train_accuracy

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    start_time = time.time()

    with torch.no_grad():
        with tqdm(dataloader, desc="Validation", unit="batch") as tepoch:
            for inputs, labels in tepoch:
                inputs, labels = inputs.to(device), labels.to(device)

                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_total += labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()
                
                running_accuracy = 100. * val_correct / val_total
                running_loss = val_loss / val_total

                # Update tqdm postfixes
                batch_loss = loss.item()
                tepoch.set_postfix(Loss=running_loss, Accuracy=f"{running_accuracy:.2f}%")

    elapsed_time = time.time() - start_time
    val_accuracy = 100 * val_correct / val_total
    val_loss /= len(dataloader.dataset)
    print(f"Validation time: {elapsed_time:.2f}s")

    return val_loss, val_accuracy


def evaluate_test(model, dataloader, criterion, device):
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    y_true = []
    y_pred = []

    start_time = time.time()

    with torch.no_grad():  # No gradients needed
        with tqdm(dataloader, desc="Testing", unit="batch") as tepoch:
            for inputs, labels in tepoch:
                inputs, labels = inputs.to(device), labels.to(device)

                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                test_loss += loss.item()
                test_total += labels.size(0)

                predicted_probs = torch.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(predicted_probs, dim=1)

                test_correct += (predicted_classes == labels).sum().item()

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted_classes.cpu().numpy())

                running_accuracy = 100. * test_correct / test_total
                running_loss = test_loss / test_total
                tepoch.set_postfix(Loss=running_loss, Accuracy=f"{running_accuracy:.2f}%")

    elapsed_time = time.time() - start_time
    test_accuracy = 100 * test_correct / test_total
    test_loss = test_loss / test_total

    print(f"Testing time: {elapsed_time:.2f}s")

    cm = confusion_matrix(y_true, y_pred, normalize='pred')
    plt.figure(figsize=(12, 12))
    sns.heatmap(cm, annot=True, fmt='f', cmap='Blues')
    plt.gca().invert_yaxis()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    return test_loss, test_accuracy
        
def main():
    seed = 42
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    # Get starting time
    current_datetime = datetime.now()
    current_datetime_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S - ")
    current_datetime_string = current_datetime_string.replace(":", "-")
    
    # Set Some Variables
    study_name = "26 char trial 29 modified dropout"
    batch_size = 256

    # Set up logging file
    log_file = 'data\\classify5\\logs\\' + study_name + ' Log.txt'
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    save_path = current_datetime_string + 'C:/Users/jaspa/Grant ML/slp/data'
    train_df, val_df, test_df= load_data(save_path)
    dataloaders = prepare_data_loaders(train_df, val_df, test_df, batch_size = batch_size,num_workers=15)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model with hyperparameters suggested by Optuna
    model = CustomNet().to(device)
 
    # Get a single batch to determine the input size
    inputs, _ = next(iter(dataloaders['train']))

    # Log model info
    summary(model, input_size=inputs.size()[1:], device='cuda' if torch.cuda.is_available() else 'cpu')
    model_summary_str = str(model)
    logging.info(f'Model Summary:\n{model_summary_str}')

    # Define the optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)


    # optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
    # criterion = nn.BCEWithLogitsLoss(reduction = 'sum')

    # class_weights = torch.tensor([1, 98880 / 93713, 98880 / 54502, 98880 / 43391, 98880 / 32625], device = device)
    # class_weights.to(device)
    criterion = nn.CrossEntropyLoss(reduction = 'sum')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)
    # Training loop with early stopping and tqdm progress bar
    patience = 10
    epochs = 100
    min_delta = 0.0001
    min_overfit = .1

    best_val_loss = float('inf')
    best_val_acc = 0.0
    epochs_no_improve = 0
    epochs_overfit = 0

    scaler = GradScaler()
    
    # Initialize tqdm progress bar
    pbar = tqdm(total=epochs, desc="Epochs", position=0, leave=True)
    try:
        for epoch in range(epochs):
            train_loss, train_accuracy = train_epoch(model, dataloaders['train'], optimizer, criterion,scaler, device)
            val_loss, val_accuracy = validate_epoch(model, dataloaders['val'], criterion, device)
            
            # Update the learning rate based on validation loss
            # Check and print if learning rate has decreased
            scheduler.step(val_loss)
            

                    # Early Stopping check and progress bar update
            # Early Stopping check and progress bar update
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # best_val_accuracy = val_accuracy
            else:
                epochs_no_improve +=1

            if (val_loss - train_loss) / train_loss > min_overfit:
                epochs_overfit += 1
                
            #  # Early Stopping and Learning Rate Adjustment
            # if  epochs_overfit >= patience:
            #     pbar.write(f'Early stopping triggered at epoch {epoch + 1}')
            #     pbar.close()
            #     break
            # if epochs_no_improve == 3:
            #     # Manually decrease learning rate
            #     epochs_no_improve = 0
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = param_group['lr'] * 0.1
            #     # logging.info(f'Learning rate decreased to {optimizer.param_groups[0]["lr"]}')
                
                


            # Update progress bar
            pbar.set_postfix_str(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.6f}, Best Validation Accuracy: {best_val_acc:.4f}, LR: {optimizer.param_groups[0]['lr']:.8f}")
            pbar.update(1)  # Move the progress bar by one epoch
            # Log Losses
            logging.info(f'Epoch {epoch + 1}/{epochs} - Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Best Validation Accuracy: {best_val_acc:.4f}')

            # Check early stopping condition
            if epochs_overfit >= patience or epochs_no_improve >= patience:
                pbar.write(f'Early stopping triggered at epoch {epoch + 1}')
                pbar.close()  # Close the progress bar
                break

    except KeyboardInterrupt:
        print("Training interrupted by user.")
        pbar.close()
    print('+')
    # Evaluate model on test set after training is complete (if necessary)
    test_loss, test_accuracy = evaluate_test(model, dataloaders['test'], criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    logging.info(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    
        # Save the final model after training
    final_model_path = 'C:/Users/jaspa/Grant ML/Models/26_modelS_2523.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f'Final model saved at {final_model_path}')

if __name__ == '__main__':
    main()
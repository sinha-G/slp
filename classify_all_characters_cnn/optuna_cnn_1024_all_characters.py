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


from datetime import datetime, timedelta
import os
import gc
import logging

class CustomNet(nn.Module):
    def __init__(self, trial,num_classes):
        super(CustomNet, self).__init__()
        self.trial = trial
        # Fixed dropout rate (not tuned by Optuna)
        # num_conv_layers = trial.suggest_int("num_conv_layers", 5, 7)
        num_conv_layers = 7
        # num_fc_layers = trial.suggest_int("num_conv_layers", 3, 7)
        num_fc_layers = 5
        
        dropout_rate_conv = trial.suggest_float("dropout_rate_conv", 0.1, 0.5)
        dropout1d_rate_conv = trial.suggest_float("dropout1d_rate_conv", 0.1, 0.5)
        dropout_rate_fc = trial.suggest_float("dropout_rate_fc", 0.2, 0.5)
        # dropout_rate_conv /= 2
        # dropout1d_rate_conv /= 2
        # dropout_rate_fc = .3

        in_channels = 9  # Fixed input channel size

        # Define the first convolution layer with depthwise convolution
        out_channels_1 = trial.suggest_int("out_channels_1", 9, 9 * 16, 9)
        kernel_size_1 = trial.suggest_int("kernel_size_1", 3, 7, 2)
        stride_1 = trial.suggest_int("stride_1", 1, 2)
        # padding_1 = trial.suggest_int("padding_1", 1, 3)
        padding_1 = 3

        conv_layers = [nn.Conv1d(in_channels, out_channels_1, kernel_size_1,stride=stride_1, padding=padding_1, groups=in_channels)]
        
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.LazyBatchNorm1d())
        conv_layers.append(nn.Dropout(dropout_rate_conv)),
        conv_layers.append(nn.Dropout1d(dropout1d_rate_conv))
        
        in_channels = out_channels_1

        # Initialize variable for counting total stride of conv layers:
        stride_count = 1 if stride_1 == 2 else 0

        out_channels_middle = trial.suggest_int(f"out_channels_middle", 16, 256)

        
        for i in range(2, num_conv_layers):
            kernel_size_i = trial.suggest_int(f"kernel_size_{i}", 3, 11, 2)
            stride_i = trial.suggest_int(f"stride_{i}", 1, 2)
            # padding_i = trial.suggest_int(f"padding_{i}", 1, 5)
            padding = 3 

            conv_layers.append(nn.Conv1d(
                in_channels, out_channels_middle, kernel_size_i,
                stride=stride_i, padding=padding
            ))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.LazyBatchNorm1d())
            conv_layers.append(nn.Dropout(dropout_rate_conv)),
            conv_layers.append(nn.Dropout1d(dropout1d_rate_conv)),
            in_channels = out_channels_middle
            if stride_i > 1:
                stride_count += 1
        i += 1
        
        kernel_size_last = trial.suggest_int(f"kernel_size_{i}", 3, 11, 2)
        stride_last = trial.suggest_int(f"stride_{i}", 1, 2)
        padding = 3
        last_conv_layer_out = trial.suggest_int(f"last_conv_layer_out", 32, 128)
        conv_layers.append(nn.Conv1d(   
            in_channels, last_conv_layer_out, kernel_size_last,
            stride=stride_last, padding=padding
            ))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.LazyBatchNorm1d())
        conv_layers.append(nn.Dropout(dropout_rate_conv)),
        conv_layers.append(nn.Dropout1d(dropout1d_rate_conv)),
        if stride_i > 1:
            stride_count += 1
        
        if stride_count < 3:            # StrideCount 3 => ~2^7 = 256 time steps
            raise optuna.TrialPruned()

        # Define the first fully connected layer with LazyLinear
        fc_output_size = trial.suggest_int("fc_output_size_0", 64, 256)
        fc_layers = [
            nn.LazyLinear(fc_output_size),
            nn.ReLU(),
            nn.LazyBatchNorm1d(),
            nn.Dropout(dropout_rate_fc)
        ]

        fc_mid_output_size = trial.suggest_int(f"fc_mid_output_size", 32, 128)
        # Define the rest of the fully connected layers
        # for i in range(1, trial.suggest_int("num_fc_layers", 1, 3)):
        for i in range(1,num_fc_layers-1):    
            
            fc_layers.extend([
                nn.LazyLinear(fc_mid_output_size),
                nn.ReLU(),
                nn.LazyBatchNorm1d(),
                nn.Dropout(dropout_rate_fc)
            ])
            # fc_input_size = fc_output_size

        # Output layer
        fc_layers.append(nn.LazyLinear(num_classes))   # Output layer

        # Combine all layers
        self.conv_layers = nn.Sequential(*conv_layers)
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):

        x = self.conv_layers(x)

        # Reshape for fully connected layers
        x = torch.flatten(x, 1)

        x = self.fc_layers(x)

        return x
    
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
    # characters_to_keep = ['FOX', 
    #                     #'FALCO', 
    #                     #'MARTH', 
    #                     'SHEIK', 
    #                     #'CAPTAIN_FALCON',
    #                     'YOSHI',
    #                     ]
    
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

def objective(trial, dataloaders, num_classes, characters, study_name):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model with hyperparameters suggested by Optuna
    model = CustomNet(trial,num_classes).to(device)
 
    # Get a single batch to determine the input size
    inputs, _ = next(iter(dataloaders['train']))

    # Log model info
    summary(model, input_size=inputs.size()[1:], device='cuda' if torch.cuda.is_available() else 'cpu')
    model_summary_str = str(model)
    logging.info(f'Model Summary:\n{model_summary_str}')
    
    scaler = GradScaler()
    # Define the optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    # optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
    # criterion = nn.BCEWithLogitsLoss(reduction = 'sum')

    # class_weights = torch.tensor([1, 98880 / 93713, 98880 / 54502, 98880 / 43391, 98880 / 32625], device = device)
    # class_weights.to(device)
    criterion = nn.CrossEntropyLoss(reduction = 'sum')
    def train_epoch(model, dataloader, optimizer, criterion, scaler):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

 # Forward pass with mixed precision
            # with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # Apply .float() to labels

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # optimizer.zero_grad()
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()    

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
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass with mixed precision
                # with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)  # Apply .float() to labels

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

    def evaluate_test(model, dataloader, criterion, study_name,characters):
        model.eval()  # Set model to evaluation mode
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        y_true = []
        y_pred = []
        
        confusion_matrix_path = 'data\\classifyall\\confusion_matrices\\' + study_name + '\\'
        os.makedirs(confusion_matrix_path, exist_ok=True)
        with torch.no_grad():  # No gradients needed
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                # with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)  # Apply .float() to labels

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
            # opponents = [
            #     'FOX', 
            #     'FALCO', 
            #     'MARTH', 
            #     'SHEIK', 
            #     'CAPTAIN_FALCON', 
            #     'PEACH', 
            #     'JIGGLYPUFF', 
            #     'SAMUS', 
            #     'ICE_CLIMBERS', 
            #     'GANONDORF', 
            #     'YOSHI', 
            #     'LUIGI', 
            #     'PIKACHU', 
            #     'DR_MARIO', 
            #     'NESS', 
            #     'LINK', 
            #     'MEWTWO', 
            #     'GAME_AND_WATCH', 
            #     'DONKEY_KONG', 
            #     'YOUNG_LINK', 
            #     'MARIO', 
            #     'ROY', 
            #     'BOWSER', 
            #     'ZELDA', 
            #     'KIRBY', 
            #     'PICHU'
            #     ]
            plt.figure(figsize=(1.5 * len(characters), 1.5 * len(characters)))
            sns.heatmap(cm, annot=True, fmt='f', cmap='Blues', xticklabels=characters, yticklabels=characters)
            plt.gca().invert_yaxis()
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.savefig(confusion_matrix_path + current_datetime_string + ' Confusion Matrix (pred).png')

            return test_loss, test_accuracy
                
    # Training loop with early stopping and tqdm progress bar
    patience = 2
    epochs = 2
    min_delta = 0.0001
    min_overfit = .1

    best_val_loss = float('inf')
    best_val_acc = 0.0
    epochs_no_improve = 0
    epochs_overfit = 0

    # Initialize tqdm progress bar
    pbar = tqdm(total=epochs, desc="Epochs", position=0, leave=True)

    for epoch in range(epochs):
        train_loss, train_accuracy = train_epoch(model, dataloaders['train'], optimizer, criterion,scaler)
        val_loss, val_accuracy = validate_epoch(model, dataloaders['val'], criterion)
        
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

        # Update progress bar
        pbar.set_postfix_str(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Best Validation Accuracy: {best_val_acc:.4f}")
        pbar.update(1)  # Move the progress bar by one epoch
        # Log Losses
        logging.info(f'Epoch {epoch + 1}/{epochs} - Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Best Validation Accuracy: {best_val_acc:.4f}')

        # Check early stopping condition
        if epochs_no_improve >= patience or epochs_overfit >= patience:
            pbar.write(f'Early stopping triggered at epoch {epoch + 1}')
            pbar.close()  # Close the progress bar
            break

    # Evaluate model on test set after training is complete (if necessary)
    test_loss, test_accuracy = evaluate_test(model, dataloaders['test'], criterion, study_name,characters)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    logging.info(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    pbar.close()  # Ensure the progress bar is closed
    gc.collect()
    torch.cuda.empty_cache()
    return best_val_loss


def main():
    # Ensure reproducibility with a unique(ish) string
    seed = int(datetime.now().strftime('%Y%m%d%H%M%S')) % (2**32 - 1)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    sampler = optuna.samplers.TPESampler(seed = seed)

    # Set Some Variables
    # study_name = current_datetime_string + "Basic CNN - Classify All Characters"
    study_name = "Multiday CNN - Classify All - 2 Epochs Per Trial"
    batch_size = 256

    
        
    current_time = datetime.now()
    target_time = datetime(current_time.year, current_time.month, current_time.day, 5, 0, 0)

    # If it's already past 5 am today, calculate the time until 5 am tomorrow
    if current_time >= target_time:
        target_time += timedelta(days=1)

    time_difference = target_time - current_time
    hours_until_5_am = time_difference.total_seconds() / 3600

    # Set up logging file. Log the training hours and seed.
    log_file = 'C:\\Users\\jaspa\\Grant ML\\slp\\data\\classifyall\\logs\\' + study_name + ' Log.txt'
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Commencing training for {hours_until_5_am} hours.")
    logging.info(f"Seed: {seed}")

    save_path = 'C:\\Users\\jaspa\\Grant ML\\slp\\data'
    train_df, val_df, test_df = load_data(save_path)
    num_classes = max(train_df['labels']) + 1
    characters = train_df['character'].unique()
    loaders = prepare_data_loaders(train_df, val_df, test_df, batch_size)

    study = optuna.create_study(study_name = study_name,
                                sampler = sampler,
                                direction = "minimize",
                                storage = "mysql+pymysql://root:MomentusPigs@localhost:3306/optuna_trials",
                                load_if_exists = True
                                )

    objective_with_loaders = lambda trial: objective(trial, loaders, num_classes, characters, study_name = study_name)
    study.optimize(objective_with_loaders, n_trials = 1000, show_progress_bar = True, timeout=3600 * hours_until_5_am)

    # Print the overall best hyperparameters
    print("Best trial overall:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == '__main__':
    main()

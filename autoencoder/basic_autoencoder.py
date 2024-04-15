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

import sys
sys.path.insert(0,'/workspace/slp_jaspar/')
# sys.path.append('../')
# from slp_package.slp_functions import create_merged_game_data_df
# from input_dataset import InputDataSet
# import pytorch_functions as slp_pytorch_functions
from slp_package.input_dataset import InputDataSet
import slp_package.pytorch_functions as slp_pytorch_functions



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
        self.fc2 = nn.LazyLinear(128, 5)  # Assuming 5 classes for classification

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
    
def main():
    
    # We classify 5 characters on competitive stages

    source_data = ['ranked']

    general_features = {
        'stage_name': ['FOUNTAIN_OF_DREAMS','FINAL_DESTINATION','BATTLEFIELD','YOSHIS_STORY','POKEMON_STADIUM','DREAMLAND'],
        'num_players': [2],
        'conclusive': [True]
    }
    player_features = {
        # 'netplay_code': ['MANG#0'],
        'character_name': ['FOX', 'FALCO', 'MARTH', 'CAPTAIN_FALCON', 'SHEIK'],
        # 'character_name': ['FOX', 'SHEIK'],
        'type_name': ['HUMAN']
        
    }
    opposing_player_features = {
        # 'character_name': ['MARTH'],
        # 'netplay_code': ['KOD#0', 'ZAIN#0']
        'type_name': ['HUMAN']
    }
    label_info = {
        'source': ['player'], # Can be 'general', 'player
        # 'feature': ['netplay_code']
        'feature': ['character_name']
    }
  
    dataset = InputDataSet(source_data, general_features, player_features, opposing_player_features, label_info)

    print(dataset.dataset['labels'].value_counts())
    dataset.dataset.head()
    
    labels_order =  dataset.number_of_segments_per_game(12,6000)
    print(labels_order)
    labels_order = labels_order['Label'].values

    
    train_df, test_df  = dataset.train_test_split_dataframes(test_ratio = .20, val = False)
    
    loaders = slp_pytorch_functions.prepare_data_loaders_no_val(train_df, test_df, 32, 16)
    

    
    # start_time = time.time()
    model = SimpleCNN().to('cuda')
    criterion = nn.CrossEntropyLoss(reduction = 'sum')
    optimizer = Adam(model.parameters(), lr=0.001)
    # gc.collect()
    # torch.cuda.empty_cache()
    slp_pytorch_functions.train_model(model, criterion, optimizer, loaders, 'cuda', 20)
    slp_pytorch_functions.evaluate_model(model, loaders['test'], 'cuda')
    
    # print('Time to train Rocket:', end_time - start_time)
if __name__ == '__main__':
    main()

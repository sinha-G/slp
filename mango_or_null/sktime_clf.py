from sktime.classification.deep_learning.cnn import CNNClassifier, LSTMFCNClassifier
from sktime.classification.kernel_based import RocketClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchsummary import summary

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
# from prettytable import PrettyTable

from datetime import datetime
import os
import gc
import logging

def main():
    # Ensure reproducibility (TODO: fix this - I don't think this works)
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
    model = CNNClassifier(n_epochs=epochs, batch_size=batch_size, loss = 'categorical_crossentropy')

    # Set up logging file
    log_file = 'data\\mango_or_null\\logs\\' + study_name + ' Log.txt'
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f'Seed: {seed}')

    # model.fit(X, y)


if __name__ == '__main__':
    main()

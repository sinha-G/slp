import os
import sys
import gzip

import pandas as pd
import numpy as np

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sktime.classification.deep_learning.cnn import CNNClassifier
# from sktime.classification.kernel_based import RocketClassifier

import tqdm
from joblib import Parallel, delayed
from multiprocessing import Manager

# sys.path.append('../..')
# from slp.slp_package.slp_functions import create_merged_game_data_df
# from slp.slp_package.input_dataset import InputDataSet

import tensorflow as tf

sys.path.append('..')
from slp.slp_package.input_dataset import InputDataSet

def main():
    source_data = ['ranked']

    general_features = {
        'stage_name': ['FOUNTAIN_OF_DREAMS','FINAL_DESTINATION','BATTLEFIELD','YOSHIS_STORY','POKEMON_STADIUM'],
        'num_players': [2],
        'conclusive': [True]
    }
    player_features = {
        # 'netplay_code': ['MANG#0'],
        'character_name': ['FOX', 'FALCO', 'MARTH', 'CAPTAIN_FALCON', 'SHEIK'],
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

    dataset.number_of_segments_per_game(10, 100000)
    X_train, X_test, y_train, y_test  = dataset.train_test_split_numpy(test_ratio = .30,  val = False)
    
    print(X_train.shape)

    batch_size = 32
    epochs = 50
    model = CNNClassifier(n_epochs=epochs, batch_size=batch_size, loss = 'categorical_crossentropy')
    # model = RocketClassifier(n_jobs = -1)

    model.fit(X_train, y_train)
    
    acc = model.score(X_test, y_test)

    print(acc)
    
    Explicitly set the device to GPU (if available)
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))  

if __name__ == '__main__':
    main()
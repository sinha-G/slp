import pandas as pd
import numpy as np
import tensorflow as tf
import time

from sktime.classification.deep_learning.cnn import CNNClassifier
from sktime.classification.deep_learning.resnet import ResNetClassifier
from sktime.classification.deep_learning import InceptionTimeClassifier
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.hybrid import HIVECOTEV2


from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt

import sys
sys.path.append('..')
# from slp_package.slp_functions import create_merged_game_data_df
from slp_package.input_dataset import InputDataSet
import joblib

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
    
    # dataset.number_of_segments_per_game(10, 50000)
    # dataset.number_of_segments_per_game(10, 10000)
    dataset.number_of_segments_per_game(10, 500)
    
    X_train, X_test, y_train, y_test  = dataset.train_test_split_numpy(test_ratio = .30,  val = False)
    
    start_time = time.time()
    clf = RocketClassifier(num_kernels=1000, n_jobs=-1, random_state=42, rocket_transform = 'minirocket') 
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test) 
    print('Accuracy of Rocket:', accuracy_score(y_test, y_pred))
    # ConfusionMatrixDisplay.from_predictions(y_test, y_pred,normalize='true', xticks_rotation='vertical')
    # plt.show()
    end_time = time.time()
    
    print('Time to train Rocket:', end_time - start_time)
if __name__ == '__main__':
    main()

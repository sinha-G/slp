import os
import sys
import gzip

import pandas as pd
import numpy as np

from sklearn.utils import shuffle

sys.path.append('..')
from slp.slp_package import slp_functions


# Load game data df containing only mango's data
df = slp_functions.create_merged_game_data_df(['mango'])
df = df[((df['player_1_character_name'] == 'FOX') & ((df['player_1_netplay_code'] == 'MANG#0') | (df['player_1_netplay_code'] == 'NULL#478'))) |
         ((df['player_2_character_name'] == 'FOX') & ((df['player_2_netplay_code'] == 'MANG#0') | (df['player_2_netplay_code'] == 'NULL#478')))]
df = df[df['length'] >= 1024]

save_directory = '../data/'
X = []
y = []

# Get file names for desired rows
netcodes = ['MANG#0', 'NULL#478']
for netcode in netcodes:
    paths = df.loc[df['player_1_netplay_code'] == netcode, 'player_1_inputs_np_save_path'].tolist()
    X.extend(paths)
    y.extend([1 if netcode == 'MANG#0' else 0] * len(paths))
    paths = df.loc[df['player_2_netplay_code'] == netcode, 'player_2_inputs_np_save_path'].tolist()
    X.extend(paths)
    y.extend([1 if netcode == 'MANG#0' else 0] * len(paths))


# Shuffle the dataset to mix up the order of characters
X, y = shuffle(np.array(X), np.array(y), random_state=42)

# At this point, X and y are your balanced dataset ready for further processing
print('Total number of data points: ', X.shape[0])

# Save data lists
save_path = 'C:/Users/jaspa/Grant ML/slp/data/'

with gzip.open(os.path.join(save_path,'mango_or_null_X.npy.gz'), 'wb') as f:
    np.save(f, X)

with gzip.open(os.path.join(save_path,'mango_or_null_y.npy.gz'), 'wb') as f:
    np.save(f, y)

"""
input_dataset.py

Implements the InputDataSet class, which provides a higher-level API to:
 - Merge and filter Slippi data according to user-defined criteria (general features, player features, labels, etc.).
 - Partition data into training, validation, and test splits without leakage (split by game).
 - Generate data segments (e.g., 1-second or 60-second slices) for sequence modeling.
 - Convert the data into either a DataFrame or a NumPy array for training with various ML frameworks.

Usage highlights:
 1. Configure source datasets (ranked, public, mango).
 2. Define feature filters for both general game properties and specific players.
 3. Specify label info to train on (e.g., which character or netplay code).
 4. Generate training/test splits in a reproducible way, preventing data leakage by separating entire games between sets.

The class supports flexible segment lengths, overlapping segments, and optional validation sets, making it well-suited for time-series or sequence-based model development.
"""

import os
import sys
import gzip

import pandas as pd
import numpy as np

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

import tqdm
from joblib import Parallel, delayed
from multiprocessing import Manager

sys.path.append('..')
from slp_package.slp_functions import create_merged_game_data_df

class InputDataSet():
    # list atributes
    
    def __init__(self, source_data, general_features, player_features, opposing_player_features, label_info):
        self.source_data = source_data
        self.general_features = general_features
        self.player_features = player_features
        self.opposing_player_features = opposing_player_features
        self.label_info = label_info
        self.dataset = self.prepare_data_for_training()
        # The proceeding 3 attributes are assigned in the number_of_segments_per_game method
        self.divide_games_df_input = None
        self.num_segments_per_label = 0
        self.segment_length = 0
        
    def prepare_data_for_training(self):
        """
        Prepares data for training based on specified features and filters.

        :param source_data: List of sources to include in the data merge.
        :param general_features: Dictionary of general game features and their desired values.
        :param player_features: Dictionary of features for the player we are training on.
        :param opposing_player_features: Dictionary of features for the opposing player.
        :param label_info: Dictionary specifying the source and feature name for the label.
        :return: A pandas DataFrame with the prepared training data, containing only specified features and the label.
        """
        def apply_general_filters(df, filters):
            """
            Applies filters to the dataframe based on the provided dictionary of filters.

            :param df: The pandas DataFrame to filter.
            :param filters: Dictionary of column names and their desired values.
            :return: The filtered DataFrame.
            """
            for feature, values in filters.items():
                if isinstance(values, list):
                    df = df[df[feature].isin(values)]
                else:
                    df = df[df[feature] == values]
            return df

        def identify_and_label_players(df, player_features, opposing_player_features):
            """
            Identifies which player (player_1 or player_2) matches the specified features and renames columns accordingly,
            also considering the opposing player features.

            :param df: The merged DataFrame.
            :param player_features: Dictionary of features for the player we are training on.
            :param opposing_player_features: Dictionary of features for the opposing player.
            :return: DataFrame with columns renamed for player and opposing player features, including input paths.
            """
            # Reset the index of the DataFrame to ensure alignment
            df = df.reset_index(drop=True)

            # Initialize masks for player 1 and player 2
            player_1_mask = pd.Series([True] * len(df))
            player_2_mask = pd.Series([True] * len(df))

            # Update masks for player features
            for feature, values in player_features.items():
                player_1_mask &= df[f'player_1_{feature}'].isin(values) if isinstance(values, list) else df[f'player_1_{feature}'] == values
                player_2_mask &= df[f'player_2_{feature}'].isin(values) if isinstance(values, list) else df[f'player_2_{feature}'] == values

            # Update masks for opposing player features
            for feature, values in opposing_player_features.items():
                player_1_mask &= df[f'player_2_{feature}'].isin(values) if isinstance(values, list) else df[f'player_2_{feature}'] == values
                player_2_mask &= df[f'player_1_{feature}'].isin(values) if isinstance(values, list) else df[f'player_1_{feature}'] == values

            # Apply the masks to filter the DataFrame
            player_1_df = df[player_1_mask]
            player_2_df = df[player_2_mask]
            
            # Rename columns for player_1 and player_2 in their respective DataFrames
            player_1_df = player_1_df.rename(columns=lambda col: col.replace('player_1_', 'player_') if 'player_1_' in col else col.replace('player_2_', 'opposing_player_'))
            player_2_df = player_2_df.rename(columns=lambda col: col.replace('player_2_', 'player_') if 'player_2_' in col else col.replace('player_1_', 'opposing_player_'))

            # Concatenate the two DataFrames
            processed_df = pd.concat([player_1_df, player_2_df], ignore_index=True)

            return processed_df

        def extract_label(df, label_info):
            """
            Extracts the label column from the dataframe based on label_info and renames it to 'label'.

            :param df: The DataFrame to extract the label from.
            :param label_info: Dictionary specifying the source and feature name for the label.
            :return: DataFrame with the label column extracted and renamed to 'label'.
            """
            label_source = label_info['source'][0]  # Assuming label_source is passed as a list
            label_feature = label_info['feature'][0]  # Assuming label_feature is passed as a list

            # Construct the full column name based on the source
            if label_source == 'player':
                label_column = f'player_{label_feature}'
            elif label_source == 'opposing_player':
                label_column = f'opposing_player_{label_feature}'
            else:
                label_column = label_feature

            # Check if the column exists after renaming
            if label_column not in df.columns:
                raise KeyError(f"{label_column} not found in the DataFrame columns")
            
            df['labels'] = df[label_column]
            return df
        
        # Merge data from specified sources
        merged_df = create_merged_game_data_df(self.source_data)

        # Apply filters to general game data
        merged_df = apply_general_filters(merged_df, self.general_features)

        # Identify and label player and opposing player features
        merged_df = identify_and_label_players(merged_df, self.player_features, self.opposing_player_features)

        # Extract and set the label for training
        merged_df = extract_label(merged_df, self.label_info)
        
        merged_df['length'] -= 123
        
        # Define the order of columns to be selected
        general_feature_columns = list(self.general_features.keys())
        player_feature_columns = [f'player_{feature}' for feature in self.player_features.keys()]
        opposing_player_feature_columns = [f'opposing_player_{feature}' for feature in self.opposing_player_features.keys()]
        input_path_column = ['player_inputs_np_sub_path']
        length_column = ['length']
        label_column = ['labels']

        # Combine all columns in the desired order
        final_columns = general_feature_columns + player_feature_columns + opposing_player_feature_columns + input_path_column + length_column + label_column

        # Select only the specified columns from the DataFrame
        final_df = merged_df[final_columns]

        return final_df
    
    def number_of_segments_per_game(self, segment_length, num_segments_per_label):
        """
        Calculate the floating-point number of segments for each game in the dataframe based on the game's length
        and the desired total number of segments per label.

        Parameters:
        df (DataFrame): Dataframe containing game data with at least 'labels' and 'length' columns.
        segment_length_power (int): Power of 2 to determine the segment length.
        num_segments_per_label (int): Desired total number of segments per label.

        Returns:
        DataFrame: Updated dataframe with an additional column 'float_num_segments'.
        DataFrame: Summary information about the labels, their counts, and estimated shift values.
        """
        # Copy the dataframe to avoid modifying the original data
        df = self.dataset.copy()

        # Calculate segment length as a power of 2
        # segment_length = 2 ** segment_length_power

        # Filter out games where length is less than or equal to the segment length
        df = df[df['length'] > segment_length]

        # Initialize the column to store the floating-point number of segments
        df['float_num_segments'] = 0.0

        # Initialize a list to store information about each label for later summary
        label_info_list = []

        # Iterate through each unique label to process segments
        for label in df['labels'].unique():
            # Identify rows matching the current label
            label_indices = df['labels'] == label

            # Adjust game length to ensure segments fit within the game length
            adjusted_game_length = df.loc[label_indices, 'length'] - segment_length

            # Sum the lengths of all games with the current label to estimate the shift value
            game_length_sum = adjusted_game_length.sum()
            shift_estimate = game_length_sum / num_segments_per_label

            # Calculate the floating-point number of segments for each game
            df.loc[label_indices, 'float_num_segments'] = adjusted_game_length / shift_estimate

            # Collect label information including the total count and shift estimate
            label_info = [label, adjusted_game_length.count(), round(shift_estimate)]
            label_info_list.append(label_info)

        # Create a dataframe from the label information list
        label_info_df = pd.DataFrame(label_info_list, columns=['Label', 'Count', 'Shift'])

        # Sort the label_info DataFrame by 'Count' in descending order for better readability
        label_info_df = label_info_df.sort_values(by='Count', ascending=False).reset_index(drop=True)
        
        return_columns = ['player_inputs_np_sub_path',  'length', 'labels','float_num_segments']

        self.divide_games_df_input = df[return_columns]
        self.segment_length = segment_length
        self.num_segments_per_label = num_segments_per_label

        return label_info_df

    def divide_games(self, test_ratio=0.15, val_ratio=0.15, val=True):
        """
        Splits the games into training, testing, and optionally validation sets based on the approximate number of segments per game.
        
        Parameters:
        df (DataFrame): The output of number_of_segments_per_game containing game data with 'labels' and 'float_num_segments'.
        num_segments_per_label (int): Total number of segments desired per label.
        test_ratio (float): The proportion of data to be used for the test set.
        val_ratio (float): The proportion of data to be used for the validation set.
        val (bool): Whether to create a validation set.
        
        Returns:
        test_df (DataFrame): Data for testing.
        val_df (DataFrame): Data for validation (if val is True, otherwise an empty DataFrame).
        train_df (DataFrame): Data for training.
        """

        # Copy the dataframe to avoid modifying the original data
        df = self.divide_games_df_input.copy()
        
        # Initialize empty lists to store split dataframes
        test_dfs, val_dfs, train_dfs = [], [], []

        # Calculate the number of segments for each split based on the provided ratios
        num_segments_test = round(self.num_segments_per_label * test_ratio)
        num_segments_val = round(self.num_segments_per_label * val_ratio) * val
        num_segments_train = self.num_segments_per_label - num_segments_test - num_segments_val
        
        # Process each label separately
        for label in df['labels'].unique():
            # Filter the dataframe for the current label and shuffle
            label_df = df[df['labels'] == label].sample(frac=1).reset_index(drop=True)
            # Ensure 'float_num_segments' is in label_df before proceeding
            if 'float_num_segments' not in label_df.columns:
                print(f"'float_num_segments' column is missing in label_df for label {label}")
                continue  # Skip this label if the required column is missing
            
            # Calculate cumulative sum to find the cutoff points for splitting
            num_segments_cumsum = label_df['float_num_segments'].cumsum()

            # Determine the index to split test and train datasets
            test_idx = num_segments_cumsum[num_segments_cumsum <= num_segments_test].last_valid_index() or 0
            val_idx = num_segments_cumsum[num_segments_cumsum <= num_segments_test + num_segments_val].last_valid_index() or test_idx

            # Split the data based on calculated indices
            test_label_df = label_df.iloc[:test_idx + 1].copy()
            val_label_df = label_df.iloc[test_idx + 1:val_idx + 1].copy() if val else pd.DataFrame(columns = label_df.columns)
            train_label_df = label_df.iloc[val_idx + 1:].copy()
            # print(test_label_df.head())

            # Calculate the actual number of segments to extract for each set
            # This process adjusts the 'num_segments' by distributing the rounding errors across the segments
            # to ensure that the total number of segments remains as close as possible to the desired count
            for split_df, num_segments_split in zip(
                [test_label_df, val_label_df, train_label_df],
                [num_segments_test, num_segments_val, num_segments_train]
            ):
                # Start with floor values of 'float_num_segments' and calculate the residual fractional part
                split_df['num_segments'] = split_df['float_num_segments'].astype(int)
                split_df['frac_part'] = split_df['float_num_segments'] - split_df['num_segments']
                split_df.sort_values(by='frac_part', ascending=False, inplace=True)

                # Distribute rounding residuals to match the total segment count precisely
                residual_count = num_segments_split - split_df['num_segments'].sum()
                split_df.iloc[:residual_count, split_df.columns.get_loc('num_segments')] += 1

            # Append the processed dataframes to their respective lists
            test_dfs.append(test_label_df)
            val_dfs.append(val_label_df)
            train_dfs.append(train_label_df)

        # Concatenate all the dataframes in each list to create the final splits
        return_columns = ['player_inputs_np_sub_path',  'length', 'num_segments','labels']
        test_df = pd.concat(test_dfs, ignore_index=True)[return_columns]
        val_df = pd.concat(val_dfs, ignore_index=True)[return_columns] if val else pd.DataFrame(columns=return_columns)
        train_df = pd.concat(train_dfs, ignore_index=True)[return_columns]
        
        # Encode the labels for training
        label_encoder = LabelEncoder()
        label_encoder.fit(df['labels'].unique())
        test_df['encoded_labels'] = label_encoder.fit_transform(test_df['labels'])
        val_df['encoded_labels'] = label_encoder.fit_transform(val_df['labels'])
        train_df['encoded_labels'] = label_encoder.fit_transform(train_df['labels'])

        return test_df, val_df, train_df

    # def create_training_dataframe(self, df):
    #     """
    #     Generate a DataFrame that lists the segments for training, where each row corresponds to a segment.
        
    #     Parameters:
    #     df (DataFrame): DataFrame containing the output from `divide_games`, which includes 'num_segments' and 'length'.
    #     segment_length_power (int): The power of 2 used to determine the segment length.
        
    #     Returns:
    #     DataFrame: A new DataFrame where each row represents a segment, including the start index of each segment.
    #     """
    #     # Calculate the segment length as a power of 2
    #     # segment_length = 2 ** self.segment_length_power
        
    #     # Retrieve the 'num_segments' column as an array to determine how many times to repeat each row
    #     repeats = df['num_segments'].values

    #     # Repeat each index in the DataFrame according to the number of segments it should be split into
    #     index_repeated = np.repeat(df.index, repeats)
        
    #     # Duplicate rows in the DataFrame based on the repeat counts for each row
    #     df_repeated = df.loc[index_repeated].reset_index(drop=True)
        
    #     # Generate a sequential 'segment_index' for each group of repeated rows
    #     segment_indices = np.concatenate([np.arange(n, dtype=np.int16) for n in repeats])
        
    #     # Calculate the start index of each segment within the game
    #     df_repeated['segment_start_index'] = ((df_repeated['length'] - self.segment_length) // df_repeated['num_segments']) * segment_indices
        
    #     # Drop columns that are no longer necessary after computing 'segment_start_index'
    #     df_repeated = df_repeated.drop(columns=['length', 'num_segments'])

    #     # Add 'segment_index' to the DataFrame to keep track of each segment within its group
    #     df_repeated['segment_index'] = segment_indices
        
    #     df_repeated['segment_length'] = self.segment_length
        
    #     return df_repeated
    
    def create_training_dataframe(self, df, shift=None):
        """
        Generate a DataFrame that lists the segments for training, where each row corresponds to a segment.

        Parameters:
        -----------
        df (DataFrame): 
            Must include ['num_segments', 'length'] columns (output from `divide_games` or `divide_games_all_segments`).
        shift (int): 
            The difference in start indices between consecutive segments (controls overlap). 
            If None, defaults to `segment_length` for no overlap.

        Returns:
        --------
        DataFrame: A new DataFrame where each row represents a segment, 
                    including the start index of each segment ('segment_start_index').
        """
        # If shift isn't given, default to using segment_length (so no overlap by default)
        if shift is None:
            shift = self.segment_length

        # We want one row per segment, so we look at the 'num_segments' column:
        repeats = df['num_segments'].values  # how many segments each game row is split into
        # Example: If num_segments[i] = 4, that row repeats 4 times.

        # Expand the DataFrame by repeating each row according to 'num_segments'
        index_repeated = np.repeat(df.index, repeats)
        df_repeated = df.loc[index_repeated].reset_index(drop=True)

        # Create a segment_index array: 0, 1, 2, ... for each repeated group
        segment_indices = np.concatenate([np.arange(n, dtype=np.int32) for n in repeats])
        df_repeated['segment_index'] = segment_indices

        # The start index of each segment. 
        #   first segment: 0
        #   second: shift
        #   n-th: n * shift
        df_repeated['segment_start_index'] = df_repeated['segment_index'] * shift

        # Optionally drop columns you do not need
        # df_repeated = df_repeated.drop(columns=['length', 'num_segments'])

        # We'll store the chosen segment_length if you want it
        df_repeated['segment_length'] = self.segment_length

        return df_repeated

    # def create_training_numpy(df, segment_length_power):
    def create_training_numpy(self, df):
        """
        Creates a NumPy array containing all the segments from the dataframe, with parallel processing for efficiency.
        
        Parameters:
        df (DataFrame): The DataFrame containing game data, must be one of the outputs from `divide_games`.
        segment_length_power (int): The power of 2 that defines the length of each segment.
        
        Returns:
        tuple: A tuple containing two elements; the first is a NumPy array of input arrays, 
            and the second is a NumPy array of corresponding labels.
        """
        
        def process_game(path, label, length, num_segments, segment_length):
            """
            Loads the game data from the specified path and extracts segments of the specified length.
            
            Parameters:
            path (str): File path to the numpy array containing game data.
            label (str): The label associated with the game data.
            length (int): The total length of the game data.
            num_segments (int): The number of segments to be extracted from the game data.
            segment_length (int): The length of each segment.
            
            Appends the extracted segments and their labels to a shared list accessible by the parent process.
            """
            # Return immediately if there are no segments to process
            if num_segments == 0:
                return
            path = path.replace('\\','/')
            # Load the game data from the specified path
            with gzip.open('/workspace/melee_project_data/input_np/' + path, 'rb') as f:
                inputs_array = np.load(f)
            
            # Initialize an array to hold the extracted segments
            segments_array = np.empty((num_segments, 9, segment_length), dtype=np.single)
            
            # Calculate the shift between starting points of consecutive segments
            segment_shift = (length - segment_length) // num_segments
            
            # Extract segments from the input array
            for i in range(num_segments):
                start_index = segment_shift * i
                segments_array[i, :, :] = inputs_array[:, start_index : start_index + segment_length]
            
            # Append the extracted segments and their label to the shared list
            shared_list.append((segments_array, [label] * num_segments))
        
        # Calculate the segment length using the power of 2
        segment_length = self.segment_length
        
        # Prepare tasks for parallel processing
        tasks = [
            (row['player_inputs_np_sub_path'], row['labels'], row['length'], row['num_segments']) 
            for index, row in df.iterrows()
        ]
        
        # Use Manager to create a shared list for collecting results from parallel processes
        manager = Manager()
        shared_list = manager.list()
        
        # Process each game in parallel to extract segments
        Parallel(n_jobs=-1, verbose=0)(
            delayed(process_game)(task[0], task[1], task[2], task[3], segment_length) 
            for task in tqdm.tqdm(tasks)
        )
        
        # After parallel processing, extract the segments and labels from the shared list
        input_arrays, label_lists = zip(*list(shared_list))
        
        # Combine all segment arrays into one array and all labels into one list
        input_array = np.concatenate(input_arrays, axis=0)
        labels = np.concatenate(label_lists)

        return input_array, labels

    def train_test_split_numpy(self, test_ratio = .15, val_ratio = .15, val = True):
        test_df, val_df, train_df = self.divide_games(test_ratio, val_ratio, val)
        
        X_train, y_train = self.create_training_numpy(train_df)
        
        X_test, y_test = self.create_training_numpy(test_df)
        
        if not val_df.empty:
            X_val, y_val = self.create_training_numpy(val_df)
            return X_train, X_test, X_val, y_train, y_test, y_val
        
        return X_train, X_test, y_train, y_test 
    
    
    def train_test_split_dataframes(self, test_ratio = .15, val_ratio = .15, val = True):
        test_df, val_df, train_df = self.divide_games(test_ratio, val_ratio, val)
        print(test_df.head())
        
        X_train_df = self.create_training_dataframe(train_df)
        
        X_test_df = self.create_training_dataframe(test_df)
        
        if not val_df.empty:
            X_val_df = self.create_training_dataframe(val_df)
            return X_train_df, X_test_df, X_val_df
    
        return X_train_df, X_test_df   
    
    # def divide_games_all_segments(self, segment_length, proportion_of_segments=1, test_ratio = .15, val_ratio = .15, val = True):
        
    #     # Copy the dataframe to avoid modifying the original data
    #     df = self.dataset.copy()
    #     df = df[df['length'] > segment_length]
    #     df = df[df['length'] <= 8*60*60]
        
    #     # Add a column with the number of segments you will get from each game
    #     df['num_segments'] = round((df['length'] // segment_length) * proportion_of_segments)
        
    #     # print(df.head())
        
    #     # Initialize empty lists to store split dataframes
    #     test_dfs, val_dfs, train_dfs = [], [], []
        
    #     # Process each label separately
    #     for label in df['labels'].unique():
    #         # Filter the dataframe for the current label and shuffle
    #         label_df = df[df['labels'] == label].sample(frac=1).reset_index(drop=True)
            
    #         # Calculate cumulative sum to find the cutoff points for splitting
    #         num_segments_cumsum = label_df['num_segments'].cumsum()
            
    #         num_segments_test = round(num_segments_cumsum.iloc[-1] * test_ratio)
    #         # print(label, num_segments_cumsum.iloc[-1], num_segments_test,  num_segments_test / num_segments_cumsum.iloc[-1])
    #         num_segments_val = round(num_segments_cumsum.iloc[-1] * val_ratio) * val

    #         # Determine the index to split test and train datasets
    #         test_idx = num_segments_cumsum[num_segments_cumsum <= num_segments_test].last_valid_index() or 0
    #         val_idx = num_segments_cumsum[num_segments_cumsum <= num_segments_test + num_segments_val].last_valid_index() or test_idx
    
            
            
    #         # Split the data based on calculated indices
    #         test_label_df = label_df.iloc[:test_idx + 1].copy()
    #         val_label_df = label_df.iloc[test_idx + 1:val_idx + 1].copy() if val else pd.DataFrame(columns = label_df.columns)
    #         train_label_df = label_df.iloc[val_idx + 1:].copy()
            

    #         # Append the processed dataframes to their respective lists
    #         test_dfs.append(test_label_df)
    #         val_dfs.append(val_label_df)
    #         train_dfs.append(train_label_df)
            

    #         # print(label, test_label_df['num_segments'].sum() / train_label_df['num_segments'].sum() )
    #         # print()
        
    #     # Concatenate all the dataframes in each list to create the final splits
    #     return_columns = ['player_inputs_np_sub_path',  'length', 'num_segments','labels']
    #     test_df = pd.concat(test_dfs, ignore_index=True)[return_columns]
    #     val_df = pd.concat(val_dfs, ignore_index=True)[return_columns] if val else pd.DataFrame(columns=return_columns)
    #     train_df = pd.concat(train_dfs, ignore_index=True)[return_columns]
        
    #     # Encode the labels for training
    #     label_encoder = LabelEncoder()
    #     label_encoder.fit(df['labels'].unique())
    #     test_df['encoded_labels'] = label_encoder.fit_transform(test_df['labels'])
    #     val_df['encoded_labels'] = label_encoder.fit_transform(val_df['labels'])
    #     train_df['encoded_labels'] = label_encoder.fit_transform(train_df['labels'])


    #     return test_df, val_df, train_df
    
    def divide_games_all_segments(self, segment_length, proportion_of_segments=1, 
                                  test_ratio=0.15, val_ratio=0.15, val=True):
        """
        Splits games into train, test, and (optionally) validation sets by taking
        *all* possible segments of length 'segment_length' from each game, 
        without the floating/approx approach.

        Each game i has:
          num_segments_i = floor((length_i - segment_length) / shift) + 1 
          if shift is used outside. But here we do a simpler approach 
          (unless we incorporate shift below).

        NOTE: We’ll modify this so that if shift is given later, 
              we actually compute with shift. For now, we do the original method 
              (no overlap logic).
        """

        # We'll do a simpler approach: only include games with length >= segment_length
        df = self.dataset.copy()
        df = df[df['length'] >= segment_length]
        # Some safety guard in case there's extremely long games
        df = df[df['length'] <= 8*60*60]  # up to 8 hours?

        # Original logic for number_of_segments:
        # df['num_segments'] = round((df['length'] // segment_length) * proportion_of_segments)
        # We'll keep this line here, but we might move it to a separate function 
        # if you want shift-based overlap. For now, let's keep the original logic:
        df['num_segments'] = (df['length'] // segment_length)  # integer division
        df['num_segments'] = round(df['num_segments'] * proportion_of_segments)
        df.loc[df['num_segments'] < 0, 'num_segments'] = 0

        # Now do your test/val/train splits by label, proportionate to 'num_segments'
        test_dfs, val_dfs, train_dfs = [], [], []
        for label in df['labels'].unique():
            label_df = df[df['labels'] == label].sample(frac=1).reset_index(drop=True)
            cumsum_segments = label_df['num_segments'].cumsum()

            total_segments = cumsum_segments.iloc[-1] if len(cumsum_segments) > 0 else 0
            test_cut = round(total_segments * test_ratio)
            val_cut  = round(total_segments * val_ratio) if val else 0

            test_idx = cumsum_segments[cumsum_segments <= test_cut].last_valid_index() or 0
            val_idx  = cumsum_segments[cumsum_segments <= (test_cut + val_cut)].last_valid_index() or test_idx

            test_label_df = label_df.iloc[:test_idx + 1].copy()
            val_label_df  = label_df.iloc[test_idx + 1: val_idx + 1].copy() if val else pd.DataFrame(columns=label_df.columns)
            train_label_df= label_df.iloc[val_idx + 1:].copy()

            test_dfs.append(test_label_df)
            val_dfs.append(val_label_df)
            train_dfs.append(train_label_df)

        return_cols = ['player_inputs_np_sub_path', 'length', 'num_segments', 'labels']
        test_df  = pd.concat(test_dfs, ignore_index=True)[return_cols]
        val_df   = pd.concat(val_dfs,  ignore_index=True)[return_cols] if val else pd.DataFrame(columns=return_cols)
        train_df = pd.concat(train_dfs,ignore_index=True)[return_cols]

        # Encode labels 
        label_encoder = LabelEncoder()
        label_encoder.fit(df['labels'].unique())
        test_df['encoded_labels']  = label_encoder.transform(test_df['labels'])
        val_df['encoded_labels']   = label_encoder.transform(val_df['labels']) if val else None
        train_df['encoded_labels'] = label_encoder.transform(train_df['labels'])

        return test_df, val_df, train_df
    
    # def all_segments_train_test_split_dataframes(self, segment_length, proportion_of_segments = 1, test_ratio = .15, val_ratio = .15, val = True):
    #     test_df, val_df, train_df = self.divide_games_all_segments(segment_length,proportion_of_segments, test_ratio, val_ratio, val)
    #     # print(test_df.head())
        
    #     X_train_df = self.create_training_dataframe(train_df)
        
    #     X_test_df = self.create_training_dataframe(test_df)
        
    #     if not val_df.empty:
    #         X_val_df = self.create_training_dataframe(val_df)
    #         return X_train_df, X_test_df, X_val_df
    
    #     return X_train_df, X_test_df   
    
    def all_segments_train_test_split_dataframes(self, segment_length, shift=None, 
                                                 proportion_of_segments=1, 
                                                 test_ratio=0.15, val_ratio=0.15, val=True):
        """
        Extends `divide_games_all_segments` to allow overlapping segments by specifying `shift`.

        Parameters:
        -----------
        segment_length (int): 
            Number of frames in each segment.
        shift (int or None): 
            How many frames to move forward for each subsequent segment (per game).
            - If None, defaults to `segment_length` (no overlap).
            - If shift < segment_length, segments will overlap.
            - If shift > segment_length, there will be a gap.
        proportion_of_segments (float):
            A factor to multiply the total number of segments by (if you want to reduce or increase).
        test_ratio (float): 
            Proportion of segments for the test set.
        val_ratio (float): 
            Proportion of segments for the validation set.
        val (bool): 
            Whether to create a validation set.

        Returns:
        --------
        (X_train_df, X_test_df, [X_val_df if val=True])
        Each one is a dataframe with repeated rows: 
            each row = one segment, including 'segment_start_index'.
        """
        self.segment_length = segment_length

        # Copy the dataset
        df = self.dataset.copy()
        # Only keep games at least as long as segment_length
        df = df[df['length'] >= segment_length]
        df = df[df['length'] <= 8*60*60]

        # If shift is None, default to no overlap
        if shift is None:
            shift = segment_length

        # Calculate how many segments per game using overlap logic:
        # num_segments = floor((length - segment_length)/shift) + 1
        df['num_segments'] = (df['length'] - segment_length) // shift + 1
        # If it's negative or zero, fix that:
        df.loc[df['num_segments'] < 0, 'num_segments'] = 0
        # Apply proportion_of_segments
        df['num_segments'] = round(df['num_segments'] * proportion_of_segments)

        # Split into train/test/val by label, proportionate to the 'num_segments'
        test_dfs, val_dfs, train_dfs = [], [], []
        for label in df['labels'].unique():
            label_df = df[df['labels'] == label].sample(frac=1, random_state=42).reset_index(drop=True)
            cumsum_segments = label_df['num_segments'].cumsum()

            total_segments = cumsum_segments.iloc[-1] if len(cumsum_segments) > 0 else 0
            test_cut = round(total_segments * test_ratio)
            val_cut  = round(total_segments * val_ratio) if val else 0

            test_idx = cumsum_segments[cumsum_segments <= test_cut].last_valid_index() or 0
            val_idx  = cumsum_segments[cumsum_segments <= (test_cut + val_cut)].last_valid_index() or test_idx

            test_label_df = label_df.iloc[:test_idx + 1].copy()
            val_label_df  = label_df.iloc[test_idx + 1: val_idx + 1].copy() if val else pd.DataFrame(columns=label_df.columns)
            train_label_df= label_df.iloc[val_idx + 1:].copy()

            test_dfs.append(test_label_df)
            val_dfs.append(val_label_df)
            train_dfs.append(train_label_df)

        return_cols = ['player_inputs_np_sub_path', 'length', 'num_segments', 'labels']
        test_df  = pd.concat(test_dfs, ignore_index=True)[return_cols]
        val_df   = pd.concat(val_dfs,  ignore_index=True)[return_cols] if val else pd.DataFrame(columns=return_cols)
        train_df = pd.concat(train_dfs,ignore_index=True)[return_cols]

        # Encode labels
        label_encoder = LabelEncoder()
        label_encoder.fit(df['labels'].unique())
        test_df['encoded_labels']  = label_encoder.transform(test_df['labels'])
        if not val_df.empty:
            val_df['encoded_labels'] = label_encoder.transform(val_df['labels'])
        train_df['encoded_labels'] = label_encoder.transform(train_df['labels'])

        # Now expand each split into a row-per-segment form using create_training_dataframe()
        # and passing the shift so we know how to compute start indices.
        X_train_df = self.create_training_dataframe(train_df, shift=shift)
        X_test_df  = self.create_training_dataframe(test_df,  shift=shift)

        if not val_df.empty:
            X_val_df = self.create_training_dataframe(val_df, shift=shift)
            return X_train_df, X_test_df, X_val_df
        else:
            return X_train_df, X_test_df
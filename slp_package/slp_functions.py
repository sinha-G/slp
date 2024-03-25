import pandas as pd

def one_hot_encode_flags(flag_enum, bitmask):
    """
    One-hot encode the set flags from the given bitmask using the specified IntFlag enumeration.

    :param flag_enum: An IntFlag enumeration class with defined flags.
    :param bitmask: The integer bitmask representing the set flags.
    :return: List of integers representing the one-hot encoded flags.
    """
    # Initialize a list to store the one-hot encoded values
    one_hot_encoded = [0] * len(flag_enum)

    # Iterate through the flags in the enumeration
    for i, flag in enumerate(flag_enum):
        if bitmask & flag:
            one_hot_encoded[i] = 1

    return one_hot_encoded

def create_merged_game_data_df(df_list, no_teams_2_player = True):
    '''
    Returns a merged dataframe comprised of the mang0, ranked, and/or public datasets.

    :df_list: List of Strings. Each String is one of:
                 * mango
                 * ranked
                 * public
    :no_teams_2_player: Boolean. Returns all data if False. Returns only 2 player games if True
    '''
    df = pd.DataFrame()

    # Either get all data or 2 player data depending on the value passed to function
    filter = 'no_teams_2_players' if no_teams_2_player else 'all_game_data'
    
    if 'mango' in df_list:
        df = pd.concat([df,
                        pd.read_parquet("C:\\Users\\jaspa\\Grant ML\\slp\\data\\mango_" + filter + "_df.parquet"),
                        ])
    if 'ranked' in df_list:
        df = pd.concat([df,
                        pd.read_parquet("C:\\Users\\jaspa\\Grant ML\\slp\\data\\ranked_" + filter + "_df_1.parquet"),
                        pd.read_parquet("C:\\Users\\jaspa\\Grant ML\\slp\\data\\ranked_" + filter + "_df_2.parquet"),
                        pd.read_parquet("C:\\Users\\jaspa\\Grant ML\\slp\\data\\ranked_" + filter + "_df_3.parquet"),
                        ])
    if 'public' in df_list:
        df = pd.concat([df, 
                        pd.read_parquet("C:\\Users\\jaspa\\Grant ML\\slp\\data\\public_" + filter + "_df_1.parquet"),
                        pd.read_parquet("C:\\Users\\jaspa\\Grant ML\\slp\\data\\public_" + filter + "_df_2.parquet"),
                        ])
    return df

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
    
    df['label'] = df[label_column]
    return df



def prepare_data_for_training(source_data, general_features, player_features, opposing_player_features, label_info):
    """
    Prepares data for training based on specified features and filters.

    :param source_data: List of sources to include in the data merge.
    :param general_features: Dictionary of general game features and their desired values.
    :param player_features: Dictionary of features for the player we are training on.
    :param opposing_player_features: Dictionary of features for the opposing player.
    :param label_info: Dictionary specifying the source and feature name for the label.
    :return: A pandas DataFrame with the prepared training data, containing only specified features and the label.
    """
    # Merge data from specified sources
    merged_df = create_merged_game_data_df(source_data)

    # Apply filters to general game data
    merged_df = apply_general_filters(merged_df, general_features)

    # Identify and label player and opposing player features
    merged_df = identify_and_label_players(merged_df, player_features, opposing_player_features)

    # Extract and set the label for training
    merged_df = extract_label(merged_df, label_info)

    # Define the order of columns to be selected
    general_feature_columns = list(general_features.keys())
    player_feature_columns = [f'player_{feature}' for feature in player_features.keys()]
    opposing_player_feature_columns = [f'opposing_player_{feature}' for feature in opposing_player_features.keys()]
    input_path_column = ['player_inputs_np_save_path']
    label_column = ['label']

    # Combine all columns in the desired order
    final_columns = general_feature_columns + player_feature_columns + opposing_player_feature_columns + input_path_column + label_column

    # Select only the specified columns from the DataFrame
    final_df = merged_df[final_columns]

    return final_df

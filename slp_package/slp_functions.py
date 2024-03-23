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
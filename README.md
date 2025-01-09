<h1> Slippi Data Analysis </h1>

<h2> Milestones </h2>

Some of the highlights from the work so far are:
* Fox-Sheik Binary Classification from inputs with >99% Accuracy
* Top-5 Character Multi Classification from inputs with >99% Accuracy
* Classify Fox's opponent from Fox's inputs Binary Classification with >65% Accuracy
* Classify Mango's Fox vs Null's Fox from inputs Binary Classification with >99% Accuracy

<h2> To Do: </h2>

A rough list of tasks, models, and ideas we hope to pursue:
* Autoencoder training
* RNN
* Transformers (built from scratch?)
* Add raw_analog_x feature if UCF
* Classify stage from inputs
* Predict winner from first t seconds
* Explainability

<h2> Research Questions </h2>

We're guided by the following questions:
* What correlates with winning? Positioning? L-cancelling? APM?






---

### `/extract_all_data/extract_all_data.ipynb`
This Jupyter Notebook demonstrates our **data extraction pipeline** for converting large batches of Slippi replay (`.slp`) files into structured datasets. We leverage a modified version of `py-slippi` to handle replays with minimal or missing metadata and store the resulting data in various formats (Parquet, Feather, and compressed NumPy arrays). The notebook also shows how we:

1. **Set up parallelized data processing** using [`joblib`](https://joblib.readthedocs.io/) and [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html) to handle thousands of `.slp` files efficiently.
2. **Use Python libraries** such as `numpy`, `pandas`, and `pyarrow` for data wrangling, serialization, and high-performance I/O.
3. **Organize the extracted data** into logical hierarchies and file formats that are easy to query later (i.e., storing replay metadata in CSV/Parquet and detailed frame data in separate files).
4. **Employ object-oriented methods and functional design** to modularize the code (e.g., separate functions for extracting replay metadata, saving frame data, handling errors, etc.).
5. **Handle large-scale data** tasks, including file path management, directory creation, and data integrity checks within big datasets.  
6. **Document our process** with descriptive markdown cells in the notebook, following best practices for readability and reproducibility.

---

### **slp_functions.py**
A utilities module containing core functions for merging, filtering, and labeling Slippi replay data. Key features include:

- **Merging multiple datasets** (Ranked, Public, Mang0) into a single DataFrame.  
- **Flexible filtering** based on game-level metadata (e.g., stage, number of players, outcome).  
- **Automatic column renaming** to distinguish the “target” player from the “opposing” player.  
- **Support for labeling** (e.g., by character or netplay code) to facilitate training tasks.  
- **Segment overlap calculations** for advanced time-series data augmentation.



## The **input_dataset** (input_dataset.py) module
Below is an improved guide to using the **`input_dataset`** class based on the code and context in your notebook. The guide includes an overview of how to instantiate and work with the `InputDataSet` object, retrieve labels, create train/test splits, and proceed to segment data if desired.

---

## 1. Overview

The `input_dataset` module (specifically the `InputDataSet` class) streamlines the process of filtering, labeling, and loading Melee input data. It:

1. Reads multiple sources of Melee data (e.g., `ranked`, `public`, `mango`) in the specified structure.
2. Applies user-defined filtering criteria (e.g., stage, number of players, character name).
3. Applies user-defined labeling criteria (e.g., which columns become the labels).
4. Generates ether Pandas DataFrames (`train`, `test`, and optionally `val`) with file paths (to compressed `.npy.gz` files) or numpy arrays containing the data itself. The Pandas DataFrame is used together with a custom dataset class to train PyTorch models on more data than fits in memory.

This class is particularly helpful if you have many .npy (or .npy.gz) input files from different data sources and want to unify, filter, and label them in a consistent way.

---

## 2. Quick Start

### 2.1 Installing and Importing

Make sure your environment can import the `input_dataset` module (usually part of the same directory or installed as a package in your environment). Then:

```python
from slp_package.input_dataset import InputDataSet
```

### 2.2 Creating an `InputDataSet` Instance

You will typically define a few dictionaries to control how the dataset is generated:

1. **`source_data`**: A list of subfolders or data sources (e.g., `['ranked','public','mango']`).
2. **`general_features`**: High-level criteria to filter games (e.g., which stage, how many players, whether the game was conclusive, etc.).
3. **`player_features`**: Criteria about the _player_ whose inputs you want to load (e.g., which character is used).
4. **`opposing_player_features`**: Criteria about the _opponent’s_ features (e.g., also a human player, or a specific netplay code).
5. **`label_info`**: Tells `InputDataSet` what the label source is (e.g., `'general'` or `'player'`) and which column in the final DataFrame is going to serve as the label.

Example:

```python
source_data = ['ranked', 'public', 'mango']

general_features = {
    'stage_name': ['FOUNTAIN_OF_DREAMS','FINAL_DESTINATION','BATTLEFIELD',
                   'YOSHIS_STORY','POKEMON_STADIUM','DREAMLAND'],
    'num_players': [2],
    'conclusive': [True]
}

player_features = {
    # e.g. 'character_name': ['FALCO'] 
    'character_name': ['FOX', 'CAPTAIN_FALCON', 'SHEIK', 'FALCO', 'MARTH'],
    'type_name': ['HUMAN']
}

opposing_player_features = {
    'type_name': ['HUMAN']
}

label_info = {
    'source': ['player'],         # can be 'general' or 'player'
    'feature': ['character_name'] # which column is the label
}
```

Then instantiate:

```python
dataset = InputDataSet(source_data,
                       general_features,
                       player_features,
                       opposing_player_features,
                       label_info)
```

### 2.3 Verifying the Dataset

Once created, your `dataset.dataset` property is a Pandas DataFrame with columns describing each game file (such as stage name, path to the numpy data, label, length, etc.):

```python
print(dataset.dataset['labels'].value_counts())
dataset.dataset.head()
```

You’ll see something like:

- The distribution of labels (e.g., `character_name` counts).
- A preview of columns such as `[stage_name, num_players, conclusive, player_character_name, ... , labels]`.

---

## 3. Train/Test/Val Splits

### 1. **`number_of_segments_per_game(...)` and `divide_games(...)` approach**  
This approach is designed to produce an *equal* number of segments **per label** for the final training, validation, and test sets. Because some labels may appear more frequently or have many more/longer games than others, the code estimates how many segments we can extract from each game (using a “shift” value) to match the global user-defined target (`num_segments_per_label`).  

- **Potential Overlap of Segments**  
  If the `shift` (the estimated distance between the start of consecutive segments) ends up being smaller than the segment length itself, some segments **will** overlap. This is because the system is trying to extract a target number of segments per label across all of that label’s games. These overlapping segments will show up across training, validation, and test sets in a consistent manner.  

- **Avoiding Leakage**  
  Data leakage is prevented because games themselves are *split* across train, test, and validation sets. We pick which games go to train, test, or validation; only then do we generate segments within those games. Hence, even if segments overlap within a given game, that game belongs to only one of the train/test/val sets.

- **Balanced Data**  
  Since each label aims to end up with the *same total number* of segments, this approach is well-suited if you require a balanced dataset by label (e.g., for classification tasks).  

- **Functions Involved**  
  1. `number_of_segments_per_game(segment_length, num_segments_per_label)`  
     - Determines how many segments (possibly overlapping) can be extracted from each game, so as to meet `num_segments_per_label` for each label.  
  2. `divide_games(...)`  
     - Splits entire games into train, test, and optionally validation sets based on the approximate number of segments for each label.  
  3. `create_training_dataframe(...)` or `create_training_numpy(...)`  
     - Generates the final set of segments from each game (now assigned to train/test/val), including start indices, segment lengths, and labels.  
     - With the “numpy” version, it loads and slices the actual game input arrays in parallel, returning the stacked array of segments and their labels.  

### 2. **`all_segments_train_test_split_dataframes(...)` approach**  
In contrast, the `divide_games_all_segments(...)` / `all_segments_train_test_split_dataframes(...)` methods create **all possible non-overlapping segments** from each game. The user can choose `segment_length` (in frames) and optionally specify a `proportion_of_segments` (< 1.0 if you want fewer than the total possible segments).  

- **No Overlapping Segments**  
  By definition, these methods take each game and cut it into consecutive chunks of `segment_length`, one after another, until the end of the game. Because each segment immediately follows the previous one, there is no overlap.  

- **Potential Class Imbalance**  
  This method does **not** try to produce an equal number of segments for each label. If one label has far more data/games than another, the final dataset may be heavily skewed toward that label.  

- **Avoiding Leakage**  
  As with the first approach, entire games are first assigned to train, test, or validation (in proportion to `test_ratio` and `val_ratio`). Only after that do we slice them into segments. Thus, no game's segments are split across multiple sets, and leakage is avoided.  

- **Functions Involved**  
  1. `divide_games_all_segments(segment_length, proportion_of_segments, ...)`  
     - Partitions entire games into train, test, val sets, but also computes `num_segments` for each game as `floor(length_of_game / segment_length) * proportion_of_segments`.  
     - Because it simply floors the counts, you get strictly non-overlapping consecutive segments of length `segment_length`.  
  2. `all_segments_train_test_split_dataframes(...)`  
     - A convenience wrapper that calls the above function, then uses `create_training_dataframe(...)` to produce the final listing of segments (including indices, labels, etc.).  
  3. *(Currently there is no built-in “NumPy version” analogous to `create_training_numpy(...)` for this approach.* However, you can still create a DataFrame of segments and manually load them into NumPy arrays later if needed.)*

### 3. **Choosing Which Approach to Use**  
- If you need *balanced* classes at the expense of some possible overlap, use the `number_of_segments_per_game(...)` + `divide_games(...)` approach.  
- If you need *all possible non-overlapping* segments and are comfortable with unbalanced classes (or you will handle rebalancing downstream), use the `all_segments_train_test_split_dataframes(...)` approach.

### 4. **Final Notes on Data Leakage**  
Regardless of the approach, **segments from the same game are never split** across train, test, or validation sets. This ensures that the same game cannot appear in more than one set, preventing leakage. Within a single game, segments can either overlap (if you’re using the first method and the computed shift < segment length) or remain strictly non-overlapping (if you’re using the “all segments” method).  

## 4. Next Steps for Model Training

In typical usage, you combine this dataset with a PyTorch `Dataset`/`DataLoader`. By only loading the `.npy.gz` files when needed, you can train on a large dataset that doesn’t fit in memory:

1. **Define** a custom PyTorch `Dataset` reading from `train_df` or `test_df`.  
2. **Use** `.npy.gz` file paths from `train_df['player_inputs_np_sub_path']`.  
3. Optionally apply transforms or shifts.  
4. Return tensors for each sample.

Example skeleton (similar to your notebook’s `TrainingDataset`):

```python
class TrainingDataset(Dataset):
    def __init__(self, df, transform=None):
        self.file_paths = df['player_inputs_np_sub_path'].to_numpy()
        self.encoded_labels = df['encoded_labels'].to_numpy()
        self.segment_start_index = df['segment_start_index'].to_numpy()
        self.segment_length = df['segment_length'].to_numpy()
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = '/workspace/melee_project_data/input_np/' + self.file_paths[idx].replace('\\\\','/')
        with gzip.open(path, 'rb') as f:
            segment = np.load(f)

        start_idx = self.segment_start_index[idx]
        end_idx = start_idx + self.segment_length[idx]
        segment = segment[:, int(start_idx):int(end_idx)]

        # (Optional) transform
        if self.transform:
            segment = your_transform_function(segment)

        return torch.from_numpy(segment).float()
```

Then in your training code:

```python
def prepare_data_loaders(train_df, test_df, batch_size, num_workers):
    train_dataset = TrainingDataset(train_df, transform=True)
    test_dataset  = TrainingDataset(test_df, transform=True)

    loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True,  num_workers=num_workers, pin_memory=True),
        'test' : DataLoader(test_dataset,  batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True),
    }
    return loaders
```


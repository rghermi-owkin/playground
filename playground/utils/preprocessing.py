from typing import Union, List, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    StandardScaler,
)

import torch
from torch.utils.data import default_collate


def filter_genes(
    df_exp,
    max_genes: int = 1_000,
):
    gene_list = list(df_exp.mad().sort_values(ascending=False)[:max_genes].index.values)
    df_exp = df_exp[gene_list + ["patient_id", "sample_id"]]
    return df_exp, gene_list


def log_scale(df, columns):
    df.loc[:, columns] = np.log(1. + df[columns])
    return df


def scale(df, columns, method="min_max"):
    if method == "min_max":
        scaler = MinMaxScaler()
    elif method == "mean_std":
        scaler = StandardScaler()
    elif method == "mean":
        scaler = StandardScaler(with_std=False)
    
    df.loc[:, columns] = scaler.fit_transform(df[columns])

    return df, scaler


def handle_nan_values(df):
    # Remove columns with only NaN values
    for c in df.columns:
        if df[c].isna().sum() == len(df):
            df[c] = ["Unknown"] * len(df)
            
    # Fill NaN values with median (cont.) or "Unknown" (cat.)
    for c in df.columns:
        try:
            df[c] = df[c].fillna(df[c].median())
        except:
            df[c] = df[c].fillna("Unknown")

    return df


def encode(df, columns):
    encoders = {}
    for c in columns:
        if c in df.columns:
            encoders[c] = LabelEncoder().fit(df[c])
            df[c] = encoders[c].transform(df[c]).astype(int)
    return df, encoders


def align_modalities(data, id: str = "patient_id"):
    # Take the intersection of ids
    ids_list = []

    for key, value in data.items():
        if value is not None:
            ids_list.append(value[id].unique())

    ids_list = list(set.intersection(*map(set, ids_list)))
    ids_list.sort()

    # Take the subset of ids, drop duplicates and sort by id
    for key in data.keys():
        if data[key] is not None:
            data[key] = data[key][(data[key][id].isin(ids_list))]
            data[key] = data[key].sort_values(by=[id])
            data[key] = data[key].drop_duplicates(subset=[id])

    return data, ids_list


def load_features(
    fpaths: Union[List, np.ndarray],
    n_tiles: Optional[int] = 1_000,
    shuffle: Optional[bool] = False,
) -> List:
    features = []

    for slide_features in tqdm(fpaths, total=len(fpaths)):
        if n_tiles is not None:
            # Using memory map to not load the entire np array when we
            # only want `n_tiles <= len(slide_features)` tiles' features
            if isinstance(slide_features, Path) or isinstance(slide_features, str):
                slide_features = np.load(slide_features, mmap_mode="r")

            indices = np.arange(len(slide_features))

            if shuffle:
                # We do not shuffle inplace using `np.random.shuffle(slide_features)` as this will load the whole
                # numpy array, removing all benefits of above `mmap_mode='r'`
                # Instead we shuffle indices and slice into the numpy array
                np.random.shuffle(indices)

            # Take the desired amount of tiles
            indices = indices[:n_tiles]

            # Indexing will make the array contiguous by loading it in RAM
            slide_features = slide_features[indices]

        else:
            # Load the whole np.array
            if isinstance(slide_features, Path) or isinstance(slide_features, str):
                slide_features = np.load(slide_features)

            if shuffle:
                # Shuffle inplace
                np.random.shuffle(slide_features)

        features.append(slide_features)

    return features


def filter_tiles(features):
    return features


def encode_features(features):
    return features


def multimodal_collate(
    batch,
    max_len: int = 1_000,
):
    # Collate ids
    ids = np.concatenate([[item[0]] for item in batch])
    
    # Collate histo
    sequences = [item[1]["HISTO"] for item in batch]
    
    trailing_dims = sequences[0].size()[1:]
    
    padded_dims = (len(sequences), max_len) + trailing_dims
    masks_dims = (len(sequences), max_len, 1)
    
    padded_sequences = sequences[0].data.new(*padded_dims).fill_(0.0)
    masks = torch.ones(*masks_dims, dtype=torch.bool)
    
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        padded_sequences[i, :length, ...] = tensor[:max_len, ...]
        masks[i, :length, ...] = False
        
    histo = padded_sequences
    histo_mask = masks
    
    # Collate exp
    exp = default_collate([item[1]["EXP"] for item in batch])
    
    return {
        "ID": ids,
        "HISTO": histo,
        "HISTO_MASK": histo_mask,
        "EXP": exp,
    }

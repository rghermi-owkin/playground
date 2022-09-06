import pandas as pd

from playground.utils import (
    load_features,
    filter_tiles,
    encode_features,
)

def preprocess_histo(
    df_histo: pd.DataFrame,
    n_tiles: int = None,
):
    # Load features
    #features = load_features(
    #    fpaths=df_histo.feature_path,
    #    n_tiles=n_tiles,
    #    shuffle=False,
    #)

    # Filter tiles
    # features = filter_tiles(features)

    # Encode features
    # features = encode_features(features)

    return df_histo.dropna()

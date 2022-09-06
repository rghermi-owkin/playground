from playground.utils import (
    handle_nan_values,
    encode,
    scale,
)


def preprocess_clin(df_clin):
    # Handle NaN values
    df_clin = handle_nan_values(df_clin)

    # Encode categorical variables
    categorical_vars = ["gender", "race", "stage", "grade"]
    df_clin, _ = encode(df_clin, categorical_vars)

    # Scale continuous variables
    continuous_vars = ["age"]
    df_clin, _ = scale(df_clin, continuous_vars)

    return df_clin

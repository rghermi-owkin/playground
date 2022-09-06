from playground.utils import (
    filter_genes,
    log_scale,
    scale,
)


def preprocess_exp(df_exp):
    # Filter genes
    df_exp, _ = filter_genes(df_exp)
    
    # Select expression columns
    exp_columns = list(df_exp.columns)
    exp_columns.remove("patient_id")
    exp_columns.remove("sample_id")

    # Log-scale
    df_exp = log_scale(df_exp, exp_columns)

    # Scale
    df_exp, _ = scale(df_exp, exp_columns)

    return df_exp

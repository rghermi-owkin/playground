import numpy as np
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
)

from playground.utils import (
    filter_genes,
    log_scale,
)


class ExpPreprocessor:

    def __init__(
        self,
        max_genes: int = 1_000,
        scaling_method: str = "min_max",
    ):
        self.max_genes = max_genes
        self.scaling_method = scaling_method

        if self.scaling_method == "min_max":
            self.scaler = MinMaxScaler()
        elif self.scaling_method == "mean_std":
            self.scaler = StandardScaler()
        elif self.scaling_method == "mean":
            self.scaler = StandardScaler(with_std=False)

        self.gene_list = None

    def fit_transform(self, df_exp):
        self.gene_list = list(df_exp.columns)
        self.gene_list.remove("patient_id")
        self.gene_list.remove("sample_id")

        # Filter genes
        df_exp, gene_list = filter_genes(
            df_exp,
            max_genes=self.max_genes,
        )

        self.gene_list = gene_list

        # Log-scale
        df_exp.loc[:, self.gene_list] = log_scale(df_exp[self.gene_list])

        # Scale
        df_exp.loc[:, self.gene_list] = self.scaler.fit_transform(df_exp[self.gene_list])

        return df_exp

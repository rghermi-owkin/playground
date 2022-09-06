from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    StandardScaler,
)

from playground.utils import (
    handle_nan_values,
)


class ClinPreprocessor:
    
    def __init__(
        self,
        scaling_method: str = "mean_std",
    ):
        self.scaling_method = scaling_method

    def fit_transform(self, df_clin):
        # Handle NaN values
        df_clin = handle_nan_values(df_clin)

        # Encode categorical variables
        categorical_vars = ["gender", "race", "stage", "grade"]
        self.encoders = {c: LabelEncoder().fit(df_clin[c]) for c in categorical_vars}
        for c in categorical_vars:
            df_clin.loc[:, c] = self.encoders[c].transform(df_clin[c])

        # Scale continuous variables
        continuous_vars = ["age"]
        if self.scaling_method == "min_max":
            self.scalers = {c: MinMaxScaler().fit(df_clin[c]) for c in continuous_vars}
        elif self.scaling_method == "mean_std":
            self.scalers = {c: StandardScaler().fit(df_clin[c]) for c in continuous_vars}
        elif self.scaling_method == "mean":
            self.scalers = {c: StandardScaler(with_std=False).fit(df_clin[c]) for c in continuous_vars}
        for c in continuous_vars:
            df_clin.loc[:, c] = self.scalers[c].transform(df_clin[c])

        return df_clin

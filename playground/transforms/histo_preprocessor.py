class HistoPreprocessor:

    def __init__(self):
        pass

    def fit_transform(self, df_histo):
        return df_histo.dropna()

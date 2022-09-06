from hydra.utils import instantiate

from .base_trainer import BaseTrainer


class SklearnTrainer(BaseTrainer):

    def __init__(
        self,
        model_cfg,
        metric_cfg,
    ):
        super().__init__()
        
        self.model_cfg = model_cfg
        self.metric_cfg = metric_cfg

        self.model = None
        self.metric = None
    
    def train(self, X_train, y_train, X_valid, y_valid):
        # Model, criterion, metric
        self.model = instantiate(self.model_cfg)
        self.metric = instantiate(self.metric_cfg)

        # Training
        self.model.fit(X_train, y_train)

        # Prediction
        y_train_preds = self.model.predict(X_train)
        y_valid_preds = self.model.predict(X_valid)

        # Compute metrics
        train_metric = self.metric(y_train, y_train_preds)
        valid_metric = self.metric(y_valid, y_valid_preds)

        return train_metric, valid_metric

    def evaluate(self, X, y):
        # Prediction
        y_preds = self.model.predict(X)

        # Compute metrics
        metric = self.metric(y, y_preds)

        return metric

    def predict(self, X, y):
        # Prediction
        y_preds = self.model.predict(X)

        return y_preds, y

from hydra.utils import instantiate
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam

from omics_rpz.data import OmicsDataset
from omics_rpz.utils import (
    train_step,
    eval_step,
)

from .base_trainer import BaseTrainer


class TorchTrainer(BaseTrainer):

    def __init__(
        self,
        model_cfg,
        loss_cfg,
        metric_cfg,
        batch_size: int = 16,
        learning_rate: float = 1.0e-3,
        weight_decay: float = 0.0,
        device: str = "cpu",
    ):
        super().__init__()
        
        self.model_cfg = model_cfg
        self.loss_cfg = loss_cfg
        self.metric_cfg = metric_cfg

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device

        self.model = None
        self.criterion = None
        self.metric = None

        self.train_losses = None
        self.valid_losses = None
        self.train_metrics = None
        self.valid_metrics = None
    
    def train(self, X_train, y_train, X_valid, y_valid):
        # Datasets
        train_dataset = OmicsDataset(
            X=X_train, y=y_train)
        valid_dataset = OmicsDataset(
            X=X_train, y=y_train)

        # Dataloaders
        train_dataloader = DataLoader(
            dataset=train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            drop_last=True,
        )
        valid_dataloader = DataLoader(
            dataset=valid_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            drop_last=False,
        )

        # Model, criterion, metric
        self.model = instantiate(self.model_cfg).to(self.device)
        self.criterion = instantiate(self.loss_cfg).to(self.device)
        self.metric = instantiate(self.metric_cfg)

        # Optimizer
        trainable_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.optimizer = Adam(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Training...
        train_losses, valid_losses = [], []
        train_metrics, valid_metrics = [], []
        for epoch in tqdm(range(num_epochs), total=num_epochs):
            # Training step
            train_loss, train_epoch_logits, train_epoch_labels = train_step(
                model=self.model,
                train_dataloader=train_dataloader,
                criterion=self.criterion,
                optimizer=self.optimizer,
                device=self.device,
            )

            # Validation step
            valid_loss, valid_epoch_logits, valid_epoch_labels = eval_step(
                model=self.model,
                valid_dataloader=valid_dataloader,
                criterion=self.criterion,
                device=self.device,
            )

            # Compute metrics
            train_metric = self.metric(train_epoch_labels, train_epoch_logits)
            valid_metric = self.metric(valid_epoch_labels, valid_epoch_logits)

            # Save logs
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            train_metrics.append(train_metric)
            valid_metrics.append(valid_metric)

        self.train_losses = train_losses
        self.valid_losses = valid_losses
        self.train_metrics = train_metrics
        self.valid_metrics = valid_metrics

        return train_metrics[-1], valid_metrics[-1]

    def evaluate(self, X, y):
        # Dataset & Dataloader
        dataset = OmicsDataset(X=X, y=y)
        dataloader = DataLoader(
            dataset=dataset,
            shuffle=False,
            batch_size=self.batch_size,
            drop_last=False,
        )

        # Prediction step
        _, epoch_logits, epoch_labels = eval_step(
            model=self.model,
            valid_dataloader=dataloader,
            criterion=self.criterion,
            device=self.device,
        )

        return self.metric(epoch_labels, epoch_logits)

    def predict(self, X, y):
        # Dataset & Dataloader
        dataset = OmicsDataset(X=X, y=y)
        dataloader = DataLoader(
            dataset=dataset,
            shuffle=False,
            batch_size=self.batch_size,
            drop_last=False,
        )

        # Prediction step
        _, epoch_logits, epoch_labels = eval_step(
            model=self.model,
            valid_dataloader=dataloader,
            criterion=self.criterion,
            device=self.device,
        )

        return epoch_logits, epoch_labels

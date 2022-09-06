from typing import Tuple
import numpy as np
import torch


def train_step(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Training loop.
    Parameters
    ----------
    model : torch.nn.Module
        The model to fit.
    train_dataloader : torch.utils.data.DataLoader
        Dataloader yielding batches of training data (features, masks, labels).
    criterion : torch.nn.Module
        Loss criterion.
    optimizer: torch.optim.Optimizer
        Optimization algorithm.
    device: str
        Device used for training/evaluation.
    Returns
    -------
    _epoch_loss: np.ndarray
        List of losses computed on training data.
    _epoch_logits: np.ndarray
        List of logits computed on training data.
    _epoch_labels: np.ndarray
        List of labels from training data.
    """

    model.train()

    _epoch_loss, _epoch_logits, _epoch_labels = [], [], []

    for i, batch in enumerate(train_dataloader):
        
        # Get data
        features, labels = batch

        # Put on device
        features = features.to(device)
        labels = labels.to(device)
        
        # Compute logits and loss
        logits = model(features)
        loss = criterion(logits, labels)
        
        # Run backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stack logits & labels to compute epoch metrics
        _epoch_loss.append(loss.detach().cpu().numpy())
        _epoch_logits.append(logits.detach())
        _epoch_labels.append(labels.detach())

    _epoch_loss = np.mean(_epoch_loss)
    _epoch_logits = torch.cat(_epoch_logits, dim=0).cpu().numpy()
    _epoch_labels = torch.cat(_epoch_labels, dim=0).cpu().numpy()

    return _epoch_loss, _epoch_logits, _epoch_labels


def eval_step(
    model: torch.nn.Module,
    valid_dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluation loop.
    Parameters
    ----------
    model : torch.nn.Module
        The model to evaluate.
    valid_dataloader : torch.utils.data.DataLoader
        Dataloader yielding batches of validation data (features, masks, labels).
    criterion : torch.nn.Module
        Loss criterion.
    device: str
        Device used for training/evaluation.
    Returns
    -------
    _epoch_loss: np.ndarray
        List of losses computed on validation data.
    _epoch_logits: np.ndarray
        List of logits computed on validation data.
    _epoch_labels: np.ndarray
        List of labels from validation data.
    """

    model.eval()

    with torch.no_grad():

        _epoch_loss, _epoch_logits, _epoch_labels = [], [], []

        for i, batch in enumerate(valid_dataloader):

            # Get data
            features, labels = batch

            # Put on device
            features = features.to(device)
            labels = labels.to(device)

            # Compute logits and loss
            logits = model(features)
            loss = criterion(logits, labels)

            # Stack logits & labels to compute epoch metrics
            _epoch_loss.append(loss.detach().cpu().numpy())
            _epoch_logits.append(logits.detach())
            _epoch_labels.append(labels.detach())

    _epoch_loss = np.mean(_epoch_loss)
    _epoch_logits = torch.cat(_epoch_logits, dim=0).cpu().numpy()
    _epoch_labels = torch.cat(_epoch_labels, dim=0).cpu().numpy()

    return _epoch_loss, _epoch_logits, _epoch_labels

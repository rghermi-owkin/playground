from typing import Tuple
import numpy as np
from tqdm import tqdm
import torch


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    feature_keys,
    label_key,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.train()

    _epoch_loss, _epoch_logits, _epoch_labels = [], [], []

    for i, batch in enumerate(dataloader):

        # Put on device
        for key, value in batch.items():
            if key == "ID":
                continue
            batch[key] = value.to(device)
                
        # Define features and labels
        features = (batch[k] for k in feature_keys)
        labels = batch[label_key]
        
        # Compute logits and loss
        logits = model(*features)
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
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: str,
    feature_keys,
    label_key,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    
    with torch.no_grad():
        _epoch_loss, _epoch_logits, _epoch_labels = [], [], []

        for i, batch in enumerate(dataloader):

            # Put on device
            for key, value in batch.items():
                if key == "ID":
                    continue
                batch[key] = value.to(device)

            # Define features and labels
            features = (batch[k] for k in feature_keys)
            labels = batch[label_key]

            # Compute logits and loss
            logits = model(*features)
            loss = criterion(logits, labels)

            # Stack logits & labels to compute epoch metrics
            _epoch_loss.append(loss.detach().cpu().numpy())
            _epoch_logits.append(logits.detach())
            _epoch_labels.append(labels.detach())

        _epoch_loss = np.mean(_epoch_loss)
        _epoch_logits = torch.cat(_epoch_logits, dim=0).cpu().numpy()
        _epoch_labels = torch.cat(_epoch_labels, dim=0).cpu().numpy()

    return _epoch_loss, _epoch_logits, _epoch_labels

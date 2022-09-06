import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
)

from omics_rpz.utils import (
    sigmoid,
    softmax,
)


def compute_binary_auc(labels, logits):
    preds = sigmoid(logits)
    try:
        return roc_auc_score(labels, preds)
    except AssertionError:
        return 0.5
    except ValueError:
        return np.nan


def compute_accuracy(labels, logits):
    preds = np.expand_dims(np.argmax(logits, axis=1), axis=1)
    return accuracy_score(labels, preds)

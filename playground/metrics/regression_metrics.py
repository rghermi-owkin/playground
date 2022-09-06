from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def compute_root_mean_squared_error(labels, logits):
    return mean_squared_error(labels, logits, squared=False)


def compute_mean_absolute_error(labels, logits):
    return mean_absolute_error(labels, logits)


def compute_r2_score(labels, logits):
    return r2_score(labels, logits)

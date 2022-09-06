import numpy as np


def sigmoid(x: np.ndarray):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray):
    z = x - x.max(axis=1, keepdims=True)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    return numerator / denominator

import numpy as np
from lifelines.utils import concordance_index


def compute_cindex(labels, logits):
    times, events = np.abs(labels), 1 * (labels > 0)
    try:
        return concordance_index(times, -logits, events)
    except AssertionError:
        return 0.5

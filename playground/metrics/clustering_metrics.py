from sklearn.metrics import adjusted_rand_score


def compute_adjusted_rand_score(labels_true, labels_pred):
    return adjusted_rand_score(labels_true, labels_pred)

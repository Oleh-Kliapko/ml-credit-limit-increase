# threshold_optimization.py
# Пошук оптимального порогу для бінаризації ймовірностей класу за допомогою метрики F-beta

import numpy as np
from sklearn.metrics import fbeta_score


def find_best_threshold(y_true, y_proba, beta=2):
    thresholds = np.arange(0.05, 0.95, 0.01)

    best_threshold = 0.5
    best_score = 0

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        score = fbeta_score(y_true, y_pred, beta=beta)

        if score > best_score:
            best_score = score
            best_threshold = t

    return best_threshold, best_score

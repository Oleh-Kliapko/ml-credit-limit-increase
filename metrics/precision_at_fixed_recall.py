# precision_at_fixed_recall.py
# Пошук порогу для бінаризації ймовірностей класу,
# який максимізує точність (Precision) при фіксованому значенні повноти (Recall)

import numpy as np
from sklearn.metrics import precision_score, recall_score


def precision_at_fixed_recall(y_true, y_proba, target_recall=0.8, steps=1000):
    thresholds = np.linspace(0, 1, steps)
    best = {
        "threshold": None,
        "precision": 0,
        "recall": 0
    }

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        recall = recall_score(y_true, y_pred)

        if recall >= target_recall:
            precision = precision_score(y_true, y_pred, zero_division=0)

            if precision > best["precision"]:
                best.update({
                    "threshold": t,
                    "precision": precision,
                    "recall": recall
                })

    return best


def precision_recall_comment(precision):
    if precision < 0.4:
        return "❌ Низька точність: багато хибних тривог\n"
    if precision < 0.6:
        return "⚠️ Прийнятна точність для консервативного ризик-контролю\n"
    if precision >= 0.6:
        return "🔥 Висока точність при контрольованому ризику\n"

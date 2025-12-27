# models/logreg.py

from sklearn.linear_model import LogisticRegression


def get_logreg_model(_y=None):  # _y=None для сумісності з іншими моделями
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear"
    )

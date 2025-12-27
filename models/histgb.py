# models/histgb.py

from sklearn.ensemble import HistGradientBoostingClassifier


def get_histgb_model(_y=None):
    return HistGradientBoostingClassifier(
        max_iter=500,
        max_depth=None,
        learning_rate=0.05,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
        class_weight='balanced',
        scoring='roc_auc',
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.1
    )

# models/xgboost.py

from xgboost import XGBClassifier


def get_xgboost_model(_y=None):
    return XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        scale_pos_weight=10,
        n_jobs=-1,
        tree_method="hist",
        random_state=42
    )

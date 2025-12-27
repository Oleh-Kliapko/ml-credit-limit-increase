# models/lightgbm.py

import lightgbm as lgb


def get_lightgbm_model(_y=None):
    return lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=-1,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
        class_weight=None,
        scale_pos_weight=10,
        verbose=-1
    )

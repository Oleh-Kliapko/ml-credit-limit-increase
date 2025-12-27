# feature_engineering.py
# Проводимо інженерію ознак (фінальні 10 фіч)

import logging
import numpy as np
import pandas as pd

from .validate_columns import validate_columns

logger = logging.getLogger(__name__)


FINAL_FEATURES = [
    "max_outstand_1m",
    "max_outstand_3m",
    "sum_outstand_all",
    "delta_outstand_3m_1m",
    "growth_ratio_outstand",
    "cnt_active_products",
    "avg_openamount",
    "max_pd",
    "no_credit_history_flag",
    "std_outstand_3m",
]


def feature_engineering(df):
    logger.info("\n=== Інженерія ознак стартувала ===")
    logger.info("Початкова форма: %s", df.shape)
    validate_columns(df)

    X = pd.DataFrame(index=df.index)

    # 1–3
    X["max_outstand_1m"] = df["CredHist_MaxCC_out_1m"]
    X["max_outstand_3m"] = df["CredHist_MaxCC_out_3m"]
    X["sum_outstand_all"] = df["CredHist_Sum_cur_out"]

    # 4
    X["delta_outstand_3m_1m"] = (
        X["max_outstand_3m"] - X["max_outstand_1m"]
    )

    # 5
    X["growth_ratio_outstand"] = (
        X["max_outstand_3m"] / (X["max_outstand_1m"] + 1)
    )

    # 6
    X["cnt_active_products"] = df[
        "CredHist_count_all_loans_open_now"
    ]

    # 7
    X["avg_openamount"] = (
        X["sum_outstand_all"] /
        (X["cnt_active_products"] + 1)
    )

    # 8
    X["max_pd"] = df["CredHist_Max_dpd_ever"]

    # 9
    X["no_credit_history_flag"] = (
        (df["CredHist_IsDataAvailable"] == 0).astype(int)
    )

    # 10
    X["std_outstand_3m"] = X[
        ["max_outstand_1m", "max_outstand_3m"]
    ].std(axis=1)

    # Очистка від inf та NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    logger.info("Фінальні ознаки: %s", FINAL_FEATURES)
    logger.info("Фінальна форма: %s", X.shape)
    logger.info("=== Інженерія ознак завершена ===\n\n")

    return X[FINAL_FEATURES]

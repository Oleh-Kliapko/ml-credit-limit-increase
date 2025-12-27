# missing_analysis.py
# Рахуємо пропуски для прийняття рішення

import pandas as pd


def analyze_missing(df: pd.DataFrame) -> pd.DataFrame:
    missing_count = df.isna().sum()
    missing_percent = missing_count / len(df) * 100

    miss_df = (
        pd.DataFrame({
            "missing_count": missing_count,
            "missing_percent": missing_percent
        })
        .query("missing_count > 0")
        .sort_values("missing_percent", ascending=False)
    )

    print("\n\n========= Аналіз пропущених значень =========")

    if miss_df.empty:
        print("Пропущених значень немає ✅\n\n")
    else:
        print(miss_df)

    return miss_df

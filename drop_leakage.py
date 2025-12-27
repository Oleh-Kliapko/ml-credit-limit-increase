# drop_leakage.py
# Видаляємо колонки для запобігання витоку даних

import pandas as pd

def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    drop_cols = ["clienthash", "limit_pre", "limit_in30d"]
    existing = [c for c in drop_cols if c in df.columns]

    df = df.drop(columns=existing)

    print("\n\n========= Видалення leakage колонок =========")
    print(f"Видалені колонки: {existing}")
    print(f"Форма після видалення: {df.shape}")

    return df

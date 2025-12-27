# handle_missing_values.py
# Обробка пропусків на основі аналізу

import pandas as pd


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    print("\n========= Обробка пропущених значень =========")

    # 1. Drop технічних колонок
    tech_cols = ["id_order"]
    drop_cols = [c for c in tech_cols if c in df.columns]

    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"Видалені технічні колонки: {drop_cols}")

    # 2. Категоріальні колонки
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in cat_cols:
        missing = df[col].isna().sum()
        if missing > 0:
            df[col] = df[col].fillna("Missing")

    print("Категоріальні колонки: заповнено 'Missing'")

    # 3. Числові колонки
    num_cols = df.select_dtypes(include=["number"]).columns

    for col in num_cols:
        missing = df[col].isna().sum()
        if missing > 0:
            median = df[col].median()
            df[col] = df[col].fillna(median)

    print("Числові колонки: заповнено медіаною")

    date_cols = ["Appl_Application_date", "upload_date"]

    # 4. Колонки з датою
    existing_dates = [c for c in date_cols if c in df.columns]
    if existing_dates:
        df = df.drop(columns=existing_dates)
        print(f"Видалені колонки з датою: {existing_dates}")

    print(
        f"Обробка пропущених значень завершена - Max missing: {df.isna().sum().max()}\n\n")

    return df

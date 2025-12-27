# create_target.py
# Створення таргету за абсолютним приростом ліміту

import pandas as pd


def create_target_absolute(df: pd.DataFrame, threshold: float = 5000) -> pd.DataFrame:
    df = df.copy()

    print("\n\n========= Формування таргету (абсолютний) =========")
    print(f"Поріг зростання ліміту: {threshold}")

    df["target"] = (df["limit_in30d"] >= threshold).astype("int8")

    total = len(df)
    positive = df["target"].sum()
    negative = total - positive

    print(f"Всього рядків : {total}")
    print(
        f"Клас 1 (limit ≥ {threshold}): {positive} ({positive / total * 100:.2f}%)")
    print(f"Клас 0              : {negative} ({negative / total * 100:.2f}%)")

    print("Таргет сформовано.\n\n")

    return df

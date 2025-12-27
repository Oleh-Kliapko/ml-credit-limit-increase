# optimize_numerical.py
# Оптимізація числових типів даних

import pandas as pd
import numpy as np


def optimize_numerical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    print("\n\n========= Оптимізація ЧИСЛОВИХ типів даних =========")
    mem_before = df.memory_usage(deep=True).sum()
    print(f"Пам’ять до оптимізації: {mem_before / 1024**2:.2f} MB\n")

    # ---- Логування змін типів ----
    numeric_logs = []

    # 1. Оптимізація числових колонок
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in num_cols:
        old_type = df[col].dtype
        old_memory = df[col].memory_usage(deep=True)

        col_min = df[col].min()
        col_max = df[col].max()

        # Вибір оптимального типу
        if pd.api.types.is_integer_dtype(df[col]):
            if col_min >= 0:   # unsigned int
                if col_max < 255:
                    new_type = "uint8"
                elif col_max < 65535:
                    new_type = "uint16"
                elif col_max < 4294967295:
                    new_type = "uint32"
                else:
                    new_type = "uint64"
            else:  # signed int
                if np.iinfo("int8").min <= col_min <= np.iinfo("int8").max:
                    new_type = "int8"
                elif np.iinfo("int16").min <= col_min <= np.iinfo("int16").max:
                    new_type = "int16"
                elif np.iinfo("int32").min <= col_min <= np.iinfo("int32").max:
                    new_type = "int32"
                else:
                    new_type = "int64"
        else:
            new_type = "float32"

        # Конвертація
        df[col] = df[col].astype(new_type)

        new_memory = df[col].memory_usage(deep=True)
        saved = old_memory - new_memory
        saved_pct = (saved / old_memory * 100) if old_memory > 0 else 0

        numeric_logs.append(
            f"Колонка '{col}': {old_type} → {new_type} "
            f"(економія {saved / 1024**2:.4f} MB, {saved_pct:.1f}%)"
        )

    # Памʼять після оптимізації
    mem_after = df.memory_usage(deep=True).sum()
    print(
        f"\nПам’ять після оптимізації ЧИСЛОВИХ типів даних: {mem_after / 1024**2:.2f} MB")
    print(
        f"Загальна економія після оптимізації ЧИСЛОВИХ типів даних: {(mem_before - mem_after) / 1024**2:.2f} MB")
    print("Оптимізація ЧИСЛОВИХ типів даних завершена.\n\n")

    return df

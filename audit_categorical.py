# audit_categorical.py
# Аудит категоріальних ознак (БЕЗ змін датасету)

import pandas as pd

def audit_categorical(df: pd.DataFrame) -> pd.DataFrame:
    print("\n========= Аудит категоріальних колонок =========")

    cat_cols = df.select_dtypes(include=["object"]).columns
    total_rows = len(df)

    audit_rows = []

    for col in cat_cols:
        unique_cnt = df[col].nunique(dropna=True)
        unique_pct = unique_cnt / total_rows * 100

        # Інтерпретація
        if unique_pct < 5:
            decision = "category / one-hot"
        elif unique_pct < 20:
            decision = "target / frequency encoding"
        elif unique_pct < 80:
            decision = "обережно: можлива агрегація"
        else:
            decision = "ймовірно ID → drop"

        audit_rows.append({
            "column": col,
            "unique_count": unique_cnt,
            "unique_percent": round(unique_pct, 2),
            "suggested_action": decision
        })

    audit_df = pd.DataFrame(audit_rows)\
        .sort_values("unique_percent")\
        .reset_index(drop=True)

    print(audit_df)

    print("\nПояснення:")
    print("- <5%     → стабільні категорії - category / one-hot")
    print("- 5–20%   → можна кодувати статистично - target / frequency encoding")
    print("- 20–80%  → обережно: можлива агрегація - потребує аналізу")
    print("- >80%    → майже унікальні (ID, шум) - ймовірно ID → drop")

    return audit_df

# correlation_analysis.py
# Аналізуємо кореляцію числових фіч з таргетом

import pandas as pd


def analyze_correlation_with_target(
    df: pd.DataFrame,
    target_col: str = "target",
    top_n: int = 20,
    min_abs_corr: float = 0.01,
):
    df = df.copy()

    print("\n========= Аналіз кореляції з таргетом =========")

    # 1. Вибір числових колонок
    numeric_cols = df.select_dtypes(include=["int", "float"]).columns.tolist()

    if target_col not in numeric_cols:
        df[target_col] = df[target_col].astype(int)
        numeric_cols.append(target_col)

    numeric_cols.remove(target_col)

    print(f"Кількість числових фіч: {len(numeric_cols)}")

    # 2. Кореляція з таргетом
    corr = (
        df[numeric_cols]
        .corrwith(df[target_col])
        .reset_index()
        .rename(columns={"index": "feature", 0: "correlation"})
    )

    corr["abs_correlation"] = corr["correlation"].abs()
    corr = corr.sort_values("abs_correlation", ascending=False)

    # 3. Вивід ТОП фіч
    print(f"\nТОП-{top_n} фіч за |correlation|:")
    print(corr.head(top_n))

    # 4. Зведена статистика
    weak_corr = (corr["abs_correlation"] < min_abs_corr).sum()
    strong_corr = (corr["abs_correlation"] >= min_abs_corr).sum()

    print("\n========= Зведення по кореляції =========")
    print(f"Фіч з |corr| < {min_abs_corr}: {weak_corr}")
    print(f"Фіч з |corr| ≥ {min_abs_corr}: {strong_corr}")

    return corr

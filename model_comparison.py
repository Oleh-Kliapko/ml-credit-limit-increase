# model_comparison.py
# Створення порівняльного звіту по моделях (таблиця CSV + PNG) для презентацій

import pandas as pd
import matplotlib.pyplot as plt
import logging

from .config import PROCESSED_DATA_PATH

logger = logging.getLogger(__name__)


def business_score(row):
    if row["auc"] < 0.55:
        return "❌ Не придатна"
    if row["precision_at_recall"] < 0.05:
        return "⚠️ Слабкий сигнал"
    if row["precision_at_recall"] < 0.15:
        return "✅ Можна для review"
    return "🔥 Сильна модель"


def build_model_comparison_report(model_results: list, save_png: bool = True):
    if not model_results:
        logger.warning("Немає результатів моделей для порівняння.")
        return pd.DataFrame()

    df = pd.DataFrame(model_results)

    df["business_score"] = df.apply(business_score, axis=1)

    df = df[
        [
            "model",
            "auc",
            "gini",
            "precision",
            "recall",
            "f1",
            "f2",
            "opt_threshold",
            "precision_at_recall",
            "business_score"
        ]
    ]

    csv_path = f"{PROCESSED_DATA_PATH}/model_comparison.csv"
    df.to_csv(csv_path, index=False)
    logger.info("\n\n📊 Порівняльна таблиця збережена: %s", csv_path)

    # ===== PNG =====
    if save_png:
        _fig, ax = plt.subplots(figsize=(20, 4 + len(df)))
        ax.axis("off")

        table = ax.table(
            cellText=df.round(4).values,
            colLabels=df.columns,
            loc="center",
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        png_path = f"{PROCESSED_DATA_PATH}/model_comparison.png"
        plt.savefig(png_path, bbox_inches="tight")
        plt.close()

        logger.info("🖼 Порівняльна таблиця (PNG) збережена: %s", png_path)

    return df

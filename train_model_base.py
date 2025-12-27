# train_model_base.py
# Базовий компонент для тренування будь-якої моделі
# Включає спільну логіку: розподіл даних, тренування, метрики, збереження, графіки

import os
import joblib
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_curve,
    precision_recall_curve,
    auc
)

from .metrics.threshold_optimization import find_best_threshold
from .config import PROCESSED_DATA_PATH
from .metrics_report import build_metrics_report, final_conclusion, next_step_recommendation
from .metrics.precision_at_fixed_recall import (
    precision_at_fixed_recall,
    precision_recall_comment
)


logger = logging.getLogger(__name__)


def train_model_base(X, y, model, model_name, test_size=0.3, random_state=42):
    logger.info("\n=== Побудова моделі: %s ===", model_name)
    logger.info("Фіч: %s", list(X.columns))
    logger.info("Shape X: %s, y: %s", X.shape, y.shape)

    # Розділення на тренування та тестування
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logger.info("Train: %s, Test: %s", X_train.shape, X_test.shape)

    # Тренування моделі
    model.fit(X_train, y_train)

    # Прогнози
    y_proba = model.predict_proba(X_test)[:, 1]

    # Пошук оптимального порогу для бінаризації ймовірностей класу за допомогою метрики F2
    best_threshold, best_f2 = find_best_threshold(y_test, y_proba, beta=2)
    logger.info(
        "🎯 Оптимальний поріг = %.2f (max F2 = %.4f)",
        best_threshold, best_f2
    )
    y_pred_opt = (y_proba >= best_threshold).astype(int)

    # Precision при фіксованому Recall = 0.8
    # Бізнес-метрика "Якщо ми хочемо зловити не менше 80% ризикових клієнтів,
    # то наскільки чистим буде список, який ми передамо в роботу?"
    result = precision_at_fixed_recall(
        y_true=y_test,
        y_proba=y_proba,
        target_recall=0.8
    )
    comment = precision_recall_comment(result["precision"])
    print("\n=== Precision @ Fixed Recall ===\n")
    print("Target recall:         0.80")
    print(f"Optimal threshold:    {result['threshold']:.3f}")
    print(f"Recall achieved:      {result['recall']:.3f}")
    print(f"Precision achieved:   {result['precision']:.3f}")
    print(f"Conclusion:           {comment}")

    # Метрики
    auc_metric = roc_auc_score(y_test, y_proba)
    gini = 2 * auc_metric - 1
    precision = precision_score(y_test, y_pred_opt)
    recall = recall_score(y_test, y_pred_opt)
    f1 = f1_score(y_test, y_pred_opt)
    f2 = fbeta_score(y_test, y_pred_opt, beta=2)

    metrics = {
        "auc": auc_metric,
        "gini": gini,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
    }

    build_metrics_report(metrics)

    print(f"\n=== 🧠 Фінальний висновок ({model_name}) ===")
    print(final_conclusion(metrics))

    print(f"\n=== 🛠 Рекомендовані наступні кроки ({model_name}) ===")
    print(next_step_recommendation(metrics))

    # Збереження моделі
    model_path = os.path.join(PROCESSED_DATA_PATH, f"{model_name}.pkl")
    joblib.dump(model, model_path)
    logger.info("\nМодель %s збережена: %s", model_name, model_path)

    # Графіки ROC + PR
    _, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC крива
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title(f'ROC Curve - {model_name}')
    axes[0].legend(loc="lower right")
    axes[0].grid(True)

    # Precision-Recall крива
    pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(pr_recall, pr_precision)
    axes[1].plot(pr_recall, pr_precision, color='green', lw=2,
                 label=f'PR curve (AUC = {pr_auc:.3f})')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title(f'Precision-Recall Curve - {model_name}')
    axes[1].legend(loc="upper right")
    axes[1].grid(True)

    plt.tight_layout()

    # Збереження графіків
    plots_path = os.path.join(
        PROCESSED_DATA_PATH, f"{model_name}_baseline_curves.png")
    plt.savefig(plots_path)
    logger.info("\nГрафіки %s збережено: %s", model_name, plots_path)
    plt.close()

    return {
        "model": model_name,
        "auc": auc_metric,
        "gini": gini,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "opt_threshold": best_threshold,
        "precision_at_recall": result["precision"],
    }

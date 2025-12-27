import logging
from importlib import import_module

from .model_comparison import build_model_comparison_report
from .load_data import load_dataset
from .optimize_numerical import optimize_numerical
from .audit_categorical import audit_categorical
from .limit_stats import limit_basic_stats
from .create_target import create_target_absolute
from .drop_leakage import drop_leakage_columns
from .missing_analysis import analyze_missing
from .handle_missing_values import handle_missing_values
from .correlation_analysis import analyze_correlation_with_target
from .feature_engineering import feature_engineering
from .model_selector import select_model
from .train_model_base import train_model_base

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def main():
    # Считування і оптимізація датасету
    df = load_dataset()
    df = optimize_numerical(df)

    # Аудит даних для прийняття рішення по таргету
    audit_categorical(df)
    limit_basic_stats(df)
    df = create_target_absolute(df, threshold=5000)

    # Видалення колонок, які можуть призвести до некоректності роботи моделі
    df = drop_leakage_columns(df)

    # Аналіз пропущених даних та їх обробка
    analyze_missing(df)
    df = handle_missing_values(df)

    # Аналіз пропущених даних та їх обробка
    analyze_correlation_with_target(df, target_col="target", top_n=20)

    # Інженерія ознак
    X = feature_engineering(df)
    y = df["target"]

    # Вибір моделі(ей)
    selected_models = select_model()

    # Список для накопичення результатів по моделям
    all_results = []

    # Тренування вибраних моделей
    for model_name in selected_models:

        # Динамічний імпорт конфігурації моделі
        model_module = import_module(
            f".models.{model_name}", package=__package__)
        get_model_func = getattr(model_module, f"get_{model_name}_model")

        # Створення моделі
        model = get_model_func(
            y) if model_name in ['xgboost', 'lightgbm'] else get_model_func()

        # Тренування через базовий компонент
        model_results = train_model_base(X, y, model, model_name)

        # Додаємо в список (тільки скалярні значення!)
        all_results.append(model_results)

        # Збереження підсумків по моделям
    if all_results:
        build_model_comparison_report(all_results)


if __name__ == "__main__":
    main()

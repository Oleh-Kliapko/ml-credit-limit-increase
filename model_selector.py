# model_selector.py
# Компонент для вибору моделі користувачем через консольний "список"

# Константа зі списком моделей
AVAILABLE_MODELS = ['logreg', 'xgboost', 'lightgbm', 'histgb']


def select_model():
    print("\n\nДоступні моделі для тренування:")
    for i, model in enumerate(AVAILABLE_MODELS, 1):
        print(f"{i}. {model}")
    print("all - Тренувати всі моделі")
    print("Введіть номер або назву моделі (або 'all'): ")

    choice = input().strip().lower()

    if choice == 'all':
        return AVAILABLE_MODELS

    try:
        # Якщо ввели номер
        num = int(choice)
        if 1 <= num <= len(AVAILABLE_MODELS):
            return [AVAILABLE_MODELS[num - 1]]
    except ValueError:
        # Якщо ввели назву
        if choice in AVAILABLE_MODELS:
            return [choice]

    print(
        f"❌ Невірний вибір. Використовую за замовчуванням: {AVAILABLE_MODELS[0]}")
    return [AVAILABLE_MODELS[0]]

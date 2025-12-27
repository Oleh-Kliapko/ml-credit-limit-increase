# validate_columns.py
# Перевіряємо наявність 6 необхідних колонок в датасеті для подальшої інженерії ознак

REQUIRED_RAW_COLUMNS = [
    "CredHist_MaxCC_out_1m",
    "CredHist_MaxCC_out_3m",
    "CredHist_Sum_cur_out",
    "CredHist_count_all_loans_open_now",
    "CredHist_Max_dpd_ever",
    "CredHist_IsDataAvailable",
]


def validate_columns(df):
    print("=== Перевірка наявності колонок ===")

    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]

    if missing:
        for c in missing:
            print(f"❌ Відсутня колонка: {c}")
        raise ValueError("Немає необхідних колонок для створення фіч")

    print("✅ Всі необхідні колонки присутні\n\n")

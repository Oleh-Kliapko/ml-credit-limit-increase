# limit_stats.py
# Готуємось до створення таргету -
# рахуємо базову статистику, кількість,
# частку та розподіл різниці лімітів


def limit_basic_stats(df):
    df = df.copy()

    print("\n\n========= Базова статистика по лімітах =========")

    cols = ["limit_pre", "limit_in30d"]
    stats = df[cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    print(stats)

    print("\n========= Порівняння лімітів =========")

    total = len(df)

    greater = (df["limit_in30d"] > df["limit_pre"]).sum()
    equal = (df["limit_in30d"] == df["limit_pre"]).sum()
    lower = (df["limit_in30d"] < df["limit_pre"]).sum()

    print(f"Всього рядків: {total}")
    print(f"Ліміт виріс      : {greater} ({greater / total:.2%})")
    print(f"Без змін         : {equal} ({equal / total:.2%})")
    print(f"Ліміт зменшився  : {lower} ({lower / total:.2%})\n\n")

    return df

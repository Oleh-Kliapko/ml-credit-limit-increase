# load_data.py
# Зчитати датасет. Визначити структуру і типи даних

import os
import pandas as pd
from .config import DATA_PATH


def load_dataset():
    ext = os.path.splitext(DATA_PATH)[1]

    if ext == ".feather":
        df = pd.read_feather(DATA_PATH)
    else:
        raise ValueError(f"Формат датасету {ext} не коректний")

    print("\n\n========= Форма датасету: =========")
    print(df.shape)
    print("========= Структура датасету та типи даних: =========")
    print(df.dtypes)

    return df

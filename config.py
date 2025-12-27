# config.py

import os

CURRENT_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR))

FILE_NAME = "dataset.feather"

DATA_PATH = os.path.join(BASE_DIR,  "data", FILE_NAME)
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "reports")

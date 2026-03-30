import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directory
DATA_DIR = BASE_DIR / 'data'

# Data files
ARTICLES_PATH = DATA_DIR / 'articles.csv.zip'
CUSTOMERS_PATH = DATA_DIR / 'customers.csv.zip'
TRANSACTIONS_PATH = DATA_DIR / 'transactions_train.csv.zip'

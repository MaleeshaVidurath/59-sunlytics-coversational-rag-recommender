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

# Force HuggingFace to download massively heavy AI models to the F:\ drive instead of C:\
HF_CACHE_DIR = BASE_DIR / 'huggingface_cache'
HF_CACHE_DIR.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(HF_CACHE_DIR)

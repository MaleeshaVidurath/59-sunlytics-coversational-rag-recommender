import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directory
DATA_DIR = BASE_DIR / 'data'

# Sample dataset (41,795 articles — aligned with FAISS index)
SAMPLE_DATA_DIR = BASE_DIR / 'shared' / 'main_data_set'
ARTICLES_PATH = SAMPLE_DATA_DIR / 'sample_articles.csv'

# Full H&M dataset files (M1 Graph RAG / historical data)
CUSTOMERS_PATH = DATA_DIR / 'customers.csv.zip'
TRANSACTIONS_PATH = DATA_DIR / 'transactions_train.csv.zip'

# Force HuggingFace to download massively heavy AI models to the F:\ drive instead of C:\
HF_CACHE_DIR = BASE_DIR / 'huggingface_cache'
HF_CACHE_DIR.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(HF_CACHE_DIR)

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

# M2 model weights cache (BLIP + CLIP) — kept inside m2_multimodal_rag for module cohesion
HF_CACHE_DIR = BASE_DIR / 'm2_multimodal_rag' / 'models'
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_CACHE_DIR)

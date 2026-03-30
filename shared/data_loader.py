import os
from pathlib import Path
import pandas as pd
from shared.config import DATA_DIR, ARTICLES_PATH, CUSTOMERS_PATH, TRANSACTIONS_PATH

class DataLoader:
    def __init__(self):
        self.articles_df = None
        self.customers_df = None
        self.transactions_df = None
        self.api = None
        
        # Setup local image cache directory
        self.image_cache_dir = DATA_DIR / 'images'
        self.image_cache_dir.mkdir(parents=True, exist_ok=True)

    def _init_kaggle_api(self):
        """Initializes the Kaggle API for on-demand image downloading."""
        if self.api is None:
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
                self.api = KaggleApi()
                self.api.authenticate()
            except ImportError:
                raise ImportError("Please ensure the 'kaggle' package is installed: pip install kaggle")

    def load_articles(self):
        """Loads and returns the articles dataset."""
        if self.articles_df is None:
            if ARTICLES_PATH.exists():
                print(f"Loading articles from {ARTICLES_PATH}...")
                self.articles_df = pd.read_csv(ARTICLES_PATH)
            else:
                raise FileNotFoundError(f"File not found: {ARTICLES_PATH}. Please wait for background download.")
        return self.articles_df

    def load_customers(self):
        """Loads and returns the customers dataset."""
        if self.customers_df is None:
            if CUSTOMERS_PATH.exists():
                print(f"Loading customers from {CUSTOMERS_PATH}...")
                self.customers_df = pd.read_csv(CUSTOMERS_PATH)
            else:
                raise FileNotFoundError(f"File not found: {CUSTOMERS_PATH}. Please wait for background download.")
        return self.customers_df

    def load_transactions(self):
        """Loads and returns the transactions dataset."""
        if self.transactions_df is None:
            if TRANSACTIONS_PATH.exists():
                print(f"Loading transactions from {TRANSACTIONS_PATH}...")
                self.transactions_df = pd.read_csv(TRANSACTIONS_PATH)
            else:
                raise FileNotFoundError(f"File not found: {TRANSACTIONS_PATH}. Please wait for background download.")
        return self.transactions_df

    def load_all(self):
        """Loads and returns all datasets (articles, customers, transactions)."""
        self.load_articles()
        self.load_customers()
        self.load_transactions()
        return self.articles_df, self.customers_df, self.transactions_df

    def get_image(self, article_id: str) -> Path:
        """
        Dynamically fetches an article image using the Kaggle API and returns its local path.
        If the image is already cached, it instantly returns the cached path.
        
        Args:
            article_id (str): The unique 10-digit ID string of the article.
        Returns:
            Path: The local path to the downloaded image.
        """
        # Ensure ID is a 10-digit string
        article_id_str = str(article_id).zfill(10)
        
        # H&M dataset categorizes images by first 3 digits of article_id
        folder_prefix = article_id_str[:3]
        remote_path = f"images/{folder_prefix}/{article_id_str}.jpg"
        
        # Local cache path
        local_image_path = self.image_cache_dir / folder_prefix / f"{article_id_str}.jpg"
        
        if local_image_path.exists():
            return local_image_path

        # Lazy initialize Kaggle API to save memory if never used
        self._init_kaggle_api()
        
        print(f"Downloading image for article {article_id_str}...")
        try:
            # We must specify the exact destination to match local_image_path mapping
            dest_dir = self.image_cache_dir / folder_prefix
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Note: The api downloads the file preserving its name (e.g. 0108775015.jpg)
            self.api.competition_download_file(
                'h-and-m-personalized-fashion-recommendations', 
                remote_path, 
                path=str(dest_dir)
            )
            return local_image_path
            
        except Exception as e:
            print(f"Warning: Could not fetch image for article {article_id_str}. Details: {e}")
            return None

# Global instance for easy import across M1, M2, and M3 modules
data_loader = DataLoader()

import pandas as pd
import os

# 1. Define the path to your shared data folder
# Step UP one level from m1_graph_rag to the root, then DOWN into shared/main_data_set
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../shared/main_data_set")

def load_sample_data():
    print("Loading sample data...")
    
    # 2. Read the CSV files into Pandas DataFrames
    articles_df = pd.read_csv(os.path.join(DATA_DIR, "sample_articles.csv"))
    customers_df = pd.read_csv(os.path.join(DATA_DIR, "sample_customers.csv"))
    transactions_df = pd.read_csv(os.path.join(DATA_DIR, "sample_transactions.csv"))
    
    print(f"Loaded {len(customers_df)} customers, {len(articles_df)} articles, and {len(transactions_df)} transactions.")
    
    return customers_df, articles_df, transactions_df

if __name__ == "__main__":
    # Test run
    load_sample_data()
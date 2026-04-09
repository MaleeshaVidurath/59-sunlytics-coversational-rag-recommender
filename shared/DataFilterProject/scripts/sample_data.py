import pandas as pd
from collections import Counter

# File paths
transactions_file = "../data/transactions_train.csv"
customers_file = "../data/customers.csv"
articles_file = "../data/articles.csv"

print("Step 1: Counting transactions per user...")

user_counts = Counter()

# Read in chunks (important for large file)
for chunk in pd.read_csv(transactions_file, chunksize=500000):
    counts = chunk['customer_id'].value_counts()
    user_counts.update(counts.to_dict())

df_counts = pd.DataFrame(user_counts.items(), columns=['customer_id', 'transaction_count'])
df_counts = df_counts.sort_values(by='transaction_count', ascending=False)

print(df_counts.head(250))
# Save top 250 users with transaction counts
df_counts.head(250).to_csv("../data/top_250_user_transaction_counts.csv", index=False)

# 🔥 CHANGE THIS AFTER SEEING RESULTS
TOP_N = 250

top_users = df_counts.head(TOP_N)['customer_id']

print(f"\nSelected top {TOP_N} users")

# Step 2: Filter transactions
print("Step 2: Filtering transactions...")
transactions = pd.read_csv(transactions_file)
filtered_transactions = transactions[transactions['customer_id'].isin(top_users)]

# Step 3: Filter customers
print("Step 3: Filtering customers...")
customers = pd.read_csv(customers_file)
filtered_customers = customers[customers['customer_id'].isin(top_users)]

# Step 4: Filter articles
print("Step 4: Filtering articles...")
articles = pd.read_csv(articles_file)

article_ids = filtered_transactions['article_id'].unique()
filtered_articles = articles[articles['article_id'].isin(article_ids)]

# Step 5: Save outputs
print("Step 5: Saving files...")

filtered_transactions.to_csv("../data/sample_transactions.csv", index=False)
filtered_customers.to_csv("../data/sample_customers.csv", index=False)
filtered_articles.to_csv("../data/sample_articles.csv", index=False)

print("✅ Done! Sample dataset created.")
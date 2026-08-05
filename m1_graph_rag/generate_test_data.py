import pandas as pd
import json
import random

def main():
    print("1. Loading raw CSV data...")
    # Ensure these file names match your actual files exactly
    transactions = pd.read_csv("sample_transactions.csv")
    articles = pd.read_csv("sample_articles.csv")

    print("2. Slicing the final 7 days of data...")
    transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])
    max_date = transactions['t_dat'].max()
    split_date = max_date - pd.Timedelta(days=7)
    test_transactions = transactions[transactions['t_dat'] >= split_date]

    print("3. Grouping purchases by customer...")
    grouped = test_transactions.groupby('customer_id')['article_id'].apply(list).reset_index()

    random.seed(42)
    # Sample 50 random customers for the test set
    if len(grouped) > 50:
        eval_customers = grouped.sample(n=50, random_state=42)
    else:
        eval_customers = grouped

    print("4. Synthesizing conversational queries...")
    test_dataset = []
    articles_dict = articles.set_index('article_id').to_dict(orient='index')

    templates = [
        "I am looking for a {color} {type}.",
        "Do you have any {color} {type}s?",
        "Show me some options for a {type}, preferably {color}.",
        "I need a new {type}. I like {color}.",
        "Could you recommend a {color} {type}?"
    ]

    for _, row in eval_customers.iterrows():
        customer_id = row['customer_id']
        purchased_items = row['article_id']
        
        # Use the first purchased item as the "seed" to generate the text prompt
        seed_item_id = purchased_items[0]
        
        if seed_item_id in articles_dict:
            item_metadata = articles_dict[seed_item_id]
            color = str(item_metadata.get('colour_group_name', '')).lower()
            prod_type = str(item_metadata.get('product_type_name', '')).lower()
            
            template = random.choice(templates)
            user_message = template.format(color=color, type=prod_type)
            
            test_dataset.append({
                "customer_id": customer_id,
                "user_message": user_message,
                # Convert IDs to strings so they perfectly match your graph nodes
                "ground_truth_articles": [str(item) for item in purchased_items], 
                "metadata_used": {
                    "color": color,
                    "type": prod_type
                }
            })

    print("5. Saving output to JSON...")
    with open("test_dataset.json", 'w') as f:
        json.dump(test_dataset, f, indent=4)

    print(f"✅ Success! Generated {len(test_dataset)} test cases and saved to 'test_dataset.json'.")

if __name__ == "__main__":
    main()
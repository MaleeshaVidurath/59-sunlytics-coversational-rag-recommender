import pandas as pd
import torch
import os
from sentence_transformers import SentenceTransformer

# Match the exact path from your data_loader.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../shared/main_data_set")

def generate_product_embeddings():
    print("\n--- Starting AI Text Translator ---")
    print("Loading the language model (this takes a few seconds the first time)...")
    
    # Load a highly efficient, lightweight AI model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 1. Load your article data
    csv_path = os.path.join(DATA_DIR, "sample_articles.csv")
    print(f"Loading products from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Clean the data: replace any empty descriptions with blank spaces so it doesn't crash
    df.fillna('', inplace=True)
    
    # 2. Stitch the text together into a rich sentence for the AI to read
    sentences = []
    article_ids = []
    
    for index, row in df.iterrows():
        # Example: "Product: Valerie Dress. Type: Dress. Color: Black. Description: Lightweight summer cotton dress."
        combined_text = f"Product: {row.get('prod_name', '')}. " \
                        f"Type: {row.get('product_type_name', '')}. " \
                        f"Color: {row.get('colour_group_name', '')}. " \
                        f"Description: {row.get('detail_desc', '')}"
        
        sentences.append(combined_text)
        
        # Save the exact ID so we can map the math back to the correct item
        article_ids.append(row['article_id'])

    # 3. The Magic: Translate the English sentences into Math
    print(f"Translating {len(sentences)} product descriptions into math vectors...")
    
    # This creates a massive matrix where every row is a product, and the columns are the 384 numbers
    embeddings = model.encode(sentences, convert_to_tensor=True)
    
    print(f"Success! Created a mathematical matrix of shape: {embeddings.shape}")
    print(f"This means we have {embeddings.shape[0]} items, and each one is defined by {embeddings.shape[1]} numbers.")
    
    # Save the math to your hard drive so we don't have to recalculate it ever again!
    torch.save(embeddings, os.path.join(BASE_DIR, 'product_embeddings.pt'))
    torch.save(article_ids, os.path.join(BASE_DIR, 'product_ids.pt'))
    print("Saved embeddings to disk!")
    return embeddings, article_ids

if __name__ == "__main__":
    # Test the script!
    product_math, id_list = generate_product_embeddings()
    
    # Show a sneak peek of the math for the very first item
    print(f"\nSneak peek at the math for Article {id_list[0]}:")
    print(product_math[0][:5].tolist(), "... (and 379 more numbers!)")
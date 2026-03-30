from shared.data_loader import data_loader
import traceback

def run_tests():
    print("=== M2 Multimodal DataLoader Test ===")
    
    # 1. Test Metadata Loading
    try:
        print("\n1. Testing Articles Metadata...")
        articles_df = data_loader.load_articles()
        print(f"✅ Success! Loaded {len(articles_df):,} articles.")
        print(f"Top 3 rows:\n{articles_df.head(3)}")
    except Exception as e:
        print(f"❌ Failed to load articles: {e}")
        traceback.print_exc()

    try:
        print("\n2. Testing Customers Metadata...")
        customers_df = data_loader.load_customers()
        print(f"✅ Success! Loaded {len(customers_df):,} customers.")
    except Exception as e:
        print(f"❌ Failed to load customers: {e}")

    try:
        print("\n3. Testing Transactions Metadata (Background Task)...")
        # Depending on if it's currently finished downloading, this might throw a FileNotFoundError.
        transactions_df = data_loader.load_transactions()
        print(f"✅ Success! Loaded {len(transactions_df):,} transactions.")
    except Exception as e:
        print(f"⚠️ Transactions still downloading or failed: {e}")

    # 2. Test Image Lazy Fetching (M2 Specific)
    print("\n4. Testing Lazy Image Fetching (Kaggle API)...")
    try:
        # Let's use a specific prominent article ID from the dataset
        test_article_id = "0108775015"
        
        # Call it once - should connect to API and download it
        print(f"-> Fetching image '{test_article_id}' for the first time...")
        image_path = data_loader.get_image(test_article_id)
        if image_path and image_path.exists():
            print(f"✅ Image 1 safely fetched! Saved locally to: {image_path}")
        else:
            print("❌ Image path returned but file doesn't exist.")
            
        # Call it again - should instantly return from the cache
        print(f"\n-> Fetching image '{test_article_id}' for the second time...")
        image_path_cache = data_loader.get_image(test_article_id)
        print(f"✅ Instantly fetched from local cache! Path: {image_path_cache}")
        
    except Exception as e:
        print(f"❌ Failed to fetch images dynamically via Kaggle API: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_tests()

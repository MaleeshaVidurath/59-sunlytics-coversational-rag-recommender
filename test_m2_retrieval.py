import sys
import os

# Ensure the root directory is accessible for imports (just in case this is run directly)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from m2_multimodal_rag.retrieval import m2_retriever

def test_pipeline():
    print("==================================================")
    print("🚀 M2 MULTIMODAL RETRIEVAL PIPELINE TEST")
    print("==================================================\n")

    print("[TEST 1]: Pure Text Query...")
    results1 = m2_retriever.get_recommendations(text_query="A stylish black leather jacket")
    print(f"✅ Output: {results1}\n")

    print("[TEST 2]: Multimodal (Image + Text) Noise Removal Test...")
    # We use the 0108775015 image we downloaded earlier!
    test_image = "data/images/010/0108775015.jpg"
    
    if os.path.exists(test_image):
        results2 = m2_retriever.get_recommendations(
            text_query="I want clothes in the same style and color as the item in this photo", 
            image_path=test_image
        )
        print(f"✅ Output: {results2}\n")
    else:
        print("Test 2 Skipped - Could not find test image. Run test_image_fetch.py first!")

if __name__ == "__main__":
    test_pipeline()

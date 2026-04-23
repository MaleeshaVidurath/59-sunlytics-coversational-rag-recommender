import faiss
import pandas as pd
import numpy as np
import os
import traceback
from shared.config import DATA_DIR
import sys

# Optional try-except logic to catch missing Faiss library locally
try:
    import faiss
except ImportError:
    print("Warning: faiss-cpu is not installed. Please run: pip install faiss-cpu")

class FAISSDatabase:
    """
    Local Vector Engine for lightning-fast retrieval over the complete 105,000 product space.
    """
    def __init__(self):
        print("Initializing Local FAISS Search Engine...")
        self.index_path = DATA_DIR / 'm2_clip_faiss.bin'
        self.mapping_path = DATA_DIR / 'm2_faiss_mapping.csv'
        self.index = None
        self.mapping = None
        
        # Check if user has downloaded the results from the Kaggle Notebook yet
        self.database_ready = self.index_path.exists() and self.mapping_path.exists()
        
        if self.database_ready:
            try:
                # Load the compiled 150MB vector index
                self.index = faiss.read_index(str(self.index_path))
                # Load the row-to-article_id mapping so we know which vector = which product
                self.mapping_df = pd.read_csv(self.mapping_path)
                self.mapping = self.mapping_df['article_id'].astype(str).str.zfill(10).tolist()
                print(f"[OK] Successfully loaded {self.index.ntotal:,} vectors from FAISS database!")
                
            except Exception as e:
                print(f"[ERROR] Error loading FAISS database: {e}")
                self.database_ready = False
        else:
            print("[WARNING] 'm2_clip_faiss.bin' or mapping NOT FOUND in /data/ directory.")
            print("[WARNING] The search will run in DUMMY mode for UI testing until you run your Kaggle Cloud notebook!")

    def search(self, query_vector: np.ndarray, top_k: int = 5):
        """
        Executes an Inner-Product (Cosine Similarity) search in FAISS using the 512-D CLIP vector.
        """
        if query_vector is None:
            return []

        # Ensure correct shape (batch_size=1, dimensionality=512)
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)

        query_vector = query_vector.astype('float32')

        if not self.database_ready:
            print(f"[WARNING] FAISS running in fallback DUMMY MODE. Returning {top_k} placeholder recommendations...")
            # Return dummy valid H&M IDs so the BLIP Verifier can still be built/tested offline!
            dummy_articles = ["0108775015", "0108775044", "0111565001", "0111586001", "0111593001"][:top_k]
            # Dummy decreasing confidence scores (1.0 -> 0.6)
            dummy_scores = [1.0 - (i * 0.1) for i in range(len(dummy_articles))]
            return list(zip(dummy_articles, dummy_scores))

        # Launch the actual real-time FAISS RAM search
        distances, indices = self.index.search(query_vector, top_k)
        
        # Yield the (article_id, score) pairs
        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx != -1:  # FAISS returns -1 if there are fewer items than top_k 
                article_id = self.mapping[idx]
                results.append((str(article_id), float(score)))
                
        return results

# Singleton access point for entire app
faiss_db = FAISSDatabase()

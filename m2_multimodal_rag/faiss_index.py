import faiss
import pandas as pd
import numpy as np
import os
import traceback
from pathlib import Path
import sys

# Vector DB lives inside the m2_multimodal_rag module, not the shared data/ folder
VECTOR_DB_DIR = Path(__file__).resolve().parent / 'vector_db'

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
        self.index_path = VECTOR_DB_DIR / 'm2_clip_faiss.bin'
        self.mapping_path = VECTOR_DB_DIR / 'm2_faiss_mapping.csv'
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
            print("[WARNING] 'm2_clip_faiss.bin' or mapping NOT FOUND in m2_multimodal_rag/vector_db/ directory.")
            print("[WARNING] Download both files from the Kaggle notebook Output tab and place them in m2_multimodal_rag/vector_db/")

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

    def get_item_vector(self, article_id: str) -> np.ndarray:
        """
        NOVELTY 3 (MMR helper): Reconstructs the stored 512-D CLIP vector for a
        given article from the FAISS index.  Works for flat index types (IndexFlatIP).
        Returns None on failure so MMR can fall back to attribute-based similarity.
        """
        if not self.database_ready:
            return None
        try:
            idx = self.mapping.index(article_id)
            vec = self.index.reconstruct(idx)
            return vec.reshape(1, -1).astype('float32')
        except Exception:
            return None

    def mmr_select(self, candidates: list, query_vector: np.ndarray,
                   top_k: int = 2, lambda_param: float = 0.7) -> list:
        """
        NOVELTY 3: Maximal Marginal Relevance (MMR) diversity-aware selection.

        Selects top_k items that maximise both relevance to the query AND
        diversity from each other, using the formula:

            MMR(i) = λ × relevance(i) − (1−λ) × max_sim(i, already_selected)

        Similarity is computed from reconstructed FAISS item vectors (cosine),
        with a metadata-attribute fallback when vector reconstruction fails.
        Paper: Gen-RecSys — diversity as an open challenge in recommendation.
        """
        if len(candidates) <= top_k:
            return candidates

        # Pre-fetch item CLIP vectors for pairwise similarity computation
        item_vectors = {c['article_id']: self.get_item_vector(c['article_id']) for c in candidates}

        selected = []
        remaining = list(candidates)

        for _ in range(top_k):
            if not remaining:
                break

            best_item = None
            best_mmr_score = float('-inf')

            for candidate in remaining:
                aid = candidate['article_id']
                relevance = candidate['final_score']

                if not selected:
                    mmr_score = relevance
                else:
                    vec_i = item_vectors.get(aid)
                    max_sim = 0.0

                    for sel in selected:
                        vec_s = item_vectors.get(sel['article_id'])

                        if vec_i is not None and vec_s is not None:
                            # True cosine similarity between stored CLIP vectors
                            dot = float(np.dot(vec_i.flatten(), vec_s.flatten()))
                            norm_i = float(np.linalg.norm(vec_i))
                            norm_s = float(np.linalg.norm(vec_s))
                            sim = dot / (norm_i * norm_s + 1e-8)
                        else:
                            # Fallback: attribute overlap as diversity proxy
                            m_i = candidate.get('metadata', {})
                            m_s = sel.get('metadata', {})
                            attrs = ['colour_group_name', 'product_type_name',
                                     'department_name', 'graphical_appearance_name']
                            sim = sum(
                                1 for a in attrs
                                if str(m_i.get(a, '')).lower() == str(m_s.get(a, '')).lower()
                            ) / len(attrs)

                        max_sim = max(max_sim, sim)

                    mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_item = candidate

            if best_item:
                selected.append(best_item)
                remaining.remove(best_item)
                print(f"   [MMR] Picked {best_item['article_id']} "
                      f"(relevance={best_item['final_score']:.4f}, MMR={best_mmr_score:.4f})")

        return selected


# Singleton access point for entire app
faiss_db = FAISSDatabase()

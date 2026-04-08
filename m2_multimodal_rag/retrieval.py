from m2_multimodal_rag.query_understanding import vlm_query_processor
from m2_multimodal_rag.clip_embeddings import clip_encoder
from m2_multimodal_rag.faiss_index import faiss_db

class MultimodalRetriever:
    """
    Master Orchestrator for M2 Retrieval Phase.
    Automatically handles textual, visual, or complex multimodal inputs, removes noise,
    and returns exact matching `article_id`s from the FAISS database.
    """
    def __init__(self):
        print("M2 Orchestrator: Bringing Multimodal Retriever Pipeline online...")
        
    def get_recommendations(self, text_query=None, image_path=None, top_k=5):
        """
        Executes the 3-step retrieval pipeline seamlessly.
        """
        if not text_query and not image_path:
            raise ValueError("M2 Error: You must provide either a text query or an image path.")
            
        print("\n--- Executing M2 FAISS Retrieval Pipeline ---")
        
        # Step 1: Query Understanding & Noise Removal
        clean_text_query = vlm_query_processor.extract_search_query(text_query, image_path)
        print(f"Step 1 | VLM Output Strategy: '{clean_text_query}'")
        
        # Step 2: Vector Math Compilation
        query_vector = clip_encoder.encode_text(clean_text_query)
        if query_vector is not None:
            print(f"Step 2 | CLIP Vector Generated (Shape: {query_vector.shape})")
        else:
            print("Step 2 | FAILED to generate CLIP vector.")
            return []
            
        # Step 3: Fast FAISS Database Execution
        results = faiss_db.search(query_vector, top_k=top_k)
        print(f"Step 3 | FAISS Match Found. Top {len(results)} items isolated.")
        print("-" * 45)
        
        return results

# Establish a global accessor
m2_retriever = MultimodalRetriever()

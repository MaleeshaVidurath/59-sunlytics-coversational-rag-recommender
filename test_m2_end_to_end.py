import sys
import os

# Ensure project root is accessible
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from m2_multimodal_rag.retrieval import m2_retriever
from m2_multimodal_rag.regeneration_loop import generator_loop

def run_full_m2_pipeline_demo():
    print("=========================================================")
    print("🚀 MODULE 2: END-TO-END SYSTEM DEMONSTRATION")
    print("=========================================================\n")
    
    # ---------------------------------------------------------
    # STEP 1: The User's Query
    # ---------------------------------------------------------
    user_query = "I am looking for a stylish dark top for women"
    print(f"👤 USER QUERY: \"{user_query}\"\n")
    print("[SYSTEM]: Firing up Retrieval Engine...\n")
    
    # ---------------------------------------------------------
    # STEP 2: Retrieval Engine (CLIP + FAISS)
    # ---------------------------------------------------------
    # This automatically cleans the query, embeds it mathematically, 
    # and searches our Faiss vector database.
    try:
        recommendations = m2_retriever.get_recommendations(text_query=user_query, top_k=3)
    except Exception as e:
        print(f"❌ Error in Retrieval: {e}")
        return

    if not recommendations:
        print("❌ Retrieval failed to find matches.")
        return
        
    print("\n[SYSTEM]: FAISS Database found 3 vector matches.")
    
    # We take the absolute best match (the #1 result)
    top_article_id, top_score = recommendations[0]
    print(f"🎯 BEST MATCH: Article ID '{top_article_id}' (Match Score: {top_score:.2f})\n")

    # ---------------------------------------------------------
    # STEP 3: LLM Generation + VLM Verification Loop
    # ---------------------------------------------------------
    print("[SYSTEM]: Sending top match to LLM for conversational description...")
    print("[SYSTEM]: Activating Visual Verification Guard to ensure no hallucinations...\n")
    
    final_verified_response = generator_loop.generate_faithful_explanation(
        article_id=top_article_id, 
        force_hallucination_test=False # Set to True if we want to force the Manager to catch a lie
    )

    # ---------------------------------------------------------
    # STEP 4: Final Output
    # ---------------------------------------------------------
    print("\n=========================================================")
    print("✅ FINAL VERIFIED SYSTEM RESPONSE TO USER:")
    print("=========================================================")
    print(f"\"Based on your search for '{user_query}', {final_verified_response}\"")

if __name__ == "__main__":
    run_full_m2_pipeline_demo()

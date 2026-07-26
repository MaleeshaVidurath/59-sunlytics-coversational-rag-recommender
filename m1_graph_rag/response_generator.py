import os
from groq import Groq
from nli_guard import verify_against_graph

try:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    print(f"Warning: Groq client failed to initialize. {e}")
    client = None

def translate_graph_paths(action, raw_data, counterfactuals=None):
    """
    Translates raw NetworkX graph data and counterfactual prunings into a structured text template.
    """
    context = ""
    if action == "catalog_search":
        if raw_data:
            context += "Recommended items based on Knowledge Graph:\n"
            for item in raw_data:
                context += f"- {item['name']} (ID: {item['article_id']}). Graph Score: {item['final_score']}\n"
                context += f"  Logical Reasoning: {item['reasoning_path']}\n"
            
        if counterfactuals:
            context += "\nItems Pruned/Discarded (Counterfactual Reasoning):\n"
            for cf in counterfactuals:
                context += f"- Skipped {cf['name']} (ID: {cf['article_id']}): {cf['reason']}\n"

    elif action == "item_attribute_lookup":
        for item in raw_data:
            context += f"Details for {item['name']} (ID: {item['article_id']}):\n"
            context += f"Attributes for '{item['topic_requested']}': {', '.join(item['graph_attributes_found'])}\n"
            
    elif action == "item_compare":
        for comp in raw_data:
            context += f"Comparing {comp['item_a']['name']} and {comp['item_b']['name']}:\n"
            context += f"- Shared traits: {', '.join(comp['shared_features'])}\n"
            context += f"- Unique to {comp['item_a']['name']}: {', '.join(comp['item_a']['unique_features'])}\n"
            context += f"- Unique to {comp['item_b']['name']}: {', '.join(comp['item_b']['unique_features'])}\n"
            
    elif action == "explanation_generate":
        for item in raw_data:
            context += f"Why we suggested {item['name']}:\n"
            context += f"Verified graph connections: {', '.join(item['verified_graph_paths'])}\n"
            
    elif action == "item_detail_lookup":
        for item in raw_data:
            context += f"Full profile for {item['name']}:\n"
            context += f"- Description: {item['description']}\n"
            context += f"- Connected Graph Nodes: {', '.join(item['graph_connections'])}\n"

    return context

def _call_groq(system_prompt, user_prompt, temperature=0.3):
    """Helper function to execute Groq LLM completion."""
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model="llama-3.1-8b-instant",
        temperature=temperature,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

def generate_llm_response(action, user_message, raw_graph_data, counterfactuals=None):
    """
    Generates response with DeBERTa NLI Hallucination Guard and Counterfactual Explanations.
    Returns: (final_response_text: str, hallucination_flag: bool, hallucination_score: float)
    """
    if not raw_graph_data and not counterfactuals:
        return "I'm sorry, I couldn't find any items matching your request in the catalog.", False, 0.0

    # 1. Prepare Premise from Graph Data and Counterfactuals
    graph_context = translate_graph_paths(action, raw_graph_data, counterfactuals=counterfactuals)
    
    if not client:
        return f"[Raw Output Mode]\n{graph_context}", False, 0.0

    system_prompt = (
        "You are a helpful fashion shopping assistant. "
        "Answer using ONLY the provided Knowledge Graph context. "
        "Do NOT invent or hallucinate product features, prices, or colors. "
    )
    
    if action == "catalog_search":
        system_prompt += (
            "CRITICAL INSTRUCTION: You MUST provide a separate, distinct justification for EACH recommended item by translating the 'Logical Reasoning' path. "
            "You MUST also explicitly mention why alternative items were skipped by summarizing the 'Items Pruned' section."
        )
    elif action == "item_compare":
        system_prompt += (
            "CRITICAL INSTRUCTION: Compare the two items clearly. Highlight their shared traits first, then explicitly list what makes each item unique based on the provided attributes."
        )
    elif action == "explanation_generate":
        system_prompt += (
            "CRITICAL INSTRUCTION: Explain why this item is a good match for the user by translating the 'Verified Graph Paths'. Write a natural, friendly sentence acknowledging that the item directly matches their requested preferences."
        )
    elif action == "item_detail_lookup":
        system_prompt += (
            "CRITICAL INSTRUCTION: Provide a helpful and engaging overview of the item. "
            "You MUST use the provided 'description' and graph attributes to describe its style, material, and features naturally."
        )
    elif action == "none":
        system_prompt += (
            "CRITICAL INSTRUCTION: The user is just making friendly conversation or asking a general question. "
            "Do not try to recommend products. Respond naturally, politely, and warmly as a helpful fashion assistant."
        )            
    user_prompt = f"User Message: {user_message}\n\nContext:\n{graph_context}\n\nDraft response:"

    # --- ATTEMPT 1: Standard Generation ---
    try:
        candidate_text = _call_groq(system_prompt, user_prompt, temperature=0.3)
        has_hallucination, score = verify_against_graph(graph_context, candidate_text)

        if not has_hallucination:
            return candidate_text, False, score

        # --- ATTEMPT 2: Strict Retry ---
        print("--> Hallucination detected! Retrying with strict parameters (temp=0.0)...")
        strict_system_prompt = system_prompt + " STRICT WARNING: Stick strictly to the exact facts provided."
        retry_text = _call_groq(strict_system_prompt, user_prompt, temperature=0.0)
        has_hallucination_2, score_2 = verify_against_graph(graph_context, retry_text)

        if not has_hallucination_2:
            return retry_text, False, score_2

        # --- FALLBACK: Deterministic Path Translation ---
        print("--> Retry failed NLI check. Falling back to deterministic graph template.")
        fallback_text = f"Here is the verified information from our catalog:\n\n{graph_context}"
        return fallback_text, True, score_2

    except Exception as e:
        print(f"--> Error during LLM generation: {e}")
        return f"Here is the verified information:\n{graph_context}", False, 0.0
import networkx as nx
import json
import torch
import os
from sentence_transformers import SentenceTransformer
from build_graph import construct_knowledge_graph
from gnn_model import FashionGNN

# ==========================================
# WAKING UP THE AI (Loads once at startup)
# ==========================================
print("Booting up the Recommendation Engine...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Load the Dictionaries
user_mapping = torch.load(os.path.join(BASE_DIR, 'user_mapping.pt'), weights_only=True)
item_mapping = torch.load(os.path.join(BASE_DIR, 'item_mapping.pt'), weights_only=True)

# 2. Load the Text Signatures
product_math = torch.load(os.path.join(BASE_DIR, 'product_embeddings.pt'), weights_only=True)

# 3. Load the Language Model 
print("Loading Language Translator...")
text_model = SentenceTransformer('all-MiniLM-L6-v2')

# 4. Wake up the Trained Brain
print("Loading Trained GNN...")
num_users = len(user_mapping)
num_items = product_math.shape[0]

gnn = FashionGNN(num_users, num_items, product_math)
gnn.load_state_dict(torch.load(os.path.join(BASE_DIR, 'trained_gnn.pt'), weights_only=True))
gnn.eval() 

# 5. Pre-calculate the GNN math
edge_index = torch.load(os.path.join(BASE_DIR, 'edge_index.pt'), weights_only=True)
train_edges = edge_index.clone()
train_edges[1] = train_edges[1] + num_users

with torch.no_grad():
    FINAL_USERS, FINAL_ITEMS = gnn(train_edges)
    
print("AI is fully awake and ready for live searches!\n")

# ==========================================
# REASONING PATH GENERATOR
# ==========================================
def generate_reasoning_path(G, customer_id, recommended_item_id):
    """Finds the logical graph bridge between the user and the item."""
    try:
        raw_path = nx.shortest_path(G, source=customer_id, target=recommended_item_id)
        # Formats the path to look clean: Customer -> Item A -> Black -> Item B
        return " ➔ ".join(str(node) for node in raw_path)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return "Trending item perfectly matching your current vibe."

# ==========================================
# THE CORE SEARCH ENGINE
# ==========================================
def run_catalog_search(G, payload, items_in_context, exclude_ids, user_message, customer_id):
    print("\n--- Executing AI-Powered Graph Catalog Search ---")
    
    filters = payload.get("filters", {})
    soft_constraints = payload.get("soft_constraints", {})
    boosts = payload.get("preference_boosts", [])
    penalties = payload.get("penalties", {})
    hints = payload.get("purchase_history_hints", {})
    
    # ------------------------------------------
    # LIVE AI TRANSLATION
    # ------------------------------------------
    print(f"--> Translating user message: '{user_message}'")
    user_text_math = text_model.encode(user_message, convert_to_tensor=True)
    
    gnn_user_math = None
    if customer_id in user_mapping:
        math_idx = user_mapping[customer_id]
        gnn_user_math = FINAL_USERS[math_idx]
    else:
        print(f"--> Note: Customer '{customer_id}' is new! Relying strictly on Text Vibe.")

    # ------------------------------------------
    # PART A: STRICT FILTERS
    # ------------------------------------------
    all_articles = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'article']
    valid_items = []
    
    for article_id in all_articles:
        if article_id in exclude_ids:
            continue 
            
        is_valid = True
        for key, required_value in filters.items():
            if not G.has_edge(article_id, required_value):
                is_valid = False
                break 
                
        if is_valid:
            valid_items.append(article_id)

    if not valid_items:
        return {"status": "success", "data": []}

    # ------------------------------------------
    # PART B: THE SCORING ALGORITHM
    # ------------------------------------------
    item_scores = {}
    
    for item_id in valid_items:
        score = 1.0 # Base score
        item_attributes = list(G.neighbors(item_id))
        
        # --- 1. THE AI SCORES ---
        if item_id in item_mapping:
            item_idx = item_mapping[item_id]
            
            # THE VIBE SCORE (Text Match)
            item_text_math = product_math[item_idx]
            vibe_match = torch.nn.functional.cosine_similarity(user_text_math.unsqueeze(0), item_text_math.unsqueeze(0)).item()
            score += (vibe_match * 5.0) # Multiply by 5 to give the text match strong weight!
            
            # THE HISTORY SCORE (GNN Prediction)
            if gnn_user_math is not None:
                item_gnn_math = FINAL_ITEMS[item_idx]
                # Sigmoid squashes the complex graph math into a clean 0.0 to 1.0 percentage
                history_match = torch.sigmoid(torch.dot(gnn_user_math, item_gnn_math)).item()
                score += (history_match * 3.0) # Give history a solid 3 point weight

        # --- 2. THE STRICT RULES (NetworkX) ---
        for boost in boosts:
            if boost.get("value") in item_attributes:
                score += boost.get("weight", 0.0)
                
        for penalty_key, bad_values in penalties.items():
            for bad_val in bad_values:
                if bad_val in item_attributes:
                    score -= 5.0 # Make the penalty massive so they drop to the bottom!
                    
        for constraint_type, constraint_val in soft_constraints.items():
            if constraint_val in item_attributes:
                score += 0.5 
                
        top_colours = hints.get("top_colours", [])
        if any(color in item_attributes for color in top_colours):
            score += 0.1 

        item_scores[item_id] = score

    # ------------------------------------------
    # FINAL SELECTION & REASONING
    # ------------------------------------------
    ranked_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
    top_2_results = []
    
    print("\n--> Scoring complete! Top 2 items selected:")
    for item_id, final_score in ranked_items[:2]:
        item_name = G.nodes[item_id].get('name', 'Unknown Product')
        
        # Generate the logical Graph bridge!
        reasoning = generate_reasoning_path(G, customer_id, item_id)
        
        result_package = {
            "article_id": item_id,
            "name": item_name,
            "final_score": round(final_score, 2),
            "reasoning_path": reasoning
        }
        top_2_results.append(result_package)
        print(f"    ⭐ {item_name} (ID: {item_id}) | Math Score: {round(final_score, 2)}")
        print(f"       Path: {reasoning}")
        
    return {"status": "success", "data": top_2_results}

# ==========================================
# SECONDARY HELPERS
# ==========================================
def run_attribute_lookup(G, payload): return {"status": "success", "data": "Placeholder"}
def run_item_compare(G, payload): return {"status": "success", "data": "Placeholder"}
def run_explanation_generate(G, payload): return {"status": "success", "data": "Placeholder"}
def run_item_detail_lookup(G, payload): return {"status": "success", "data": "Placeholder"}

# ==========================================
# THE TICKET READER (The Router)
# ==========================================
def handle_retrieval_request(G, retrieval_input):
    if retrieval_input is None: return None

    action = retrieval_input.get("action")
    items_in_context = retrieval_input.get("items_in_context", {})
    exclude_ids = retrieval_input.get("exclude_ids", [])
    payload = retrieval_input.get("payload", {})
    
    # Extract the live message and user ID!
    user_message = retrieval_input.get("user_message", "")
    customer_id = retrieval_input.get("customer_id", "Unknown") 

    if action == "catalog_search":
        # Pass the message and ID directly into our upgraded search engine!
        return run_catalog_search(G, payload, items_in_context, exclude_ids, user_message, customer_id)
    else:
        return {"status": "success", "data": "Routed to secondary function."}

# ==========================================
# TEST THE ENTIRE PIPELINE
# ==========================================
if __name__ == "__main__":
    kg = construct_knowledge_graph() 
    
    dummy_ticket = {
        "action": "catalog_search",
        "retrieval_strategy": "FULL",
        "customer_id": "7f0ac4394297dc4a885d3b9277ba526cbbfbf7fb7cae465b256ed8e55b864f03", # This triggers the GNN History check!
        "user_message": "are there any skirts", # This triggers the Vibe check!
        "items_in_context": {"item_a": None, "item_b": None},
        "exclude_ids": ["108775015"],
        "payload": {
            "filters": {
            }
            
        }
    }

    handle_retrieval_request(kg, dummy_ticket)
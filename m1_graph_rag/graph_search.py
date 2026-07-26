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
    """Finds the logical graph bridge between the user and the item, translating IDs to real names."""
    try:
        raw_path = nx.shortest_path(G, source=customer_id, target=recommended_item_id)
        
        translated_path = []
        for node in raw_path:
            # 1. If the node is the user's long ID, change it to "User"
            if node == customer_id:
                translated_path.append("User")
                
            # 2. If the node exists in the graph and has a 'name' (it's a product)
            elif G.has_node(node) and 'name' in G.nodes[node]:
                product_name = G.nodes[node]['name']
                translated_path.append(f"'{product_name}'")
                
            # 3. Otherwise, it's an attribute like 'Skirt' or 'Black', just keep the text
            else:
                translated_path.append(str(node))
                
        # Formats the path to look clean: User ➔ 'Old Skirt' ➔ Skirt ➔ 'Elsa'
        return " ➔ ".join(translated_path)
        
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return "Trending item matching your preferences."

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
    # PART A: STRICT FILTERS & COUNTERFACTUAL TRACKING
    # ------------------------------------------
    all_articles = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'article']
    valid_items = []
    discarded_reasons = [] # Counterfactual tracking list
    
    for article_id in all_articles:
        item_name = G.nodes[article_id].get('name', 'Unknown Product')
        
        if article_id in exclude_ids:
            discarded_reasons.append({
                "article_id": article_id,
                "name": item_name,
                "reason": "Explicitly excluded by user history/context"
            })
            continue 
            
        is_valid = True
        for key, required_value in filters.items():
            if not G.has_edge(article_id, required_value):
                is_valid = False
                discarded_reasons.append({
                    "article_id": article_id,
                    "name": item_name,
                    "reason": f"Failed filter constraint: Missing attribute node '{required_value}'"
                })
                break 
                
        if is_valid:
            valid_items.append(article_id)

    if not valid_items:
        return {
            "status": "success", 
            "data": [], 
            "counterfactuals": discarded_reasons[:5] # Return top reasons for empty results
        }

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
            score += (vibe_match * 5.0)
            
            # THE HISTORY SCORE (GNN Prediction)
            if gnn_user_math is not None:
                item_gnn_math = FINAL_ITEMS[item_idx]
                history_match = torch.sigmoid(torch.dot(gnn_user_math, item_gnn_math)).item()
                score += (history_match * 3.0)

        # --- 2. THE STRICT RULES & PENALTIES ---
        for boost in boosts:
            if boost.get("value") in item_attributes:
                score += boost.get("weight", 0.0)
                
        for penalty_key, bad_values in penalties.items():
            for bad_val in bad_values:
                if bad_val in item_attributes:
                    score -= 5.0
                    discarded_reasons.append({
                        "article_id": item_id,
                        "name": G.nodes[item_id].get('name', 'Unknown Product'),
                        "reason": f"Penalized due to attribute '{bad_val}'"
                    })
                    
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
        reasoning = generate_reasoning_path(G, customer_id, item_id)
        
        result_package = {
            "article_id": str(item_id),
            "name": item_name,
            "final_score": round(final_score, 2),
            "reasoning_path": reasoning
        }
        top_2_results.append(result_package)

    # Collect counterfactuals for lower-ranked items
    for item_id, final_score in ranked_items[2:5]:
        discarded_reasons.append({
            "article_id": str(item_id),
            "name": G.nodes[item_id].get('name', 'Unknown Product'),
            "reason": f"Lower relevance match score ({round(final_score, 2)})"
        })

    return {
        "status": "success", 
        "data": top_2_results, 
        "counterfactuals": discarded_reasons[:5]
    }

# ==========================================
# SECONDARY HELPERS
# ==========================================

def resolve_graph_id(G, raw_id):
    """Safely resolves an article ID to match the graph's internal format."""
    if not raw_id:
        return None
        
    # If the input is a dictionary (e.g. from items_in_context), extract the ID
    if isinstance(raw_id, dict):
        raw_id = raw_id.get("article_id")
        if not raw_id:
            return None

    id_str = str(raw_id)
    id_padded = id_str.zfill(10)
    id_int = int(raw_id) if id_str.isdigit() else None
    
    if G.has_node(id_padded):
        return id_padded
    elif G.has_node(id_str):
        return id_str
    elif id_int is not None and G.has_node(id_int):
        return id_int
        
    return None

def run_attribute_lookup(G, payload):
    print("\n--- Executing Graph Attribute Lookup ---")
    
    actual_id = resolve_graph_id(G, payload.get("article_id"))
    topic = payload.get("attribute_topic", "general_details")
    
    if not actual_id:
        print(f"--> Error: Article {payload.get('article_id')} not found in graph.")
        return {"status": "error", "data": []}
        
    node_data = G.nodes[actual_id]
    neighbors = list(G.neighbors(actual_id))
    
    found_attributes = []
    
    # Traverse edges and filter based on the requested topic
    for neighbor in neighbors:
        edge_data = G.get_edge_data(actual_id, neighbor)
        relation = edge_data.get("relation", "")
        
        if "colour" in topic.lower() and relation == "HAS_COLOUR":
            found_attributes.append(neighbor)
        elif "type" in topic.lower() or "fit" in topic.lower() and relation == "BELONGS_TO_TYPE":
            found_attributes.append(neighbor)
        elif topic == "general_details" or topic == "material_and_care":
            found_attributes.append(f"{relation}: {neighbor}")

    result = {
        "article_id": str(actual_id),
        "name": node_data.get("name", "Unknown Product"),
        "topic_requested": topic,
        "graph_attributes_found": found_attributes,
        "description": node_data.get("description", "")
    }
    
    print(f"--> Successfully extracted {topic} for: {result['name']}")
    return {"status": "success", "data": [result]}

def run_item_compare(G, payload, items_in_context):
    print("\n--- Executing Graph Comparison ---")
    
    # Check payload first, fallback to items_in_context
    raw_a = payload.get("article_id_a") or items_in_context.get("item_a")
    raw_b = payload.get("article_id_b") or items_in_context.get("item_b")
    
    actual_id_a = resolve_graph_id(G, raw_a)
    actual_id_b = resolve_graph_id(G, raw_b)
    
    if not actual_id_a or not actual_id_b:
        print(f"--> Error: One or both items not found in graph. A: {raw_a} -> {actual_id_a}, B: {raw_b} -> {actual_id_b}")
        return {"status": "error", "data": []}
        
    name_a = G.nodes[actual_id_a].get("name", str(actual_id_a))
    name_b = G.nodes[actual_id_b].get("name", str(actual_id_b))
    
    # Get all connected attribute nodes
    attrs_a = set(G.neighbors(actual_id_a))
    attrs_b = set(G.neighbors(actual_id_b))
    
    # Filter out numeric item IDs AND long 64-character Customer ID hashes
    attrs_a = {str(a) for a in attrs_a if not str(a).isdigit() and len(str(a)) < 40}
    attrs_b = {str(a) for a in attrs_b if not str(a).isdigit() and len(str(a)) < 40}
    
    # Math set operations for shared vs unique traits
    shared = list(attrs_a.intersection(attrs_b))
    unique_a = list(attrs_a - attrs_b)
    unique_b = list(attrs_b - attrs_a)
    
    print(f"--> Successfully compared '{name_a}' vs '{name_b}'")
    
    return {
        "status": "success",
        "data": [{
            "item_a": {"name": name_a, "unique_features": unique_a},
            "item_b": {"name": name_b, "unique_features": unique_b},
            "shared_features": shared
        }]
    }

def run_explanation_generate(G, payload):
    print("\n--- Executing Graph Explanation Generation ---")
    
    actual_id = resolve_graph_id(G, payload.get("article_id"))
    matched_prefs = payload.get("matched_prefs", [])
    
    if not actual_id:
        print(f"--> Error: Article {payload.get('article_id')} not found in graph.")
        return {"status": "error", "data": []}
        
    name = G.nodes[actual_id].get("name", "Unknown Product")
    verified_paths = []
    
    for pref in matched_prefs:
        pref_value = pref.get("attribute_value")
        
        if G.has_edge(actual_id, pref_value):
            edge_data = G.get_edge_data(actual_id, pref_value)
            relation = edge_data.get("relation", "MATCHES")
            verified_paths.append(f"{name} ➔ {relation} ➔ {pref_value}")
            
    explanation_package = {
        "article_id": str(actual_id),
        "name": name,
        "verified_graph_paths": verified_paths
    }
    
    print(f"--> Verified {len(verified_paths)} preference connections for {name}")
    return {"status": "success", "data": [explanation_package]}

def run_item_detail_lookup(G, payload):
    print("\n--- Executing Graph Item Detail Lookup ---")
    
    actual_id = resolve_graph_id(G, payload.get("article_id"))
    
    if not actual_id:
        print(f"--> Error: Article {payload.get('article_id')} not found in graph.")
        return {"status": "error", "data": []}
        
    node_data = G.nodes[actual_id]
    
    connected_attributes = []
    for neighbor in G.neighbors(actual_id):
        # Ignore long customer ID hashes to save LLM tokens
        if len(str(neighbor)) < 40:
            edge_data = G.get_edge_data(actual_id, neighbor)
            relation = edge_data.get("relation") if edge_data else "UNKNOWN_RELATION"
            connected_attributes.append(f"{relation}: {neighbor}")
    
    item_details = {
        "article_id": str(actual_id), # Always return as string to satisfy frontend
        "name": node_data.get("name", "Unknown Product"),
        "description": node_data.get("description", ""),
        "graph_connections": connected_attributes 
    }
    
    print(f"--> Successfully retrieved graph details for: {item_details['name']}")
    return {"status": "success", "data": [item_details]}

# ==========================================
# THE TICKET READER (The Router)
# ==========================================
def handle_retrieval_request(G, retrieval_input):
    if retrieval_input is None: return None

    action = retrieval_input.get("action")
    items_in_context = retrieval_input.get("items_in_context", {})
    exclude_ids = retrieval_input.get("exclude_ids", [])
    payload = retrieval_input.get("payload", {})
    
    user_message = retrieval_input.get("user_message", "")
    customer_id = retrieval_input.get("customer_id", "Unknown") 

    if action == "catalog_search":
        return run_catalog_search(G, payload, items_in_context, exclude_ids, user_message, customer_id)
    elif action == "item_attribute_lookup":
        return run_attribute_lookup(G, payload)
    elif action == "item_compare":
        # Pass payload and items_in_context since the ID could be in either
        return run_item_compare(G, payload, items_in_context)
    elif action == "explanation_generate":
        return run_explanation_generate(G, payload)
    elif action == "item_detail_lookup":
        return run_item_detail_lookup(G, payload)
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
        "customer_id": "7f0ac4394297dc4a885d3b9277ba526cbbfbf7fb7cae465b256ed8e55b864f03",
        "user_message": "are there any skirts",
        "items_in_context": {"item_a": None, "item_b": None},
        "exclude_ids": ["108775015"],
        "payload": {
            "filters": {}
        }
    }

    handle_retrieval_request(kg, dummy_ticket)
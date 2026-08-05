import networkx as nx
import json
import torch
import os
import difflib
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
# HELPER FUNCTIONS & NODE RESOLVERS
# ==========================================

def resolve_graph_id(G, raw_id):
    """Safely resolves an article ID to match the graph's internal format."""
    if not raw_id:
        return None
        
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

def resolve_attribute_node(G, raw_val):
    """Fuzzy-resolves extracted attribute values to match exact node labels."""
    if not raw_val:
        return None
        
    val_str = str(raw_val).strip()
    
    if G.has_node(val_str):
        return val_str
        
    val_normalized = val_str.lower().replace("-", " ").replace("_", " ")
    
    for node, attr in G.nodes(data=True):
        if attr.get('type') != 'article':
            node_str = str(node)
            node_normalized = node_str.lower().replace("-", " ").replace("_", " ")
            if val_normalized == node_normalized:
                return node
                
    non_article_nodes = [str(n) for n, attr in G.nodes(data=True) if attr.get('type') != 'article']
    matches = difflib.get_close_matches(val_str, non_article_nodes, n=1, cutoff=0.7)
    if matches:
        return matches[0]
        
    return val_str

def generate_reasoning_path(G, customer_id, recommended_item_id):
    """Finds the logical graph bridge between the user and the item."""
    try:
        raw_path = nx.shortest_path(G, source=customer_id, target=recommended_item_id)
        
        translated_path = []
        for node in raw_path:
            if node == customer_id:
                translated_path.append("User")
            elif G.has_node(node) and 'name' in G.nodes[node]:
                product_name = G.nodes[node]['name']
                translated_path.append(f"'{product_name}'")
            else:
                translated_path.append(str(node))
                
        return " ➔ ".join(translated_path)
        
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return "Trending item matching your preferences."

# ==========================================
# THE CORE SEARCH ENGINE
# ==========================================

def run_catalog_search(G, payload, items_in_context, exclude_ids, user_message, customer_id, vibe_weight=5.0, history_weight=3.0):
    print("\n--- Executing AI-Powered Graph Catalog Search ---")
    
    raw_filters = payload.get("filters", {}) or {}
    soft_constraints = payload.get("soft_constraints", {}) or {}
    boosts = payload.get("preference_boosts", []) or {}
    penalties = payload.get("penalties", {}) or {}
    hints = payload.get("purchase_history_hints", {}) or {}
    
    filters = {}
    for k, v in raw_filters.items():
        resolved_v = resolve_attribute_node(G, v)
        filters[k] = resolved_v if resolved_v else v

    print(f"--> Translating user message: '{user_message}'")
    user_text_math = text_model.encode(user_message, convert_to_tensor=True)
    
    gnn_user_math = None
    if customer_id in user_mapping:
        math_idx = user_mapping[customer_id]
        gnn_user_math = FINAL_USERS[math_idx]
    else:
        print(f"--> Note: Customer '{customer_id}' is new! Relying strictly on Text Vibe.")

    all_articles = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'article']
    valid_items = []
    discarded_reasons = [] 
    
    for article_id in all_articles:
        item_name = G.nodes[article_id].get('name', 'Unknown Product')
        
        if article_id in exclude_ids:
            discarded_reasons.append({
                "article_id": str(article_id),
                "name": item_name,
                "reason": "Explicitly excluded by user history/context"
            })
            continue 
            
        is_valid = True
        for key, required_value in filters.items():
            if not G.has_edge(article_id, required_value):
                is_valid = False
                discarded_reasons.append({
                    "article_id": str(article_id),
                    "name": item_name,
                    "reason": f"Failed filter constraint: Missing attribute node '{required_value}'"
                })
                break 
                
        if is_valid:
            valid_items.append(article_id)

    soft_filter_fallback = False
    if not valid_items:
        print("--> Notice: Strict filter yielded 0 items. Softening filter constraints to prevent empty results.")
        valid_items = [n for n in all_articles if n not in exclude_ids]
        soft_filter_fallback = True

    item_scores = {}
    
    for item_id in valid_items:
        score = 1.0 
        item_attributes = list(G.neighbors(item_id))
        
        if item_id in item_mapping:
            item_idx = item_mapping[item_id]
            
            item_text_math = product_math[item_idx]
            vibe_match = torch.nn.functional.cosine_similarity(user_text_math.unsqueeze(0), item_text_math.unsqueeze(0)).item()
            score += (vibe_match * vibe_weight)
            
            if gnn_user_math is not None:
                item_gnn_math = FINAL_ITEMS[item_idx]
                history_match = torch.sigmoid(torch.dot(gnn_user_math, item_gnn_math)).item()
                score += (history_match * history_weight)

        if soft_filter_fallback:
            for key, required_value in filters.items():
                if not G.has_edge(item_id, required_value):
                    score -= 4.0

        for boost in boosts:
            raw_boost_val = boost.get("value")
            resolved_boost_val = resolve_attribute_node(G, raw_boost_val)
            if resolved_boost_val in item_attributes or raw_boost_val in item_attributes:
                score += boost.get("weight", 0.0)
                
        for penalty_key, bad_values in penalties.items():
            for bad_val in bad_values:
                resolved_bad_val = resolve_attribute_node(G, bad_val)
                if resolved_bad_val in item_attributes or bad_val in item_attributes:
                    score -= 5.0
                    discarded_reasons.append({
                        "article_id": str(item_id),
                        "name": G.nodes[item_id].get('name', 'Unknown Product'),
                        "reason": f"Penalized due to attribute '{bad_val}'"
                    })
                    
        for constraint_type, constraint_val in soft_constraints.items():
            resolved_c_val = resolve_attribute_node(G, constraint_val)
            if resolved_c_val in item_attributes or constraint_val in item_attributes:
                score += 0.5 
                
        top_colours = hints.get("top_colours", [])
        for color in top_colours:
            resolved_color = resolve_attribute_node(G, color)
            if resolved_color in item_attributes or color in item_attributes:
                score += 0.1 

        item_scores[item_id] = score

    ranked_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
    
    # ── Extract and normalize requested quantity with default fallback of 2 ──
    raw_quantity = payload.get("quantity") if payload else None

    try:
        if raw_quantity is not None and str(raw_quantity).strip().isdigit():
            requested_quantity = int(raw_quantity)
            if requested_quantity <= 0:
                requested_quantity = 2
        else:
            requested_quantity = 2
    except Exception:
        requested_quantity = 2

    final_results = []
    
    print(f"\n--> Scoring complete! Target items to retrieve: {requested_quantity} (raw input: {raw_quantity})")
    for item_id, final_score in ranked_items[:requested_quantity]:
        # 1. Grab the node data
        node_data = G.nodes[item_id]
        item_name = node_data.get('name', 'Unknown Product')
        
        # Clean the Description: Handle pandas "nan" strings and empty spaces
        raw_desc = str(node_data.get('description', ''))
        if raw_desc.lower() == 'nan' or not raw_desc.strip():
            item_desc = "A stylish and comfortable piece perfect for your wardrobe." # Fallback description
        else:
            item_desc = raw_desc.strip()
        
        # 2. Grab and Format the Price
        item_price = "Price not available"
        for neighbor in G.neighbors(item_id):
            edge_data = G.get_edge_data(item_id, neighbor)
            if edge_data and 'price' in edge_data:
                # Multiply by the exact 590 scale to convert the normalized decimal back to GBP
                try:
                    raw_price = float(edge_data['price'])
                    formatted_price = round(raw_price * 590, 2) 
                    item_price = f"£{formatted_price}"
                except (ValueError, TypeError):
                    item_price = str(edge_data['price'])
                break 
                
        reasoning = generate_reasoning_path(G, customer_id, item_id)
        
        # 3. Add the cleaned description and formatted price to the package
        result_package = {
            "article_id": str(item_id),
            "name": item_name,
            "description": item_desc,
            "price": item_price,
            "final_score": round(final_score, 2),
            "reasoning_path": reasoning
        }
        final_results.append(result_package)

    for item_id, final_score in ranked_items[requested_quantity:requested_quantity + 3]:
        discarded_reasons.append({
            "article_id": str(item_id),
            "name": G.nodes[item_id].get('name', 'Unknown Product'),
            "reason": f"Lower relevance match score ({round(final_score, 2)})"
        })

    return {
        "status": "success", 
        "data": final_results, 
        "counterfactuals": discarded_reasons[:5]
    }

# ==========================================
# SECONDARY HELPERS
# ==========================================

def run_attribute_lookup(G, payload, items_in_context):
    print("\n--- Executing Graph Attribute Lookup ---")
    
    raw_id = payload.get("article_id") or items_in_context.get("item_a")
    actual_id = resolve_graph_id(G, raw_id)
    topic = payload.get("attribute_topic", "general_details")
    
    if not actual_id:
        print(f"--> Error: Article ID not found in payload or context.")
        return {"status": "error", "data": []}
        
    node_data = G.nodes[actual_id]
    neighbors = list(G.neighbors(actual_id))
    
    found_attributes = []
    
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
    
    raw_a = payload.get("article_id_a") or items_in_context.get("item_a")
    raw_b = payload.get("article_id_b") or items_in_context.get("item_b")
    
    actual_id_a = resolve_graph_id(G, raw_a)
    actual_id_b = resolve_graph_id(G, raw_b)
    
    if not actual_id_a or not actual_id_b:
        print(f"--> Error: One or both items not found in graph. A: {raw_a} -> {actual_id_a}, B: {raw_b} -> {actual_id_b}")
        return {"status": "error", "data": []}
        
    name_a = G.nodes[actual_id_a].get("name", str(actual_id_a))
    name_b = G.nodes[actual_id_b].get("name", str(actual_id_b))
    
    attrs_a = set(G.neighbors(actual_id_a))
    attrs_b = set(G.neighbors(actual_id_b))
    
    attrs_a = {str(a) for a in attrs_a if not str(a).isdigit() and len(str(a)) < 40}
    attrs_b = {str(a) for a in attrs_b if not str(a).isdigit() and len(str(a)) < 40}
    
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

def run_explanation_generate(G, payload, items_in_context):
    print("\n--- Executing Graph Explanation Generation ---")
    
    raw_id = payload.get("article_id") or items_in_context.get("item_a")
    actual_id = resolve_graph_id(G, raw_id)
    matched_prefs = payload.get("matched_prefs", [])
    
    if not actual_id:
        print(f"--> Error: Article ID not found in payload or context.")
        return {"status": "error", "data": []}
        
    name = G.nodes[actual_id].get("name", "Unknown Product")
    verified_paths = []
    
    for pref in matched_prefs:
        pref_value = pref.get("attribute_value")
        resolved_pref_value = resolve_attribute_node(G, pref_value) or pref_value
        
        target_node = resolved_pref_value if G.has_edge(actual_id, resolved_pref_value) else (pref_value if G.has_edge(actual_id, pref_value) else None)
        
        if target_node:
            edge_data = G.get_edge_data(actual_id, target_node)
            relation = edge_data.get("relation", "MATCHES")
            verified_paths.append(f"{name} ➔ {relation} ➔ {target_node}")
            
    explanation_package = {
        "article_id": str(actual_id),
        "name": name,
        "verified_graph_paths": verified_paths
    }
    
    print(f"--> Verified {len(verified_paths)} preference connections for {name}")
    return {"status": "success", "data": [explanation_package]}

def run_item_detail_lookup(G, payload, items_in_context):
    print("\n--- Executing Graph Item Detail Lookup ---")
    
    raw_id = payload.get("article_id") or items_in_context.get("item_a")
    actual_id = resolve_graph_id(G, raw_id)
    
    if not actual_id:
        print(f"--> Error: Article ID not found in payload or context.")
        return {"status": "error", "data": []}
        
    node_data = G.nodes[actual_id]
    
    connected_attributes = []
    for neighbor in G.neighbors(actual_id):
        if len(str(neighbor)) < 40:
            edge_data = G.get_edge_data(actual_id, neighbor)
            relation = edge_data.get("relation") if edge_data else "UNKNOWN_RELATION"
            connected_attributes.append(f"{relation}: {neighbor}")
    
    item_details = {
        "article_id": str(actual_id),
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
    if retrieval_input is None: 
        return None

    action = retrieval_input.get("action")
    items_in_context = retrieval_input.get("items_in_context") or {}
    exclude_ids = retrieval_input.get("exclude_ids") or []
    payload = retrieval_input.get("payload") or {}
    
    user_message = retrieval_input.get("user_message", "")
    customer_id = retrieval_input.get("customer_id", "Unknown") 

    if action == "catalog_search":
        return run_catalog_search(G, payload, items_in_context, exclude_ids, user_message, customer_id)
    elif action == "item_attribute_lookup":
        return run_attribute_lookup(G, payload, items_in_context)
    elif action == "item_compare":
        return run_item_compare(G, payload, items_in_context)
    elif action == "explanation_generate":
        return run_explanation_generate(G, payload, items_in_context)
    elif action == "item_detail_lookup":
        return run_item_detail_lookup(G, payload, items_in_context)
    else:
        return {"status": "success", "data": "Routed to secondary function."}
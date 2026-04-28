import networkx as nx
import json
from build_graph import construct_knowledge_graph

# ==========================================
# THE HELPER FUNCTIONS (The Kitchen Stations)
# ==========================================

def run_catalog_search(G, payload, items_in_context, exclude_ids):
    print("\n--- Executing Graph Catalog Search ---")
    
    # 1. Unpack the rules from Member 3's ticket
    filters = payload.get("filters", {})
    soft_constraints = payload.get("soft_constraints", {})
    boosts = payload.get("preference_boosts", [])
    penalties = payload.get("penalties", {})
    hints = payload.get("purchase_history_hints", {})
    
    # 2. Start by gathering every single product in your graph
    all_articles = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'article']
    valid_items = []
    
    print(f"Scanning {len(all_articles)} total articles in the database...")

    # 3. APPLY FILTERS (The "Must-Haves" / Strict WHERE Clauses)
    for article_id in all_articles:
        
        # Rule 1: Is it on the "Do Not Serve" list?
        if article_id in exclude_ids:
            continue
            
        is_valid = True
        
        # Rule 2: Does it match ALL the strict filters?
        for key, required_value in filters.items():
            
            # Since you built a smart graph, categories (like 'Black' or 'Dress') are their own dots!
            # We just ask NetworkX: "Is there a line connecting this Article to this Filter value?"
            if not G.has_edge(article_id, required_value):
                is_valid = False
                break # It failed a filter, stop checking and throw it out
                
        # If it survived all the filters, it makes it to the next round!
        if is_valid:
            valid_items.append(article_id)
            
    print(f"--> Filters applied! {len(valid_items)} items survived the strict constraints.")
    
    # We will add the Scoring and Ranking here in Part B!
    
    return {"status": "success", "data": valid_items}

def run_attribute_lookup(G, payload):
    print(f"--> [ROUTED TO: item_attribute_lookup] Looking up {payload.get('attribute_topic')} for {payload.get('article_id')}...")
    return {"status": "success", "data": "Placeholder for attribute"}

def run_item_compare(G, payload):
    print(f"--> [ROUTED TO: item_compare] Comparing {payload.get('article_id_a')} and {payload.get('article_id_b')}...")
    return {"status": "success", "data": "Placeholder for comparison"}

def run_explanation_generate(G, payload):
    print(f"--> [ROUTED TO: explanation_generate] Explaining why we picked {payload.get('article_id')}...")
    return {"status": "success", "data": "Placeholder for explanation text"}

def run_item_detail_lookup(G, payload):
    print(f"--> [ROUTED TO: item_detail_lookup] Grabbing all details for {payload.get('article_id')}...")
    return {"status": "success", "data": "Placeholder for all item details"}


# ==========================================
# THE TICKET READER (The Router)
# ==========================================

def handle_retrieval_request(G, retrieval_input):
    """
    This is the main entry point. Member 3 passes the JSON object here.
    """
    print("\n[TICKET RECEIVED] Reading the retrieval input...")

    # 1. Check if we actually need to do anything
    # If it's FEEDBACK or CHITCHAT, Member 3 sends None.
    if retrieval_input is None:
        print("--> [ROUTED TO: NOWHERE] It's just chitchat or feedback. Doing nothing.")
        return None

    # 2. Open the Envelope (Extract standard fields)
    action = retrieval_input.get("action")
    items_in_context = retrieval_input.get("items_in_context", {})
    exclude_ids = retrieval_input.get("exclude_ids", [])
    payload = retrieval_input.get("payload", {})

    print(f"Action requested: {action}")

    # 3. Route to the correct helper function based on the action
    if action == "catalog_search":
        return run_catalog_search(G, payload, items_in_context, exclude_ids)
        
    elif action == "item_attribute_lookup":
        return run_attribute_lookup(G, payload)
        
    elif action == "item_compare":
        return run_item_compare(G, payload)
        
    elif action == "explanation_generate":
        return run_explanation_generate(G, payload)
        
    elif action == "item_detail_lookup":
        return run_item_detail_lookup(G, payload)
        
    else:
        print(f"Error: Unknown action '{action}' received!")
        return {"status": "error", "message": "Unknown action"}


# ==========================================
# TEST THE ROUTER
# ==========================================

if __name__ == "__main__":
    # 1. Build the REAL graph!
    print("System starting up...")
    kg = construct_knowledge_graph() 
    
    # 2. Create a fake "Ticket" exactly like Member 3 will send
    dummy_ticket = {
        "action": "catalog_search",
        "retrieval_strategy": "FULL",
        "user_message": "I want a black dress for summer under £50",
        "items_in_context": {"item_a": None, "item_b": None},
        "exclude_ids": ["108775015"],
        "payload": {
            "filters": {"colour_group_name": "Black", "product_type_name": "Dress"},
            "soft_constraints": {"style": "casual", "occasion": "summer"}
        }
    }

    # 3. Slide the ticket to the reader!
    print("\n--- Testing Catalog Search ---")
    handle_retrieval_request(kg, dummy_ticket)
    
    # 4. Test a different ticket
    print("\n--- Testing Item Compare ---")
    dummy_ticket_2 = {
        "action": "item_compare",
        "payload": {"article_id_a": "123", "article_id_b": "456"}
    }
    handle_retrieval_request(kg, dummy_ticket_2)
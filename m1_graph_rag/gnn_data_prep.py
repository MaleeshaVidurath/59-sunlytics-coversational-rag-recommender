import torch
import networkx as nx
import os
from build_graph import construct_knowledge_graph

# Match the exact path structure so it saves directly in your working folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def prepare_bipartite_data(G):
    print("\n--- Translating NetworkX Graph for PyTorch GNN ---")
    
    # 1. Separate the nodes into two distinct pools
    customers = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'customer']
    articles = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'article']
    
    print(f"Found {len(customers)} Customers and {len(articles)} Articles.")

    # 2. Create the Translation Dictionaries
    # Maps real IDs (e.g., "Cust_A") to PyTorch Math IDs (e.g., 0, 1, 2...)
    user_mapping = {real_id: math_id for math_id, real_id in enumerate(customers)}
    item_mapping = {real_id: math_id for math_id, real_id in enumerate(articles)}
    
    # Create reverse maps so the GNN can translate its answers back to Real IDs later
    reverse_user_mapping = {v: k for k, v in user_mapping.items()}
    reverse_item_mapping = {v: k for k, v in item_mapping.items()}

    # 3. Extract the Purchase History (The Edges)
    source_users = []
    target_items = []
    
    for u, v, data in G.edges(data=True):
        if data.get('relation') == 'BOUGHT':
            # NetworkX edges don't guarantee order, so we must check which node is the customer
            if u in user_mapping:
                source_users.append(user_mapping[u])
                target_items.append(item_mapping[v])
            else:
                source_users.append(user_mapping[v])
                target_items.append(item_mapping[u])

    # 4. Convert to PyTorch Tensors
    # PyTorch needs a 2D array: Row 0 is Users, Row 1 is the Items they bought
    edge_index = torch.tensor([source_users, target_items], dtype=torch.long)
    
    print(f"Successfully extracted {edge_index.shape[1]} purchase connections into a PyTorch tensor.")
    
    # ==========================================
    # 5. SAVE TO DISK (The New Addition)
    # ==========================================
    # We save the math and the translation dictionaries to your hard drive
    # so the GNN can load them instantly during training.
    torch.save(edge_index, os.path.join(BASE_DIR, 'edge_index.pt'))
    torch.save(user_mapping, os.path.join(BASE_DIR, 'user_mapping.pt'))
    torch.save(item_mapping, os.path.join(BASE_DIR, 'item_mapping.pt'))
    torch.save(reverse_item_mapping, os.path.join(BASE_DIR, 'reverse_item_mapping.pt'))
    print(f"Saved graph data to disk at: {BASE_DIR}")
    
    return edge_index, user_mapping, item_mapping, reverse_item_mapping

if __name__ == "__main__":
    # Test the bridge
    kg = construct_knowledge_graph()
    edge_tensor, u_map, i_map, rev_i_map = prepare_bipartite_data(kg)
    
    print("\nSample of the PyTorch Edge Index (First 5 purchases):")
    print("User Math IDs: ", edge_tensor[0][:5].tolist())
    print("Item Math IDs: ", edge_tensor[1][:5].tolist())
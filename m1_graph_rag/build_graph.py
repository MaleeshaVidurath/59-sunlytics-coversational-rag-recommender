import networkx as nx
import pandas as pd

from data_loader import load_sample_data

def construct_knowledge_graph():
    customers_df, articles_df, transactions_df = load_sample_data()
    
    G = nx.Graph()
    print("Building the enriched Knowledge Graph...")

    # --- STEP 1: Add Articles and their Categories ---
    for index, row in articles_df.iterrows():
        article_id = row['article_id']
        
        G.add_node(
            article_id, 
            type='article',
            name=str(row['prod_name']),
            description=str(row['detail_desc'])
        )
        
        product_type = row['product_type_name']
        if pd.notna(product_type):
            G.add_node(product_type, type='product_type')
            G.add_edge(article_id, product_type, relation='BELONGS_TO_TYPE')
            
        colour = row['colour_group_name']
        if pd.notna(colour):
            G.add_node(colour, type='colour')
            G.add_edge(article_id, colour, relation='HAS_COLOUR')

    print(f"Added Articles and Categories. Current Nodes: {G.number_of_nodes()}")

    # --- STEP 2: Add Customers and Purchase History ---
    for index, row in transactions_df.iterrows():
        customer_id = row['customer_id']
        article_id = row['article_id']
        
        # --- NEW: Extract the transaction date ---
        t_dat = row['t_dat']
        price = row['price']
        
        if not G.has_node(customer_id):
            G.add_node(customer_id, type='customer')
        
        if G.has_node(article_id):
            # --- NEW: Attach the date to the edge ---
            G.add_edge(customer_id, article_id, relation='BOUGHT', price=price, date=t_dat)

    print(f"Graph built successfully! Total Nodes: {G.number_of_nodes()}, Total Edges: {G.number_of_edges()}")
    return G

if __name__ == "__main__":
    kg = construct_knowledge_graph()
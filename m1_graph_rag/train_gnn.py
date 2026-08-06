import torch
import os
from gnn_model import FashionGNN

# Set up the folder path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def train_model():
    print("\n--- Starting GNN Training School ---")
    
    # ==========================================
    # 1. LOAD THE PREPARED DATA
    # ==========================================
    print("Loading graph and text math...")
    edge_index = torch.load(os.path.join(BASE_DIR, 'edge_index.pt'))
    text_embeddings = torch.load(os.path.join(BASE_DIR, 'product_embeddings.pt'))
    user_mapping = torch.load(os.path.join(BASE_DIR, 'user_mapping.pt'))
    
    num_users = len(user_mapping)
    num_items = text_embeddings.shape[0]
    
    print(f"Loaded {num_users} Users and {num_items} Items.")
    
    # ==========================================
    # 2. THE ID SHIFT (Crucial Step)
    # ==========================================
    # Because we stacked the Items UNDER the Users in the GNN list, 
    # we have to shift the Item IDs in the connecting lines.
    # E.g., If there are 100 users, Item 0 becomes Node 100 in the giant list.
    train_edges = edge_index.clone()
    train_edges[1] = train_edges[1] + num_users
    
    # ==========================================
    # 3. INITIALIZE THE AI
    # ==========================================
    # Create the brain and the optimizer (the tool that updates the math)
    model = FashionGNN(num_users, num_items, text_embeddings)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # ==========================================
    # 4. THE TRAINING LOOP
    # ==========================================
    epochs = 100  # How many times the AI will review the data
    print("\nBeginning Training (Watch the Loss go down!)...")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad() # Clear out the old math
        
        # 1. Ask the GNN to mix the math
        final_users, final_items = model(train_edges)
        
        # 2. Grab the specific math vectors for the real purchases
        user_vecs = final_users[edge_index[0]]
        item_vecs = final_items[edge_index[1]]
        
        # 3. Score the real purchases (We want these scores to be HIGH)
        positive_scores = (user_vecs * item_vecs).sum(dim=1)
        
        # 4. Generate the "Bad Examples" (Fake purchases)
        random_items = torch.randint(0, num_items, (edge_index.shape[1],))
        neg_item_vecs = final_items[random_items]
        
        # 5. Score the fake purchases (We want these scores to be LOW)
        negative_scores = (user_vecs * neg_item_vecs).sum(dim=1)
        
        # 6. Calculate the "Loss" (How badly the AI messed up)
        # BPR Loss: The AI gets a good grade if positive_score > negative_score
        loss = -torch.log(torch.sigmoid(positive_scores - negative_scores)).mean()
        
        # 7. Update the brain!
        loss.backward()
        optimizer.step()
        
        # Print progress every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")
            
    print("\nTraining Complete! The AI has successfully learned the fashion patterns.")
    
    # ==========================================
    # 5. SAVE THE TRAINED BRAIN
    # ==========================================
    torch.save(model.state_dict(), os.path.join(BASE_DIR, 'trained_gnn.pt'))
    print(f"Saved the fully trained AI to: {BASE_DIR}")

if __name__ == "__main__":
    train_model()
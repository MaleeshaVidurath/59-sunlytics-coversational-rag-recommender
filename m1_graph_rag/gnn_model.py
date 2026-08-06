import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class FashionGNN(nn.Module):
    def __init__(self, num_users, num_items, text_embeddings):
        super(FashionGNN, self).__init__()
        
        # ==========================================
        # 1. THE STARTING MATH
        # ==========================================
        # Users start with 64 random numbers (their "Blank Slate")
        # The AI will learn to adjust these numbers during training.
        self.user_emb = nn.Embedding(num_users, 64)
        
        # Items start with your 384 text numbers (The "Vibe")
        self.item_text_math = nn.Parameter(text_embeddings.clone().detach()) 
        
        # We build a "Compressor" layer. 384 numbers is too bulky for the graph.
        # This shrinks the text meaning down to 64 core numbers so it matches the users.
        self.text_compressor = nn.Linear(384, 64)
        
        # ==========================================
        # 2. THE GRAPH LAYERS (The "Message Passing")
        # ==========================================
        # SAGEConv (GraphSAGE) looks at a dot, grabs the math from all connected dots, 
        # and blends them together. We use two layers so the math can travel further!
        self.conv1 = SAGEConv(64, 64)
        self.conv2 = SAGEConv(64, 64)

    def forward(self, edge_index):
        # --- PREPARATION ---
        # Get the users' current math
        users = self.user_emb.weight
        
        # Compress the items' 384 text numbers into 64 numbers
        items = self.text_compressor(self.item_text_math)
        items = F.relu(items) # ReLU removes any negative noise
        
        # To run the graph, PyTorch needs all dots stacked into one giant list.
        # Users take the top half of the list, Items take the bottom half.
        x = torch.cat([users, items], dim=0)
        
        # --- THE GRAPH MIXER ---
        # Pass the giant list and the connecting lines into the Graph Neural Network
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        x = self.conv2(x, edge_index)
        
        # --- SPLIT THEM BACK APART ---
        # Now that the math is fully mixed, we separate them back into Users and Items
        # so we can easily compare them later to make recommendations!
        final_users = x[:users.size(0)]
        final_items = x[users.size(0):]
        
        return final_users, final_items
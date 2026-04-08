import torch
import open_clip
import numpy as np

class ClipTextEncoder:
    """
    Local CLIP Encoder for translating VLM Search Strings into 512-D Math Vectors.
    Crucially, it must load the EXACT same model weights ('laion2b_s34b_b79k') used on Kaggle.
    """
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading local CLIP Text Encoder on {self.device}...")
        
        try:
            # We only need the text portion, but we load the whole model to ensure parity.
            self.model, _, _ = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='laion2b_s34b_b79k'
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Load the CLIP tokenizer
            self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        except ImportError:
            print("Warning: open_clip_torch is not installed. Please run: pip install open_clip_torch")

    def encode_text(self, text_string: str) -> np.ndarray:
        """
        Converts the VLM's cleaned text query into a 512-dimension normalized vector.
        """
        if not text_string:
            return None
            
        # Tokenize the user's plain English text
        text_tokens = self.tokenizer([text_string]).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            
            # CRITICAL: We used Inner Product (Cosine Similarity) on Kaggle. 
            # We MUST normalize the text vector locally here before querying.
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
        return text_features.cpu().numpy().astype('float32')

# Singleton instantiator
clip_encoder = ClipTextEncoder()

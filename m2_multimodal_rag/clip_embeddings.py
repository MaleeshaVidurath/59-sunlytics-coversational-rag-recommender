import torch
import open_clip
import numpy as np
from PIL import Image

class ClipTextEncoder:
    """
    Local CLIP Encoder for translating VLM Search Strings into 512-D Math Vectors.
    Crucially, it must load the EXACT same model weights ('laion2b_s34b_b79k') used on Kaggle.
    Also supports image encoding for CLIPScore faithfulness scoring.
    """
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.preprocess_val = None
        print(f"Loading local CLIP Text Encoder on {self.device}...")

        try:
            # Load model + image preprocessing transforms
            self.model, _, self.preprocess_val = open_clip.create_model_and_transforms(
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

    def encode_expanded(self, queries: list) -> np.ndarray:
        """
        NOVELTY 1: Multi-Vector CLIP Ensemble.
        Encodes a list of query variants (original + LLM expansions) into individual
        512-D vectors, then returns their element-wise average — re-normalized for
        FAISS inner-product search.  A richer query representation improves recall.
        Paper: RAG-VisualRec — enriching sparse signals into richer representations.
        """
        vectors = []
        for q in queries:
            if not q:
                continue
            v = self.encode_text(q)
            if v is not None:
                vectors.append(v)

        if not vectors:
            return None

        if len(vectors) == 1:
            return vectors[0]

        # Stack → (N, 1, 512), squeeze → (N, 512), mean → (1, 512)
        stacked = np.vstack(vectors)
        avg_vector = np.mean(stacked, axis=0, keepdims=True).astype('float32')

        # Re-normalize so inner-product == cosine similarity (matches FAISS index)
        norm = np.linalg.norm(avg_vector)
        if norm > 0:
            avg_vector = avg_vector / norm

        print(f"   [CLIP Ensemble] Averaged {len(vectors)} query vectors → 1 enriched 512-D vector")
        return avg_vector


    def encode_image(self, image_path: str) -> np.ndarray | None:
        """
        Encodes a product image into a normalised 512-D CLIP vector.
        Used by CLIPScore faithfulness scorer for image-text alignment.
        """
        if self.preprocess_val is None:
            return None
        try:
            with Image.open(image_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img_tensor = self.preprocess_val(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(img_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            return image_features.cpu().numpy().astype("float32")
        except Exception as e:
            print(f"   [CLIP] Image encoding failed: {e}")
            return None


# Singleton instantiator
clip_encoder = ClipTextEncoder()
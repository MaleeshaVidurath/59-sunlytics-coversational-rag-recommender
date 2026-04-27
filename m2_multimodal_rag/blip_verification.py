import torch
import warnings
import os

# Suppress unnecessary HuggingFace and Transformers warnings to keep terminal clean
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.*")

from transformers import BlipProcessor, BlipForImageTextRetrieval, logging as hf_logging
from PIL import Image

# Suppress the verbose 'LOAD REPORT' and 'UNEXPECTED' weight warnings
hf_logging.set_verbosity_error()

class VisualVerifier:
    """
    Mathematical Visual Verification Guard for M2.
    Natively measures if a textual LLM Explanation actually matches the pixels of the target Image.
    """
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"M2 VLM: Initializing Visual Verification Guard on {self.device}...")
        
        # We uniquely use the ImageTextRetrieval BLIP model because it has an ITM (Image-Text Matching) head
        self.model_name = "Salesforce/blip-itm-base-coco"
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForImageTextRetrieval.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def verify(self, image_path: str, llm_explanation: str, threshold: float = 0.5) -> (bool, str):
        """
        Takes raw image bytes and LLM text, and outputs PASS/FAIL based on a threshold score.
        """
        try:
            raw_image = Image.open(image_path).convert('RGB')
        except Exception as e:
            return False, f"Image Load Error: {e}"
            
        inputs = self.processor(raw_image, llm_explanation, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            # Model returns ITM (Image-Text Matching) and ITC (Image-Text Contrastive) scores
            output = self.model(**inputs)
            itm_scores = output.itm_score
            
            # Convert raw logits to a 0.0 -> 1.0 probability of the text matching the image
            match_prob = torch.nn.functional.softmax(itm_scores, dim=1)[:, 1].item()
            
        print(f"   [VLM Guard] Calculated Image-Text Match score: {match_prob:.2f}")
        
        if match_prob >= threshold:
            return True, "Explanation seamlessly matches visual evidence."
        else:
            return False, f"VLM Rejected Explanation! ITM Score {match_prob:.2f} indicates the text contradicts the image pixels."

# Singleton implementation
blip_verifier = VisualVerifier()

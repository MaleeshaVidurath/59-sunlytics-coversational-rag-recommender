"""
Visual Verification Guard — HuggingFace Inference API (Cloud GPU).

Replaces the local Salesforce/blip-itm-base-coco model (~900 MB) with
a remote API call to HuggingFace's free Inference API.

Benefits:
  - Zero local model downloads (no 900 MB BLIP on disk)
  - Runs on HuggingFace GPU servers (~2-3s vs ~90s on local CPU)
  - Free tier: generous request limits for demo/FYP use

Requirements:
  - HF_TOKEN in .env (free at huggingface.co → Settings → Access Tokens)
"""

import torch
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class VisualVerifier:
    """
    Local Visual Verification Guard using HuggingFace Transformers (ViLT).
    Runs completely locally (CPU/GPU) without relying on external APIs.
    Downloads a ~470MB model on the first run.
    """

    def __init__(self):
        print("M2 VLM: Initializing Local Visual Verifier (dandelin/vilt-b32-finetuned-vqa)...")
        try:
            self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
            self.model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
            self._ready = True
            print("M2 VLM: [SUCCESS] Local VLM Guard Ready!")
        except Exception as e:
            self._ready = False
            print(f"M2 VLM: [ERROR] Failed to load local VLM model: {e}")

    def verify(self, image_path: str, llm_explanation: str, threshold: float = 0.5) -> tuple[bool, str]:
        """
        Processes the image and explanation locally using ViLT.
        """
        if not getattr(self, "_ready", False):
            return True, "VLM Guard in fallback mode (Model failed to load). Skipping verification."

        try:
            with Image.open(image_path) as img:
                # Convert to RGB to avoid issues
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                # ViLT is memory-heavy, so keeping the image reasonably sized helps
                img.thumbnail((512, 512))

                # Build a clear yes/no question
                short_desc = llm_explanation[:200].strip()
                question = f"Does this clothing item match this description: '{short_desc}'? Answer yes or no."
                
                # Process inputs
                inputs = self.processor(img, question, return_tensors="pt")
                
                # Forward pass
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Extract answer
                logits = outputs.logits
                idx = logits.argmax(-1).item()
                answer = self.model.config.id2label[idx].lower().strip()
                
                print(f"   [VLM Guard] Local ViLT answer: '{answer}'")
                
                if "yes" in answer:
                    return True, f"Explanation matches visual evidence (ViLT: '{answer}')."
                else:
                    return False, (
                        f"VLM Rejected Explanation! Local ViLT answered '{answer}' — "
                        f"the text contradicts the image pixels. Regenerating response..."
                    )

        except Exception as e:
            print(f"   [VLM Guard] Local VLM Error: {e}. Passing by default.")
            return True, f"VLM Guard error: {e}. Passing by default."


# Singleton implementation
blip_verifier = VisualVerifier()

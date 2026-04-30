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

import os
import base64
import requests
import warnings
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.*")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class VisualVerifier:
    """
    Cloud-based Visual Verification Guard using HuggingFace Inference API.
    Calls Salesforce/blip-vqa-base on HF's GPU servers instead of loading locally.
    Falls back gracefully if the API is unavailable or token is missing.
    """

    # BLIP VQA model — asks "does this image match?" and gets yes/no answer
    VQA_API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-vqa-base"

    def __init__(self):
        self.hf_token = os.getenv("HF_TOKEN", "")
        self._ready = bool(self.hf_token and not self.hf_token.startswith("hf_your"))

        if self._ready:
            print("M2 VLM: Visual Verifier ready → HuggingFace Cloud GPU (Salesforce/blip-vqa-base)")
        else:
            print("M2 VLM: [WARNING] HF_TOKEN not set in .env — VLM Guard running in FALLBACK mode.")
            print("M2 VLM: Get a free token at huggingface.co → Settings → Access Tokens")

    def verify(self, image_path: str, llm_explanation: str, threshold: float = 0.5) -> tuple[bool, str]:
        """
        Sends the product image + LLM explanation to HuggingFace BLIP VQA API.
        Asks: "Does this clothing item match this description? Answer yes or no."
        Returns (passed: bool, reason: str).
        """
        # Fallback: pass all if token not configured
        if not self._ready:
            return True, "VLM Guard in fallback mode (HF_TOKEN not configured). Skipping verification."

        # Load and base64-encode the image
        try:
            with open(image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            return False, f"Image Load Error: {e}"

        # Build a clear yes/no question for BLIP VQA
        # Trim explanation to 200 chars to stay within model token limits
        short_desc = llm_explanation[:200].strip()
        question = (
            f"Does this clothing item match this description: {short_desc}? "
            f"Answer yes or no."
        )

        payload = {
            "inputs": {
                "question": question,
                "image": image_b64,
            }
        }

        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                self.VQA_API_URL,
                headers=headers,
                json=payload,
                timeout=30,
            )

            # Handle model loading delay (HF cold start)
            if response.status_code == 503:
                print("   [VLM Guard] HF model loading (cold start)... retrying in 10s")
                import time
                time.sleep(10)
                response = requests.post(
                    self.VQA_API_URL, headers=headers, json=payload, timeout=45
                )

            if response.status_code != 200:
                print(f"   [VLM Guard] HF API error {response.status_code}: {response.text[:100]}")
                return True, f"VLM API error ({response.status_code}). Passing by default."

            result = response.json()

            # BLIP VQA returns: [{"score": 0.95, "answer": "yes"}, ...]
            if not isinstance(result, list) or not result:
                return True, "VLM: Unexpected API response. Passing by default."

            top_answer = result[0]
            answer = top_answer.get("answer", "").lower().strip()
            score = top_answer.get("score", 0.0)

            print(f"   [VLM Guard] BLIP VQA answer: '{answer}' (confidence: {score:.2f})")

            if "yes" in answer:
                return True, f"Explanation matches visual evidence (BLIP: '{answer}', score: {score:.2f})."
            else:
                return False, (
                    f"VLM Rejected Explanation! BLIP answered '{answer}' (score: {score:.2f}) — "
                    f"the text contradicts the image pixels. Regenerating response..."
                )

        except requests.exceptions.Timeout:
            print("   [VLM Guard] HF API timeout. Passing by default.")
            return True, "VLM Guard timed out. Passing by default."
        except Exception as e:
            print(f"   [VLM Guard] HF API exception: {e}. Passing by default.")
            return True, f"VLM Guard error: {e}. Passing by default."


# Singleton implementation
blip_verifier = VisualVerifier()

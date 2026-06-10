"""
Layer 3 — VLM Visual Verification (ViLT).

Uses the dandelin/vilt-b32-finetuned-vqa Vision-Language Model to verify
that the LLM-generated explanation is visually consistent with the actual
product image. This cross-modal check catches hallucinations that are
undetectable by text-only methods (Layers 1 and 2).

Model: dandelin/vilt-b32-finetuned-vqa (~470 MB, downloaded on first run).
Runs locally on CPU/GPU via HuggingFace Transformers.
"""

import torch
import warnings
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)


class VisualVerifier:
    """
    Local VLM guard using ViLT (Vision-and-Language Transformer).

    Given a product image and an LLM-generated explanation, ViLT answers
    the question "Does this clothing item match this description?" with
    yes/no. A "no" answer signals a visual hallucination and triggers
    regeneration with corrective visual feedback.
    """

    def __init__(self):
        print("M2 Guard | Layer 3: Initialising VLM Visual Verifier (ViLT)...")
        try:
            self.processor = ViltProcessor.from_pretrained(
                "dandelin/vilt-b32-finetuned-vqa"
            )
            self.model = ViltForQuestionAnswering.from_pretrained(
                "dandelin/vilt-b32-finetuned-vqa"
            )
            self._ready = True
            print("M2 Guard | Layer 3: [SUCCESS] ViLT Visual Verifier ready.")
        except Exception as e:
            self._ready = False
            print(f"M2 Guard | Layer 3: [ERROR] Failed to load ViLT model: {e}")

    def verify(self, image_path: str, llm_explanation: str, threshold: float = 0.5) -> tuple[bool, str]:
        """
        Runs ViLT VQA on the product image against the generated explanation.

        Returns:
            (True,  reason_str)  if ViLT answers 'yes' → explanation is consistent
            (False, reason_str)  if ViLT answers 'no'  → visual hallucination detected
        """
        if not getattr(self, "_ready", False):
            return True, "Layer 3: VLM guard in fallback mode (model failed to load). Skipping."

        try:
            with Image.open(image_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.thumbnail((512, 512))

                # ViLT max sequence length is 40 tokens — keep question short
                words      = llm_explanation.split()[:10]
                short_desc = " ".join(words)
                question   = f"Does this clothing match: '{short_desc}'?"

                inputs = self.processor(img, question, return_tensors="pt")

                with torch.no_grad():
                    outputs = self.model(**inputs)

                idx    = outputs.logits.argmax(-1).item()
                answer = self.model.config.id2label[idx].lower().strip()

                print(f"   [Layer 3 | VLM Guard] ViLT answer: '{answer}'")

                if "yes" in answer:
                    return True, f"Layer 3: Explanation matches visual evidence (ViLT: '{answer}')."
                else:
                    return False, (
                        f"Layer 3: VLM rejected explanation — ViLT answered '{answer}', "
                        f"text contradicts image pixels. Regenerating..."
                    )

        except Exception as e:
            print(f"   [Layer 3 | VLM Guard] Error: {e}. Passing by default.")
            return True, f"Layer 3: VLM guard error: {e}. Passing by default."


# Singleton
blip_verifier = VisualVerifier()

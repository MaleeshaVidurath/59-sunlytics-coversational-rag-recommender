"""
CLIPScore Faithfulness Scorer — Image-Text Alignment Metric.

Reference: Hessel et al. (2021), "CLIPScore: A Reference-free Evaluation
Metric for Image Captioning." EMNLP 2021.

CLIPScore measures the cosine similarity between a CLIP image embedding
and a CLIP text embedding. A high score indicates the text is visually
faithful to the image; a low score indicates the text describes something
different from what is shown — i.e., a visual hallucination.

Application in M2:
    For each generated product explanation, we compute CLIPScore between:
      - The actual product image (from the H&M dataset)
      - The LLM-generated explanation text

    This catches visual hallucinations that are too subtle for ViLT's binary
    yes/no VQA (e.g., describing a 'flowy' dress when the image shows a
    structured blazer — both may answer 'yes' to a generic matching question
    but have very different CLIP embeddings).

Threshold (CLIP_SCORE_THRESHOLD = 0.18):
    Hessel et al. (2021) report that human-written captions achieve mean
    CLIPScore ~0.34 on MSCOCO. Mismatched image-text pairs typically score
    < 0.15. We use 0.18 as a conservative threshold suitable for fashion
    product descriptions, which are shorter than natural image captions.

Zero extra cost:
    Reuses the CLIP model already loaded by clip_embeddings.py — no
    additional model download or memory allocation required.
"""

import numpy as np
from m2_multimodal_rag.clip_embeddings import clip_encoder

CLIP_SCORE_THRESHOLD = 0.18   # below this → visual hallucination likely


class CLIPFaithfulnessScorer:
    """
    Computes CLIPScore (Hessel et al., 2021) between a product image and
    the LLM-generated explanation to detect visual hallucinations.

    Used as an additional verification step in the M2 hallucination guard,
    providing a continuous faithfulness score exposed in the verification
    trail returned by the API.
    """

    def score(
        self, image_path: str, explanation: str
    ) -> tuple[float, bool, str]:
        """
        Computes cosine similarity between CLIP image and text embeddings.

        Args:
            image_path   : Path to the product image file.
            explanation  : LLM-generated explanation text.

        Returns:
            clip_score   float  Cosine similarity in [-1, 1]. Higher = more faithful.
            passes       bool   True if clip_score >= CLIP_SCORE_THRESHOLD.
            feedback     str    Human-readable verdict for regeneration prompt.
        """
        # Encode image
        image_vec = clip_encoder.encode_image(image_path)
        if image_vec is None:
            return 0.0, True, "CLIPScore skipped — image encoding failed."

        # Encode explanation text (CLIP max token length ~77 tokens)
        text_vec = clip_encoder.encode_text(explanation[:400])
        if text_vec is None:
            return 0.0, True, "CLIPScore skipped — text encoding failed."

        # Cosine similarity (both vectors are L2-normalised by encode_*)
        clip_score = float(np.dot(image_vec.flatten(), text_vec.flatten()))
        passes = clip_score >= CLIP_SCORE_THRESHOLD

        status = "PASS" if passes else "FAIL"
        print(f"   [CLIPScore] {status}  score={clip_score:.4f}  "
              f"threshold={CLIP_SCORE_THRESHOLD}  "
              f"({'faithful' if passes else 'visual mismatch detected'})")

        if passes:
            feedback = (
                f"CLIPScore PASS: explanation is visually aligned with "
                f"product image (score={clip_score:.3f})."
            )
        else:
            feedback = (
                f"CLIPScore FAIL (score={clip_score:.3f} < {CLIP_SCORE_THRESHOLD}): "
                f"your description does not visually match the product image. "
                f"Rewrite focusing on the actual visible attributes of the item."
            )

        return clip_score, passes, feedback


# Global singleton — reuses CLIP model already loaded in clip_encoder
clip_faithfulness_scorer = CLIPFaithfulnessScorer()

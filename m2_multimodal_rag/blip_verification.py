import numpy as np


class VisualVerifier:
    """
    Visual Verification Guard using CLIP cosine similarity.
    Reuses the already-loaded clip_encoder singleton — no extra model download.
    CLIP was trained on web/product images so it handles white-background
    catalog shots far better than BLIP ITM (which was trained on MS-COCO scenes).
    """

    def verify(self, image_path: str, llm_explanation: str, threshold: float = 0.25) -> tuple[bool, str]:
        """
        Encodes the image and the explanation text with CLIP, then computes
        cosine similarity (dot product of L2-normalised vectors).
        Returns (True, reason) if the similarity exceeds the threshold.
        """
        try:
            from m2_multimodal_rag.clip_embeddings import clip_encoder

            image_vec = clip_encoder.encode_image(image_path)
            text_vec  = clip_encoder.encode_text(llm_explanation[:300].strip())

            if image_vec is None or text_vec is None:
                return True, "CLIP encoding unavailable — skipping verification."

            # Both vectors are already L2-normalised by clip_encoder,
            # so dot product == cosine similarity in [-1, 1].
            score = float(np.dot(image_vec.flatten(), text_vec.flatten()))
            print(f"   [VLM Guard] CLIP similarity score: {score:.3f} (threshold: {threshold})")

            if score >= threshold:
                return True, f"Explanation matches visual evidence (CLIP score: {score:.3f})."
            else:
                return False, (
                    f"VLM Rejected Explanation! CLIP score {score:.3f} < {threshold} — "
                    f"the text contradicts the image pixels. Regenerating response..."
                )

        except Exception as e:
            print(f"   [VLM Guard] CLIP verification error: {e}. Passing by default.")
            return True, f"VLM Guard error: {e}. Passing by default."


# Singleton implementation
blip_verifier = VisualVerifier()

# =============================================================================
# predict.py
# =============================================================================
# Run this after training to classify new, unseen conversation turns.
# This is the script your actual CRS pipeline will call at inference time.
#
# The Predictor class can be imported into your larger system like this:
#
#   from predict import Predictor
#   predictor = Predictor()   # loads the model once
#   result = predictor.predict(history, current_message)
#   if result["retrieval_strategy"] == "FULL":
#       # run full catalog search
#   elif result["retrieval_strategy"] == "PARTIAL":
#       # run metadata lookup
#   else:
#       # skip retrieval entirely
# =============================================================================

import re
import torch
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
)
from config import MODEL_SAVE_DIR, LABEL_NAMES, RETRIEVAL_STRATEGY_MAP, MAX_LEN


def clean_text(text: str) -> str:
    """
    Applies the same cleaning that was applied to the training data.
    Always apply this to any text before passing it to the model,
    so the model sees inputs in the same format it was trained on.
    """
    text = re.sub(r'\*\*', '', text)
    text = re.sub(r' — ', ', ', text)
    text = re.sub(r'—', ', ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def format_input_text(history: list[dict], current_message: str) -> str:
    """
    Formats the conversation history and current message into the same
    [SEP]-joined string format that was used during training.

    Args:
        history: list of dicts like [{"role": "user", "content": "..."},
                                      {"role": "bot",  "content": "..."}]
                 Pass the last 1-3 exchanges (up to 6 turns).
        current_message: the user's latest message to classify.

    Returns:
        A single string ready to feed to the tokenizer.
    """
    # Use only the last 6 turns (= last 3 exchanges) to keep within MAX_LEN
    recent = history[-6:] if len(history) > 6 else history

    parts = []
    for turn in recent:
        role    = turn["role"].upper()
        content = clean_text(turn["content"])
        parts.append(f"{role}: {content}")

    parts.append(f"CURRENT: {clean_text(current_message)}")
    return " [SEP] ".join(parts)


class Predictor:
    """
    Wraps the trained model for easy inference.

    Load it once at application startup, then call predict() for each turn.
    Loading the model takes a few seconds — you don't want to do it on
    every single conversation turn.
    """

    def __init__(self, model_dir: str = MODEL_SAVE_DIR):
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

        print(f"Loading trained model from: {model_dir}")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        self.model     = DistilBertForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded and ready.")

    def predict(
        self,
        history: list[dict],
        current_message: str,
    ) -> dict:
        """
        Classifies one conversation turn.

        Args:
            history:         list of prior turns (role + content dicts)
            current_message: the user's latest message

        Returns a dict with:
            label_id           : integer 0-7
            label_name         : e.g. "ATTRIBUTE_QUESTION"
            retrieval_strategy : "FULL", "PARTIAL", or "NO"
            confidence         : probability of the predicted class (0.0-1.0)
            all_probabilities  : dict of label_name → probability for all 8 classes
        """
        # Format the input exactly as during training
        input_text = format_input_text(history, current_message)

        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Forward pass — no gradient computation needed at inference
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits   # shape: [1, 8]

        # Convert logits to probabilities using softmax
        probs      = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        label_id   = int(probs.argmax())
        confidence = float(probs[label_id])

        return {
            "label_id":           label_id,
            "label_name":         LABEL_NAMES[label_id],
            "retrieval_strategy": RETRIEVAL_STRATEGY_MAP[label_id],
            "confidence":         round(confidence, 4),
            "all_probabilities": {
                LABEL_NAMES[i]: round(float(probs[i]), 4)
                for i in range(len(LABEL_NAMES))
            },
        }


# =============================================================================
# Quick demo — run:  python predict.py
# =============================================================================
if __name__ == "__main__":
    predictor = Predictor()

    # --- Demo conversation 1: INITIAL REQUEST (should return FULL retrieval) ---
    result = predictor.predict(
        history=[],
        current_message="I need a casual dress for a garden party",
    )
    print("\n--- Demo 1: No history, fresh request ---")
    print(f"  Label     : {result['label_name']}")
    print(f"  Strategy  : {result['retrieval_strategy']}")
    print(f"  Confidence: {result['confidence']:.2%}")

    # --- Demo 2: ATTRIBUTE QUESTION (should return PARTIAL retrieval) ---
    history = [
        {"role": "user", "content": "Show me some black dresses"},
        {"role": "bot",  "content": "Here are two options. Option 1 is the Valerie dress (black, dress): Short dress in a crisp cotton weave. Option 2 is the Angel (dark pink, dress): Short A-line dress."},
    ]
    result = predictor.predict(
        history=history,
        current_message="What material is the first one made of?",
    )
    print("\n--- Demo 2: After recommendation, asking about material ---")
    print(f"  Label     : {result['label_name']}")
    print(f"  Strategy  : {result['retrieval_strategy']}")
    print(f"  Confidence: {result['confidence']:.2%}")

    # --- Demo 3: FEEDBACK (should return NO retrieval) ---
    result = predictor.predict(
        history=history,
        current_message="I love it, I'll take the first one",
    )
    print("\n--- Demo 3: Positive feedback ---")
    print(f"  Label     : {result['label_name']}")
    print(f"  Strategy  : {result['retrieval_strategy']}")
    print(f"  Confidence: {result['confidence']:.2%}")

    # --- Demo 4: REFINEMENT (should return FULL retrieval) ---
    result = predictor.predict(
        history=history,
        current_message="Actually can you show me something in white instead?",
    )
    print("\n--- Demo 4: Preference change ---")
    print(f"  Label     : {result['label_name']}")
    print(f"  Strategy  : {result['retrieval_strategy']}")
    print(f"  Confidence: {result['confidence']:.2%}")

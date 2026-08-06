import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("Loading DeBERTa NLI Hallucination Guard...")
MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    nli_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    nli_model.eval()
    print("NLI Hallucination Guard loaded successfully!\n")
except Exception as e:
    print(f"Warning: Could not load DeBERTa NLI model ({e}). Guard running in bypass mode.")
    tokenizer, nli_model = None, None

def verify_against_graph(premise_text: str, generated_text: str, threshold: float = 0.5):
    """
    Cross-references LLM output (hypothesis) against raw graph data (premise).
    Returns (has_hallucination: bool, contradiction_score: float)
    """
    if not nli_model or not tokenizer or not generated_text:
        return False, 0.0

    inputs = tokenizer(
        premise_text, 
        generated_text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512
    )

    with torch.no_grad():
        logits = nli_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    contradiction_score = probs[0].item() # Probability of contradiction

    # WE ONLY CHECK CONTRADICTION NOW.
    # If contradiction is higher than 50% (0.5), it is a hallucination.
    has_hallucination = contradiction_score > threshold

    if has_hallucination:
        print(f"⚠️ [NLI GUARD TRIGGERED] Contradiction Score: {round(contradiction_score, 4)}")
    else:
        print(f"✅ [NLI GUARD PASSED] Contradiction Score: {round(contradiction_score, 4)}")

    return has_hallucination, round(contradiction_score, 4)
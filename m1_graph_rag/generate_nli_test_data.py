import json
import random

def main():
    print("1. Synthesizing NLI Guardrail Test Data...")
    
    # We create Premises (The hard facts from your Knowledge Graph)
    # and Hypotheses (What Llama-3 might generate)
    
    test_cases = []
    
    # --- 1. ENTAILMENT CASES (Accurate LLM Outputs) ---
    accurate_pairs = [
        ("User ➔ HAS_PREFERENCE ➔ black. Item_123 ➔ HAS_COLOUR ➔ black.", "I recommend Item_123 because it matches your preference for black."),
        ("Item_456 ➔ BELONGS_TO_TYPE ➔ sweater.", "This item is a cozy sweater."),
        ("User ➔ BOUGHT ➔ Item_789 (shirt).", "Based on your previous purchase of a shirt, you might like this."),
        ("Item_101 ➔ HAS_COLOUR ➔ red. Item_101 ➔ BELONGS_TO_TYPE ➔ dress.", "This red dress is exactly what you asked for."),
        ("Item_202 ➔ HAS_PATTERN ➔ striped.", "I found this striped option for you.")
    ] * 4 # Multiply to get 20 cases
    
    for i, (premise, hypothesis) in enumerate(accurate_pairs):
        test_cases.append({
            "test_id": f"clean_{i}",
            "premise": premise,
            "hypothesis": hypothesis,
            "ground_truth_label": "entailment" # The Guard SHOULD pass this
        })

    # --- 2. CONTRADICTION CASES (Hallucinations we inject) ---
    hallucination_pairs = [
        ("Item_123 ➔ HAS_COLOUR ➔ black.", "I recommend Item_123 because it is a beautiful bright red color."), # Color hallucination
        ("Item_456 ➔ BELONGS_TO_TYPE ➔ trousers.", "This is a wonderful summer dress."), # Type hallucination
        ("Item_789 ➔ HAS_PRICE ➔ 50.00.", "You should buy this because it is currently on sale for 10.00!"), # Price/Sale hallucination
        ("Item_101 ➔ HAS_COLOUR ➔ blue.", "This yellow top will look great on you."), # Color hallucination
        ("User ➔ HAS_PREFERENCE ➔ cotton.", "I selected this because it is made of 100% pure silk.") # Material hallucination
    ] * 4 # Multiply to get 20 cases
    
    for i, (premise, hypothesis) in enumerate(hallucination_pairs):
        test_cases.append({
            "test_id": f"hallucination_{i}",
            "premise": premise,
            "hypothesis": hypothesis,
            "ground_truth_label": "contradiction" # The Guard MUST block this
        })

    # Shuffle the dataset so they are mixed up
    random.shuffle(test_cases)
    
    with open("nli_test_dataset.json", "w") as f:
        json.dump(test_cases, f, indent=4)
        
    print(f"✅ Success! Created 'nli_test_dataset.json' with {len(test_cases)} test cases (20 Clean, 20 Hallucinations).")

if __name__ == "__main__":
    main()
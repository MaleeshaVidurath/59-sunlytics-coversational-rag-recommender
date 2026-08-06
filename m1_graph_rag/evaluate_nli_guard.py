import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from transformers import pipeline
from tqdm import tqdm
import warnings

# Suppress HuggingFace warnings for cleaner terminal output
warnings.filterwarnings("ignore")

def main():
    print("1. Loading NLI Test Data...")
    with open("nli_test_dataset.json", "r") as f:
        test_cases = json.load(f)
        
    print("2. Booting up DeBERTa NLI Guardrail Model...")
    # Using a fast, standard DeBERTa NLI model to test the pairs
    nli_model_name = "cross-encoder/nli-deberta-v3-small" 
    classifier = pipeline("text-classification", model=nli_model_name)
    
    y_true = []
    y_pred = []
    
    print(f"3. Testing Guardrail against {len(test_cases)} LLM statements...")
    for case in tqdm(test_cases):
        premise = case["premise"]
        hypothesis = case["hypothesis"]
        ground_truth = case["ground_truth_label"] # "entailment" or "contradiction"
        
        # We ask DeBERTa: Does the Premise support the Hypothesis?
        result = classifier(f"{premise} [SEP] {hypothesis}")[0]
        label = result['label'].lower()
        
        # Map the model's output to our binary guardrail decision
        if "contradiction" in label or label == "label_2":
            predicted_label = "contradiction" # Guard BLOCKS the text
        elif "entailment" in label or label == "label_0":
            predicted_label = "entailment"    # Guard ALLOWS the text
        else:
            predicted_label = "contradiction" # Guard BLOCKS if it is unsure (Neutral)
            
        y_true.append(ground_truth)
        y_pred.append(predicted_label)

    print("\n4. Calculating Hallucination Catch Rate...")
    # We measure how well it catches CONTRADICTIONS (the "bad" text)
    precision = precision_score(y_true, y_pred, pos_label="contradiction")
    recall = recall_score(y_true, y_pred, pos_label="contradiction")
    f1 = f1_score(y_true, y_pred, pos_label="contradiction")
    
    print("\n--- NLI GUARDRAIL METRICS ---")
    print(f"Precision (When it blocked, was it right?): {precision:.2f}")
    print(f"Recall (Did it catch ALL the lies?):        {recall:.2f}")
    print(f"F1-Score (Overall Guard Reliability):       {f1:.2f}")
    
    print("\n5. Generating Confusion Matrix Graph for Presentation...")
    cm = confusion_matrix(y_true, y_pred, labels=["entailment", "contradiction"])
    plt.figure(figsize=(8, 6))
    
    # Draw a beautiful heatmap
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", 
                xticklabels=["Allowed (Passed)", "Blocked (Caught)"], 
                yticklabels=["True Fact (Clean)", "Hallucination (Lie)"],
                annot_kws={"size": 16})
                
    plt.title("NLI Guardrail: Hallucination Detection Matrix", pad=20, fontsize=14)
    plt.ylabel("Actual Reality (Ground Truth)", fontsize=12)
    plt.xlabel("DeBERTa Guard's Decision", fontsize=12)
    plt.tight_layout()
    
    plt.savefig("nli_guardrail_evaluation.png", dpi=300)
    print("✅ Done! Graph successfully saved as 'nli_guardrail_evaluation.png'!")

if __name__ == "__main__":
    main()
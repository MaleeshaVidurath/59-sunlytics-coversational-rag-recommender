import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables (Groq API Key)
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def grade_explanation(raw_path, explanation):
    """Uses LLM-as-a-Judge to grade the explanation."""
    prompt = f"""
    You are an expert NLP evaluator grading a Recommender System.
    
    Raw Graph Path: "{raw_path}"
    System's Explanation: "{explanation}"
    
    Grade the System's Explanation on two metrics from 1 to 5:
    1. Faithfulness (1-5): Does the explanation accurately reflect the raw path without adding fake details?
    2. Readability (1-5): Is the text natural, grammatical, and easy for a user to understand?
    
    Respond ONLY in valid JSON format: {{"faithfulness": int, "readability": int}}
    """
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error grading: {e}")
        return {"faithfulness": 0, "readability": 0}

def main():
    print("1. Loading Verbalization Test Dataset...")
    
    # We simulate 10 test cases covering different types of graph paths
    test_cases = [
        {"path": "User ➔ HAS_PREFERENCE ➔ black ➔ MATCHES ➔ Item_1", "text": "I picked this out because it matches your preference for the color black."},
        {"path": "Item_2 ➔ BELONGS_TO_TYPE ➔ sweater", "text": "This is a very comfortable sweater."},
        {"path": "User ➔ BOUGHT ➔ Item_3 ➔ SIMILAR_TO ➔ Item_4", "text": "Since you previously purchased a similar item, I think you'll love this one."},
        {"path": "Item_5 ➔ HAS_COLOUR ➔ red", "text": "This item features a vibrant red color."},
        {"path": "User ➔ HAS_PREFERENCE ➔ cotton ➔ MATCHES ➔ Item_6", "text": "This matches your request for cotton material."},
        {"path": "Item_7 ➔ BELONGS_TO_TYPE ➔ boots", "text": "These are a sturdy pair of boots for your collection."},
        {"path": "Item_8 ➔ HAS_PATTERN ➔ striped", "text": "I noticed you were looking for something striped, so I found this."},
        {"path": "User ➔ SEARCHED ➔ Item_9 ➔ HAS_COLOUR ➔ blue", "text": "Based on your recent search, here is a blue option."},
        {"path": "Item_10 ➔ IS_TRENDING ➔ true", "text": "This item is currently trending in our catalog right now."},
        {"path": "Item_11 ➔ HAS_FIT ➔ oversized", "text": "This piece has a relaxed, oversized fit."}
    ]
    
    results = []
    
    print("2. LLM-as-a-Judge is grading the explanations...")
    for case in tqdm(test_cases):
        grades = grade_explanation(case["path"], case["text"])
        results.append({
            "Raw Path": case["path"],
            "Explanation": case["text"],
            "Faithfulness": grades.get("faithfulness", 0),
            "Readability": grades.get("readability", 0)
        })
        
    print("\n3. Calculating Average Scores...")
    df = pd.DataFrame(results)
    avg_faithfulness = df["Faithfulness"].mean()
    avg_readability = df["Readability"].mean()
    
    print(f"\n--- VERBALIZATION RESULTS ---")
    print(f"Average Faithfulness Score: {avg_faithfulness:.2f} / 5.0")
    print(f"Average Readability Score:  {avg_readability:.2f} / 5.0")
    
    print("\n4. Drawing Bar Chart for Presentation...")
    sns.set_theme(style="whitegrid", context="talk")
    
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=["Faithfulness", "Readability"], y=[avg_faithfulness, avg_readability], palette="Blues_d")
    
    plt.title("Path Verbalization Quality (LLM-as-a-Judge)", pad=15)
    plt.ylim(0, 5.0)
    plt.ylabel("Score (1 to 5)")
    
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 9), 
                    textcoords='offset points', fontsize=12)
        
    plt.savefig("verbalization_evaluation.png", dpi=300)
    print("✅ Done! Graph successfully saved as 'verbalization_evaluation.png'!")

if __name__ == "__main__":
    main()
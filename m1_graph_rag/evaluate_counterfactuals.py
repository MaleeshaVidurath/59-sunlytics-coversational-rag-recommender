import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def grade_transparency(user_query, system_response):
    """Uses LLM-as-a-Judge to grade system transparency."""
    prompt = f"""
    You are evaluating a Recommender System for User Trust and Transparency.
    
    User Query: "{user_query}"
    System Response: "{system_response}"
    
    Grade the system's response on two metrics from 1 to 5:
    1. Explainability (1-5): How well does the system explain its decision-making process?
    2. Transparency (1-5): How clear are the system's boundaries, constraints, and filtering logic?
    
    Respond ONLY in valid JSON format: {{"explainability": int, "transparency": int}}
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
        return {"explainability": 0, "transparency": 0}

def main():
    print("1. Loading A/B Test Scenarios...")
    
    scenarios = [
        {
            "query": "I need a black summer dress.",
            "baseline": "I recommend the Midnight Dress. It is a great match for your preferences.",
            "counterfactual": "I recommend the Midnight Dress because it matches your color preference. I skipped the Obsidian Gown because it is a winter dress, and I excluded the Sunrise Dress because it is yellow."
        },
        {
            "query": "Do you have any oversized cotton shirts?",
            "baseline": "Here is the Cloud Shirt. It is 100% cotton.",
            "counterfactual": "Here is the Cloud Shirt, matching your cotton request. I didn't show the Classic Tee because it has a slim fit, and I skipped the River Blouse because it is made of polyester."
        },
        {
            "query": "Show me some red running shoes.",
            "baseline": "Check out the Speedster Pro.",
            "counterfactual": "Check out the Speedster Pro. I discarded the Trail Blazer because it is a hiking boot, and I removed the Velocity Runner because we are out of the color red."
        },
        {
            "query": "I am looking for a warm winter coat under $100.",
            "baseline": "The Arctic Parka is $89.99 and very warm.",
            "counterfactual": "The Arctic Parka is $89.99. I excluded the Everest Jacket because it costs $150, and I skipped the Autumn Windbreaker because it is not rated for winter temperatures."
        },
        {
            "query": "I want a casual blue hoodie.",
            "baseline": "The Ocean Comfort Hoodie is a great choice.",
            "counterfactual": "The Ocean Comfort Hoodie matches your blue requirement. I skipped the Navy Zip-up because you requested a hoodie, and I discarded the Royal Blazer because it is formal wear."
        }
    ]
    
    results = []
    
    print("2. Simulating User Trust Evaluation...")
    for i, case in enumerate(tqdm(scenarios)):
        # Grade Baseline (System A)
        baseline_grades = grade_transparency(case["query"], case["baseline"])
        results.append({
            "System": "Baseline (Standard RAG)",
            "Explainability": baseline_grades.get("explainability", 0),
            "Transparency": baseline_grades.get("transparency", 0)
        })
        
        # Grade Counterfactual (System B)
        counter_grades = grade_transparency(case["query"], case["counterfactual"])
        results.append({
            "System": "Proposed (Counterfactual RAG)",
            "Explainability": counter_grades.get("explainability", 0),
            "Transparency": counter_grades.get("transparency", 0)
        })

    print("\n3. Calculating Average Scores...")
    df = pd.DataFrame(results)
    summary_df = df.groupby("System").mean().reset_index()
    
    print(f"\n--- COUNTERFACTUAL ENGINE RESULTS ---")
    print(summary_df.to_markdown(index=False))
    
    print("\n4. Drawing Comparison Chart...")
    sns.set_theme(style="whitegrid", context="talk")
    df_melted = summary_df.melt(id_vars="System", var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Metric", y="Score", hue="System", data=df_melted, palette="magma")
    
    plt.title("A/B Test: Impact of Counterfactual Explanations", pad=15)
    plt.ylim(0, 5.0)
    plt.ylabel("Score (1 to 5)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 9), 
                    textcoords='offset points', fontsize=12)
        
    plt.tight_layout()
    plt.savefig("counterfactual_evaluation.png", dpi=300)
    print("✅ Done! Graph successfully saved as 'counterfactual_evaluation.png'!")

if __name__ == "__main__":
    main()
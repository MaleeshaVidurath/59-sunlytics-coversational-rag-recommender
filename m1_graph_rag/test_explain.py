import requests
import json

url = "http://127.0.0.1:8002/api/process"

# Simulating Module 3 asking for an explanation of why the 'Elsa' skirt was recommended
payload = {
    "retrieval_input": {
        "action": "explanation_generate",
        "customer_id": "7f0ac4394297dc4a885d3b9277ba526cbbfbf7fb7cae465b256ed8e55b864f03",
        "user_message": "Why did you recommend the Elsa skirt based on my preferences?",
        "items_in_context": {},
        "exclude_ids": [],
        "payload": {
            "article_id": 803647006,  # Elsa skirt ID
            "matched_prefs": [
                {"attribute_value": "Black"},
                {"attribute_value": "Skirt"}
            ]
        }
    },
    "memory_context": {}
}

print("Sending explanation request to API...")
try:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    data = response.json()
    print("\n✅ SUCCESS! Here is the explanation response:\n")
    print(json.dumps(data, indent=2))
    
except Exception as e:
    print(f"\n❌ Error connecting to API: {e}")
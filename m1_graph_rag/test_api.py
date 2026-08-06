import requests
import json

url = "http://127.0.0.1:8002/api/process"

# Simulating the exact payload Module 3 generates
payload = {
    "retrieval_input": {
        "action": "catalog_search",
        "retrieval_strategy": "FULL",
        "customer_id": "7f0ac4394297dc4a885d3b9277ba526cbbfbf7fb7cae465b256ed8e55b864f03",
        "user_message": "are there any skirts in black?",
        "items_in_context": {"item_a": None, "item_b": None},
        "exclude_ids": ["108775015"],
        "payload": {
            "filters": {
                "colour": "Black"
            },
            "preference_boosts": [],
            "penalties": {}
        }
    },
    "memory_context": {}
}

print("Sending request to M1 Graph RAG API...")
try:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    data = response.json()
    print("\n✅ SUCCESS! Here is the response from your pipeline:\n")
    print(json.dumps(data, indent=2))
    
except Exception as e:
    print(f"\n❌ Error connecting to API: {e}")
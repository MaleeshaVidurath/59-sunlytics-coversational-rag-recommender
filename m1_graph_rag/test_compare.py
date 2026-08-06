import requests
import json

url = "http://127.0.0.1:8002/api/process"

# Simulating Module 3 sending a comparison request for the two skirts
payload = {
    "retrieval_input": {
        "action": "item_compare",
        "customer_id": "7f0ac4394297dc4a885d3b9277ba526cbbfbf7fb7cae465b256ed8e55b864f03",
        "user_message": "What is the difference between the Elsa skirt and the Grace PU skirt?",
        "items_in_context": {
            "item_a": "803647006", # Elsa
            "item_b": "791293002"  # Grace PU
        },
        "payload": {}
    },
    "memory_context": {}
}

print("Sending comparison request to API...")
try:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    data = response.json()
    print("\n✅ SUCCESS! Here is the comparison response:\n")
    print(json.dumps(data, indent=2))
    
except Exception as e:
    print(f"\n❌ Error connecting to API: {e}")
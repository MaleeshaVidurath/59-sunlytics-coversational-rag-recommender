import requests
import json
import time

url = "http://127.0.0.1:8000/api/process"

# This perfectly mimics what the M3 module will send
payload = {
    "retrieval_input": {
        "action": "catalog_search",
        "retrieval_strategy": "FULL",
        "user_message": "I want a party shirt,My girlfriend frock color is red",
        "items_in_context": {},
        "exclude_ids": [],
        "payload": {
            "filters": {
                "colour_group_name": "Red",
                "product_type_name": "Dress"
            },
            "preference_boosts": [
                {"attribute": "colour_group_name", "value": "red", "weight": 0.88}
            ],
            "penalties": {}
        }
    },
    "memory_context": {}
}

print(f"==================================================")
print(f" Testing M2 Backend: {url}")
print(f"==================================================")
print("Sending simulated M3 payload...")

start_time = time.time()
try:
    response = requests.post(url, json=payload)
    end_time = time.time()
    
    print(f"\n[HTTP Status]: {response.status_code}")
    print(f"[Time Taken]: {round(end_time - start_time, 2)} seconds\n")
    
    if response.status_code == 200:
        result = response.json()
        print("--- RESPONSE FROM M2 ---")
        print(f"Action Executed: {result.get('action')}")
        print(f"Text Reply: {result.get('response_text')}")
        print(f"Recommended Items: {len(result.get('items', []))}")
        
        for i, item in enumerate(result.get("items", [])):
            print(f"  [{i+1}] {item.get('prod_name')} (ID: {item.get('article_id')}) -> Image: {item.get('image_url')}")
    else:
        print("ERROR RESPONSE:")
        print(response.text)
        
except requests.exceptions.ConnectionError:
    print("\n[ERROR]: Could not connect to the server. Is uvicorn running on port 8000?")

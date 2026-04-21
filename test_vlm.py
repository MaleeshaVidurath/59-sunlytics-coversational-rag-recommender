import os
from dotenv import load_dotenv

load_dotenv()

from m2_multimodal_rag.query_understanding import QueryUnderstandingVLM
processor = QueryUnderstandingVLM(use_vqa=True)

print('\n--- TEST: TEXT ONLY (Noisy) ---')
noisy_text = 'Hi, I am Maleesha I need party wear shirt for men,my girl friend frock color is red.'
print(f'Input: "{noisy_text}"')
result1 = processor.extract_search_query(text_query=noisy_text)
print(f'Final FAISS Search Query: "{result1}"')

print('\n--- TEST 2: SIMPLE QUERY ---')
simple_text = 'I want a blue denim jacket'
print(f'Input: "{simple_text}"')
result2 = processor.extract_search_query(text_query=simple_text)
print(f'Final FAISS Search Query: "{result2}"')

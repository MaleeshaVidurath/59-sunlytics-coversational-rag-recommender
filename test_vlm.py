import os
from dotenv import load_dotenv

load_dotenv()

from m2_multimodal_rag.query_understanding import QueryUnderstandingVLM
processor = QueryUnderstandingVLM(use_vqa=True)

print('\n--- TEST: TEXT ONLY (Noisy) ---')
noisy_text = 'Hi, I am Maleesha'
print(f'Input: "{noisy_text}"')
result1 = processor.extract_search_query(text_query=noisy_text)
print(f'Final FAISS Search Query: "{result1}"')

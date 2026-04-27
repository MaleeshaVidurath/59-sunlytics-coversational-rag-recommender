import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class QueryUnderstandingVLM:
    """
    Intelligent VLM Pre-Processor for M2 Multimodal RAG.
    Handles noisy Image + Text queries by utilizing BLIP as a VLM.
    Uses Ollama (llama3.1) for text intent extraction, BLIP for all image tasks.
    """
    def __init__(self, use_vqa=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_vqa = use_vqa
        
        # Ollama configuration for text intent extraction
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        self.ollama_available = False
        
        try:
            resp = requests.get(f"{self.ollama_base_url}/api/tags", timeout=3)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                if any(self.ollama_model in m for m in models):
                    self.ollama_available = True
                    print(f"VLM: [SUCCESS] Ollama initialized for intelligent query understanding (Model: {self.ollama_model})")
                else:
                    print(f"VLM: [WARNING] Model '{self.ollama_model}' not found in Ollama. Falling back to BLIP.")
            else:
                print("VLM: [WARNING] Ollama server error. BLIP will be used for all tasks.")
        except requests.exceptions.ConnectionError:
            print("VLM: [WARNING] Ollama not reachable. BLIP will be lazy-loaded when needed.")
        except Exception as e:
            print(f"VLM: [WARNING] Failed to connect to Ollama: {e}. Falling back to BLIP.")
        
        # Initialize placeholders for lazy loading
        self.processor = None
        self.captioning_model = None
        self.vqa_model = None

    def _ensure_blip_loaded(self):
        """Lazy-loads the heavy BLIP models only when absolutely necessary."""
        if self.processor is not None:
            return # Already loaded
            
        print(f"\n[Lazy Load] Bringing BLIP Pre-processor online on {self.device} (This will take a moment)...")
        
        # Primary VLM (from Tech Stack)
        self.captioning_model_name = "Salesforce/blip-image-captioning-base"
        self.processor = BlipProcessor.from_pretrained(self.captioning_model_name)
        self.captioning_model = BlipForConditionalGeneration.from_pretrained(self.captioning_model_name).to(self.device)

        if self.use_vqa:
            # Optional VQA for strict noise-removal tasks
            from transformers import BlipForQuestionAnswering
            self.vqa_model_name = "Salesforce/blip-vqa-base"
            self.vqa_model = BlipForQuestionAnswering.from_pretrained(self.vqa_model_name).to(self.device)
            
        print("[Lazy Load] BLIP models successfully loaded into memory.\n")

    def extract_search_query(self, text_query=None, image_path=None):
        """
        Master function to evaluate the user's messy multimodal input and 
        return a highly-clean string optimized for CLIP encoding.
        Uses Ollama for text intent extraction, BLIP for all image tasks.
        """
        # 1. TEXT ONLY
        if text_query and not image_path:
            clean_query = self._ollama_extract_text_intent(text_query)
            if clean_query:
                # If Ollama detected no fashion intent, return the flag directly
                if clean_query == "IRRELEVANT_QUERY":
                    return "IRRELEVANT_QUERY"
                return clean_query
            return text_query.strip()
            
        # 2. IMAGE ONLY — always use BLIP (llama3.1 can't see images)
        elif image_path and not text_query:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found at {image_path}")
                
            print("VLM: Using BLIP visual captioning for image query...")
            return self._generate_caption(image_path)
            
        # 3. TEXT + IMAGE (Noise Removal Challenge) — BLIP handles image, Ollama not needed
        elif image_path and text_query:
            print("VLM: Analyzing Image+Text using BLIP to isolate core fashion features...")
            
            if self.use_vqa:
                # Use Visual Question Answering to precisely isolate features
                # Prompt tuning optimized for fashion noise-removal
                vqa_prompt = f"Question: Based on this image, {text_query}. What specific fashion item and color should I search for? Answer:"
                clean_query = self._run_vqa(image_path, vqa_prompt)
                return clean_query
            else:
                # Fallback to standard Captioning + Prompt Concat
                # This string will be fed into a fast LLM or directly to CLIP
                caption = self._generate_caption(image_path)
                return f"{text_query}. The reference image shows: {caption}"
                
        return ""

    def _ollama_extract_text_intent(self, text_query: str) -> str:
        """Uses Ollama to extract fashion intent from a text query."""
        if not self.ollama_available:
            return None
        try:
            prompt = (
                f"You are a helpful fashion assistant.\n"
                f"The user said: '{text_query}'\n"
                f"What is the exact clothing item the user is trying to find? Please extract ONLY the item they want, and ignore any other clothing mentioned as background context (like a friend's dress).\n"
                f"Reply with just the keywords (e.g., 'blue denim jacket' or 'party wear shirt men'). If there is no clothing item they want to find, reply exactly with 'IRRELEVANT_QUERY'."
            )
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 30,
                    }
                },
                timeout=30
            )
            if response.status_code == 200:
                result = response.json().get("response", "").strip().replace('\n', ' ')
                if not result:
                    result = "IRRELEVANT_QUERY"
                print(f"VLM (Ollama Text): Cleaned '{text_query}' -> '{result}'")
                return result
            else:
                print(f"   [Ollama API Error] Status {response.status_code}")
        except Exception as e:
            print(f"   [Ollama API Error] {e}")
        return None

    def _generate_caption(self, image_path):
        self._ensure_blip_loaded()
        raw_image = Image.open(image_path).convert('RGB')
        # Instruct BLIP to focus on clothing if possible
        text = "a photograph of clothing showing"
        inputs = self.processor(raw_image, text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.captioning_model.generate(**inputs, max_new_tokens=40)
            
        return self.processor.decode(out[0], skip_special_tokens=True).replace("a photograph of clothing showing", "").strip()

    def _run_vqa(self, image_path, question):
        self._ensure_blip_loaded()
        raw_image = Image.open(image_path).convert('RGB')
        inputs = self.processor(raw_image, question, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.vqa_model.generate(**inputs, max_new_tokens=20)
            
        return self.processor.decode(out[0], skip_special_tokens=True)

# Instantiate a global VLM tool for the M2 Pipeline
vlm_query_processor = QueryUnderstandingVLM(use_vqa=True)

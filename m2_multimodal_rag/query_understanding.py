import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class QueryUnderstandingVLM:
    """
    Intelligent VLM Pre-Processor for M2 Multimodal RAG.
    Handles noisy Image + Text queries by utilizing BLIP as a VLM.
    """
    def __init__(self, use_vqa=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_vqa = use_vqa
        
        self.gemini_client = None
        self.gemini_model_name = "gemini-2.5-flash"
        
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key and api_key != "your_key_here":
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=api_key)
                print(f"VLM: [SUCCESS] Gemini Cloud Vision initialized for intelligent query understanding.")
            except Exception as e:
                print(f"VLM: [WARNING] Failed to initialize Gemini client: {e}. Falling back to local BLIP.")
        else:
            print("VLM: [WARNING] No GEMINI_API_KEY found. BLIP will be lazy-loaded when needed.")
        
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
        Uses Gemini 2.5 Flash for intelligent intent extraction, falling back to BLIP.
        """
        # 1. TEXT ONLY
        if text_query and not image_path:
            clean_query = self._gemini_extract_text_intent(text_query)
            if clean_query:
                # If Gemini detected no fashion intent, return the flag directly
                if clean_query == "IRRELEVANT_QUERY":
                    return "IRRELEVANT_QUERY"
                return clean_query
            return text_query.strip()
            
        # 2. IMAGE ONLY
        elif image_path and not text_query:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found at {image_path}")
            
            clean_query = self._gemini_extract_image_features(image_path)
            if clean_query:
                return clean_query
                
            print("VLM: Falling back to BLIP visual captioning for pure image query...")
            return self._generate_caption(image_path)
            
        # 3. TEXT + IMAGE (Noise Removal Challenge)
        elif image_path and text_query:
            print("VLM: Analyzing Image+Text to isolate core fashion features...")
            
            clean_query = self._gemini_multimodal_intent(image_path, text_query)
            if clean_query:
                return clean_query
                
            print("VLM: Falling back to BLIP for Image+Text analysis...")
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

    def _gemini_extract_text_intent(self, text_query: str) -> str:
        if not self.gemini_client: return None
        try:
            from google.genai import types
            prompt = (
                f"You are a helpful fashion assistant.\n"
                f"The user said: '{text_query}'\n"
                f"What is the exact clothing item the user is trying to find? Please extract ONLY the item they want, and ignore any other clothing mentioned as background context (like a friend's dress).\n"
                f"Reply with just the keywords (e.g., 'blue denim jacket' or 'party wear shirt men'). If there is no clothing item they want to find, reply exactly with 'IRRELEVANT_QUERY'."
            )
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=30)
            )
            if response.text:
                result = response.text.strip().replace('\n', ' ')
                if not result:
                    result = "IRRELEVANT_QUERY"
                print(f"VLM (Gemini Text): Cleaned '{text_query}' -> '{result}'")
                return result
            else:
                print(f"VLM (Gemini Text): Empty response from Gemini. Treating as irrelevant.")
                return "IRRELEVANT_QUERY"
        except Exception as e:
            print(f"   [Gemini API Error] {e}")
        return None

    def _gemini_extract_image_features(self, image_path: str) -> str:
        if not self.gemini_client: return None
        try:
            from PIL import Image
            from google.genai import types
            raw_image = Image.open(image_path).convert('RGB')
            prompt = (
                f"You are a strict fashion feature extractor. "
                f"Look at this image and describe the primary clothing item shown in 3-5 keywords. "
                f"Include the color and the exact item type. "
                f"Output ONLY the keywords separated by spaces. Nothing else. Do not write full sentences."
            )
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model_name,
                contents=[raw_image, prompt],
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=30)
            )
            if response.text:
                result = response.text.strip().replace('\n', ' ')
                print(f"VLM (Gemini Image): Extracted -> '{result}'")
                return result
        except Exception as e:
            print(f"   [Gemini API Error] {e}")
        return None

    def _gemini_multimodal_intent(self, image_path: str, text_query: str) -> str:
        if not self.gemini_client: return None
        try:
            from PIL import Image
            from google.genai import types
            raw_image = Image.open(image_path).convert('RGB')
            prompt = (
                f"You are a strict multimodal fashion intent extractor. "
                f"The user uploaded an image and provided this text: \"{text_query}\".\n"
                f"Synthesize the visual evidence from the image and the user's specific request "
                f"to determine the exact fashion item they are searching for.\n"
                f"For example, if the image shows a blue jacket and the text says 'I want this but in red', output 'red jacket'.\n"
                f"Output ONLY the final core keywords (color, item type, etc.) separated by spaces. "
                f"Do not write full sentences. Ignore conversational filler."
            )
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model_name,
                contents=[raw_image, prompt],
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=30)
            )
            if response.text:
                result = response.text.strip().replace('\n', ' ')
                print(f"VLM (Gemini Multimodal): Cleaned Image + '{text_query}' -> '{result}'")
                return result
        except Exception as e:
            print(f"   [Gemini API Error] {e}")
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

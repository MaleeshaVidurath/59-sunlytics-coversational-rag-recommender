import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os

class QueryUnderstandingVLM:
    """
    Intelligent VLM Pre-Processor for M2 Multimodal RAG.
    Handles noisy Image + Text queries by utilizing BLIP as a VLM.
    """
    def __init__(self, use_vqa=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_vqa = use_vqa
        
        print(f"Loading BLIP Pre-processor on {self.device}...")
        
        # Primary VLM (from Tech Stack)
        self.captioning_model_name = "Salesforce/blip-image-captioning-base"
        self.processor = BlipProcessor.from_pretrained(self.captioning_model_name)
        self.captioning_model = BlipForConditionalGeneration.from_pretrained(self.captioning_model_name).to(self.device)

        if self.use_vqa:
            # Optional VQA for strict noise-removal tasks
            from transformers import BlipForQuestionAnswering
            self.vqa_model_name = "Salesforce/blip-vqa-base"
            self.vqa_model = BlipForQuestionAnswering.from_pretrained(self.vqa_model_name).to(self.device)

    def extract_search_query(self, text_query=None, image_path=None):
        """
        Master function to evaluate the user's messy multimodal input and 
        return a highly-clean string optimized for CLIP encoding.
        """
        # 1. TEXT ONLY
        if text_query and not image_path:
            return text_query.strip()
            
        # 2. IMAGE ONLY
        elif image_path and not text_query:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found at {image_path}")
            print("VLM: Generating visual caption for pure image query...")
            return self._generate_caption(image_path)
            
        # 3. TEXT + IMAGE (Noise Removal Challenge)
        elif image_path and text_query:
            print("VLM: Analyzing Image+Text to isolate core fashion features...")
            
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

    def _generate_caption(self, image_path):
        raw_image = Image.open(image_path).convert('RGB')
        # Instruct BLIP to focus on clothing if possible
        text = "a photograph of clothing showing"
        inputs = self.processor(raw_image, text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.captioning_model.generate(**inputs, max_new_tokens=40)
            
        return self.processor.decode(out[0], skip_special_tokens=True).replace("a photograph of clothing showing", "").strip()

    def _run_vqa(self, image_path, question):
        raw_image = Image.open(image_path).convert('RGB')
        inputs = self.processor(raw_image, question, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.vqa_model.generate(**inputs, max_new_tokens=20)
            
        return self.processor.decode(out[0], skip_special_tokens=True)

# Instantiate a global VLM tool for the M2 Pipeline
vlm_query_processor = QueryUnderstandingVLM(use_vqa=True)

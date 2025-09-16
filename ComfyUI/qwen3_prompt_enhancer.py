import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc
import re

class Qwen3PromptEnhancer:
    """
    A ComfyUI node that uses Qwen3-8B model to enhance and refine prompts for better image generation.
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "Qwen/Qwen3-8B"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_prompt": ("STRING", {
                    "multiline": True,
                    "default": "a beautiful landscape",
                    "tooltip": "The original prompt that needs to be enhanced"
                }),
                "enhancement_type": (["Spatial-Enhanced", "Temporal-Enhanced"], {
                    "default": "Spatial-Enhanced",
                    "tooltip": "Spatial-Enhanced for T2I, Temporal-Enhanced for T2V"
                }),
                "target_person": ("STRING", {
                    "default": "A woman",
                    "tooltip": "Target person description (e.g., 'A woman', 'A man')"
                }),
                "model_path": ("STRING", {
                    "default": "Qwen/Qwen3-8B",
                    "tooltip": "Path to Qwen3-8B model (HuggingFace ID or local path)"
                }),
                "cache_dir": ("STRING", {
                    "default": "/xuanyuan/private/rainyjwang/_MODEL",
                    "tooltip": "Directory to cache downloaded models"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Temperature for generation (lower = more deterministic)"
                }),
                "use_quantization": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use 4-bit quantization to reduce memory usage"
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "original_prompt", "enhancement_info")
    FUNCTION = "enhance_prompt"
    CATEGORY = "text/prompt"
    
    DESCRIPTION = "Uses Qwen3-8B model for Spatial-Enhanced (T2I) and Temporal-Enhanced (T2V) prompt processing."
    
    def load_model(self, model_path, cache_dir, use_quantization=True):
        """Load Qwen3-8B model with optional quantization"""
        try:
            if self.model is not None and self.tokenizer is not None:
                return True
                
            logging.info(f"Loading Qwen3-8B model from: {model_path}")
            
            # Configure quantization for memory efficiency
            if use_quantization and torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            else:
                quantization_config = None
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                cache_dir=cache_dir,
                quantization_config=quantization_config,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            logging.info("Qwen3-8B model loaded successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load Qwen3-8B model: {str(e)}")
            return False
    
    def get_enhancement_prompt(self, enhancement_type, original_prompt, target_person):
        """Generate system prompt based on enhancement type"""
        
        if enhancement_type == "Spatial-Enhanced":
            # T2I processing - extract key elements for image generation
            system_prompt = f"""
You are a professional prompt optimizer. Given a video description instruction, you need to extract key elements from it, 
including human faces, clothing, environments, etc., and arrange them in the form of phrases.

For example, a reference output example: Real photography, a girl, RAW photo, Korean portrait photography, korean style, close-up. 

Ignore descriptions of feet, shoes, and background characters, and add a 'close-up' at the end of your output instead.

Only output the improved instruction - no explanations, notes, or additional content!

The video prompt is: {original_prompt}"""
            
        elif enhancement_type == "Temporal-Enhanced":
            # T2V processing - focus on character details and actions
            system_prompt = f"""
You are a professional prompt optimizer. Your task is:

1. Rewrite the user-provided prompt to start with {target_person}, focus on the detailed description of the main characters, including facial details, clothing, actions, etc., weaken the background description, but do not omit information.

2. Ensure the logic of the expression, you can add some subtle reasonable facial changes.

3. Only output the improved instruction - no explanations, notes, or additional content.

Original prompt: {original_prompt}"""
        
        return system_prompt
    
    def clean_response(self, response, enhancement_type, target_person):
        """Clean and filter the model response"""
        cleaned = response.strip()
        
        # Remove thinking tags if present
        cleaned = cleaned.replace("</think>\n\n", "")
        
        if enhancement_type == "Spatial-Enhanced":
            # For T2I: ensure close-up is added and add prefix
            if "close-up" not in cleaned:
                cleaned += ", close-up"
            # Add the special prefix as in original code
            cleaned = "fcsks fxhks fhyks, " + cleaned
            
        elif enhancement_type == "Temporal-Enhanced":
            # For T2V: ensure it starts with target person
            if not cleaned.startswith(target_person):
                target_index = cleaned.find(target_person)
                if target_index != -1:
                    cleaned = cleaned[target_index:]
        
        return cleaned
    
    def enhance_prompt(self, original_prompt, enhancement_type="Spatial-Enhanced", target_person="A woman", 
                      model_path="Qwen/Qwen3-8B", cache_dir="/xuanyuan/private/rainyjwang/_MODEL", 
                      temperature=0.1, use_quantization=True):
        """Main function to enhance prompts using Qwen3-8B"""
        
        try:
            # Load model if not already loaded
            if not self.load_model(model_path, cache_dir, use_quantization):
                return (original_prompt, original_prompt, "Error: Failed to load model")
            
            # Prepare the enhancement prompt
            system_prompt = self.get_enhancement_prompt(enhancement_type, original_prompt, target_person)
            
            # Format for chat (simplified single message approach)
            messages = [{"role": "user", "content": system_prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,  # Disable thinking mode prefix
                enable_thinking=False         # Disable thinking process
            )
            
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            
            # Generate enhanced prompt with strict constraints
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=1000,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode response
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            
            # Clean the response
            enhanced_prompt = self.clean_response(response, enhancement_type, target_person)
            
            # Create info string
            info = f"Enhanced using {enhancement_type} | Target: {target_person} | Temperature: {temperature}"
            
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return (enhanced_prompt, original_prompt, info)
            
        except Exception as e:
            error_msg = f"Error enhancing prompt: {str(e)}"
            logging.error(error_msg)
            return (original_prompt, original_prompt, error_msg)
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "Qwen3PromptEnhancer": Qwen3PromptEnhancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3PromptEnhancer": "Qwen3-8B Prompt Enhancer (IPVG)",
}

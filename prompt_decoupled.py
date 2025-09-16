import argparse
import os
import numpy as np
from PIL import Image
from typing import Optional, Tuple
from insightface.app import FaceAnalysis
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from prompt_optimizer import improve_prompt_t2i, improve_prompt_i2v

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Configuration constants
CONFIG = {
    'model_name': "Qwen/Qwen3-8B",
    'target_image_size': (224, 224),
    'face_detection_size': (224, 224),
    'id_range_start': 1,
    'id_range_end': 5,
    'prompts_per_id': 5,
}


def predict_gender(face_image: Image.Image, app: FaceAnalysis, 
                  target_size: Tuple[int, int] = (224, 224)) -> Optional[str]:
    if face_image.size != target_size:
        face_image = face_image.resize(target_size, Image.LANCZOS)
    img = np.array(face_image)[:, :, ::-1]  # Convert RGB to BGR
    
    # Perform face detection and gender prediction
    results = app.get(img)
    if not results:
        return None
    
    # Parse results - InsightFace returns 0 for female, 1 for male
    face = results[0]
    gender = "A woman" if face.gender == 0 else "A man"
    return gender


def setup_model(model_name: str, cache_dir: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Initialize the language model and tokenizer.
    
    Args:
        model_name: Name/path of the model to load
        cache_dir: Directory to cache model files
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        cache_dir=cache_dir
    )
    print("Model loaded successfully!")
    return model, tokenizer


def process_prompts_i2v(base_dir: str, result_dir: str, app: FaceAnalysis, 
                        model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                        id_start: int = 1, id_end: int = 6, prompts_per_id: int = 5) -> None:
    """Process all prompt files for I2V optimization."""
    id_range = range(id_start, id_end + 1)
    total_ids = len(id_range)
    total_prompts = total_ids * prompts_per_id
    completed_prompts = 0
    
    os.makedirs(result_dir, exist_ok=True)
    
    with tqdm(total=total_prompts, desc="Processing I2V prompts", unit="prompt") as pbar:
        for idx in id_range:
            image_path = os.path.join(base_dir, f"id{idx:03d}", "image.png")
            
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                pbar.update(prompts_per_id)
                continue
            
            try:
                face_image = Image.open(image_path).convert("RGB")
                target_gender = predict_gender(face_image, app)
                    
                if target_gender is None:
                    print(f"Warning: No face detected in {image_path}")
                    target_gender = "A person"
                    
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                pbar.update(prompts_per_id)
                continue
            
            output_id_dir = os.path.join(result_dir, f"id{idx:03d}")
            os.makedirs(output_id_dir, exist_ok=True)
            
            for j in range(1, prompts_per_id + 1):
                txt_path = os.path.join(base_dir, f"id{idx:03d}", f"prompt{j}.txt")
                result_path = os.path.join(output_id_dir, f"prompt{j}.txt")
                
                try:
                    if not os.path.exists(txt_path):
                        print(f"Warning: Prompt file not found: {txt_path}")
                        pbar.update(1)
                        continue
                        
                    with open(txt_path, 'r', encoding='utf-8') as file:
                        original_prompt = file.read().strip()
                    
                    if not original_prompt:
                        print(f"Warning: Empty prompt in {txt_path}")
                        pbar.update(1)
                        continue
                    
                    improved_prompt = improve_prompt_i2v(original_prompt, target_gender, model, tokenizer)
                    
                    if not improved_prompt:
                        print(f"Warning: No improved prompt generated for {result_path}")
                        improved_prompt = original_prompt
                    
                    with open(result_path, 'w', encoding='utf-8') as file:
                        file.write(improved_prompt)
                    
                except Exception as e:
                    print(f"Error processing {txt_path}: {e}")
                finally:
                    completed_prompts += 1
                    pbar.update(1)
                    pbar.set_postfix({"Completed": f"{completed_prompts}/{total_prompts}"})


def process_prompts_t2i(base_dir: str, result_dir: str, app: FaceAnalysis, 
                       model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                       id_start: int = 1, id_end: int = 6, prompts_per_id: int = 5,
                       original_base_dir: str = None) -> None:
    """Process all prompt files for T2I optimization using I2V results as input."""
    id_range = range(id_start, id_end + 1)
    total_ids = len(id_range)
    total_prompts = total_ids * prompts_per_id
    completed_prompts = 0
    
    os.makedirs(result_dir, exist_ok=True)
    
    with tqdm(total=total_prompts, desc="Processing T2I prompts", unit="prompt") as pbar:
        for idx in id_range:
            # Use original base directory for gender detection
            original_image_path = os.path.join(original_base_dir, f"id{idx:03d}", "image.png")
            
            if not os.path.exists(original_image_path):
                print(f"Warning: Original image not found: {original_image_path}")
                pbar.update(prompts_per_id)
                continue
            
            try:
                face_image = Image.open(original_image_path).convert("RGB")
                target_gender = predict_gender(face_image, app)
                    
                if target_gender is None:
                    print(f"Warning: No face detected in {original_image_path}")
                    target_gender = "A person"
                    
            except Exception as e:
                print(f"Error processing image {original_image_path}: {e}")
                pbar.update(prompts_per_id)
                continue
            
            output_id_dir = os.path.join(result_dir, f"id{idx:03d}")
            os.makedirs(output_id_dir, exist_ok=True)
            
            for j in range(1, prompts_per_id + 1):
                # Read from I2V results
                txt_path = os.path.join(base_dir, f"id{idx:03d}", f"prompt{j}.txt")
                result_path = os.path.join(output_id_dir, f"prompt{j}.txt")
                
                try:
                    if not os.path.exists(txt_path):
                        print(f"Warning: I2V prompt file not found: {txt_path}")
                        pbar.update(1)
                        continue
                        
                    with open(txt_path, 'r', encoding='utf-8') as file:
                        i2v_prompt = file.read().strip()
                    
                    if not i2v_prompt:
                        print(f"Warning: Empty I2V prompt in {txt_path}")
                        pbar.update(1)
                        continue
                    
                    # Optimize I2V prompt for T2I
                    improved_prompt = improve_prompt_t2i(i2v_prompt, target_gender, model, tokenizer)
                    
                    if not improved_prompt:
                        print(f"Warning: No improved prompt generated for {result_path}")
                        improved_prompt = i2v_prompt
                    
                    with open(result_path, 'w', encoding='utf-8') as file:
                        file.write(improved_prompt)
                    
                except Exception as e:
                    print(f"Error processing {txt_path}: {e}")
                finally:
                    completed_prompts += 1
                    pbar.update(1)
                    pbar.set_postfix({"Completed": f"{completed_prompts}/{total_prompts}"})


def process_prompts(base_dir: str, result_dir: str, app: FaceAnalysis, 
                   model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                   id_start: int = 1, id_end: int = 6, prompts_per_id: int = 5) -> None:
    """
    Main processing function that handles both I2V and T2I optimization sequentially.
    
    Args:
        base_dir: Input directory containing raw data
        result_dir: Base output directory (will create i2v/ and t2i/ subdirectories)
        app: InsightFace FaceAnalysis instance
        model: Language model for prompt optimization
        tokenizer: Tokenizer for the language model
        id_start: Starting ID number (inclusive)
        id_end: Ending ID number (inclusive)
        prompts_per_id: Number of prompts per identity
    """
    # Define output directories
    i2v_dir = os.path.join(result_dir, "i2v")
    t2i_dir = os.path.join(result_dir, "t2i")
    
    print("=" * 60)
    print("STEP 1: Processing I2V prompts...")
    print("=" * 60)
    
    # Step 1: Process I2V prompts
    process_prompts_i2v(
        base_dir=base_dir,
        result_dir=i2v_dir,
        app=app,
        model=model,
        tokenizer=tokenizer,
        id_start=id_start,
        id_end=id_end,
        prompts_per_id=prompts_per_id
    )
    
    print("\n" + "=" * 60)
    print("STEP 2: Processing T2I prompts using I2V results...")
    print("=" * 60)
    
    # Step 2: Process T2I prompts using I2V results as input
    process_prompts_t2i(
        base_dir=i2v_dir,
        result_dir=t2i_dir,
        app=app,
        model=model,
        tokenizer=tokenizer,
        id_start=id_start,
        id_end=id_end,
        prompts_per_id=prompts_per_id,
        original_base_dir=base_dir
    )


def main():
    """Main function to parse arguments and run decoupled I2V -> T2I prompt optimization."""
    parser = argparse.ArgumentParser(description='Decoupled prompt optimization: I2V -> T2I')
    parser.add_argument('--input_dir', type=str, 
                       default="./vip200k/raw/",
                       help='Input directory containing raw data')
    parser.add_argument('--output_dir', type=str,
                       default="./vip200k/",
                       help='Base output directory (will create i2v/ and t2i/ subdirectories)')
    parser.add_argument('--model_name', type=str,
                       default=CONFIG['model_name'],
                       help='Name/path of the language model to use')
    parser.add_argument('--cache_dir', type=str,
                       default="path/to/model",
                       help='Directory to cache model files')
    parser.add_argument('--id_start', type=int,
                       default=CONFIG['id_range_start'],
                       help='Starting ID number (inclusive)')
    parser.add_argument('--id_end', type=int,
                       default=CONFIG['id_range_end'],
                       help='Ending ID number (inclusive)')
    parser.add_argument('--prompts_per_id', type=int,
                       default=CONFIG['prompts_per_id'],
                       help='Number of prompts per identity')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("DECOUPLED PROMPT OPTIMIZATION: I2V -> T2I")
    print("=" * 80)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"  -> I2V results: {os.path.join(args.output_dir, 'i2v')}")
    print(f"  -> T2I results: {os.path.join(args.output_dir, 't2i')}")
    print(f"Processing IDs: {args.id_start} to {args.id_end}")
    print(f"Prompts per ID: {args.prompts_per_id}")
    print(f"Model: {args.model_name}")
    print("=" * 80)
    
    # Initialize face analysis
    print("Initializing face analysis...")
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=CONFIG['face_detection_size'])
    
    # Initialize language model
    model, tokenizer = setup_model(args.model_name, args.cache_dir)
    
    # Process all prompts with decoupled approach
    process_prompts(
        base_dir=args.input_dir,
        result_dir=args.output_dir,
        app=app,
        model=model,
        tokenizer=tokenizer,
        id_start=args.id_start,
        id_end=args.id_end,
        prompts_per_id=args.prompts_per_id
    )
    
    print("\n" + "=" * 80)
    print("✓ Decoupled prompt optimization completed!")
    print(f"✓ I2V results saved in: {os.path.join(args.output_dir, 'i2v')}")
    print(f"✓ T2I results saved in: {os.path.join(args.output_dir, 't2i')}")
    print("=" * 80)


if __name__ == "__main__":
    main()

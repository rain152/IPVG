import os
import random
import sys
import argparse
from typing import Sequence, Mapping, Any, Union, Optional, Tuple
import torch
import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis

CONFIG = {
    'model_name': "RealVisXL_v4_BakedVAE.safetensors",
    'hyper_lora_model': "sdxl_hyper_id_lora_v1_fidelity",
    'face_detector_model': "bbox/face_yolov8m.pt",
    'image_size': {'width': 832, 'height': 480},
    'generation_params': {
        'steps': 20,
        'cfg': 5,
        'sampler_name': "dpmpp_2m",
        'scheduler': "karras",
        'denoise': 1,
    },
    'face_detail_params': {
        'steps': 10,
        'cfg': 5,
        'denoise': 0.5,
    },
    'similarity_threshold': 0.5,
    'max_generation_attempts': 3,
    'lora_weight': 0.85,
    'negative_prompt': "blurry, no face, big breast, nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, human artifacts, signature, watermark, bad feet"
}

face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(224, 224))

def get_arc_feature(img) -> Optional[torch.Tensor]:
    """Extract ArcFace features for face similarity comparison."""
    if isinstance(img, Image.Image):
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Detect faces in the image
    faces = face_app.get(img)
    if not faces:
        print(f"Warning: No face detected in image")
        return None
    
    feature_tensor = torch.tensor(faces[0].embedding, dtype=torch.float32)
    return feature_tensor

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Get value at index, supports 'result' key for dicts."""
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> Optional[str]:
    """Recursively search for file or directory in parent directories."""
    if path is None:
        path = os.getcwd()
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name
    parent = os.path.dirname(path)
    if parent == path:
        return None
    return find_path(name, parent)

def add_comfyui_directory_to_sys_path() -> None:
    """Add ComfyUI directory to system path."""
    comfyui_path = find_path("ComfyUI")
    if comfyui_path and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")

def add_extra_model_paths() -> None:
    """Load extra model paths from config file."""
    try:
        from main import load_extra_path_config
    except ImportError:
        print("Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead.")
        try:
            from utils.extra_config import load_extra_path_config
        except ImportError:
            print("Could not import load_extra_path_config from utils.extra_config")
            return
    
    extra_model_paths = find_path("extra_model_paths.yaml")
    if extra_model_paths:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")

# Setup ComfyUI environment
add_comfyui_directory_to_sys_path()
add_extra_model_paths()

def import_custom_nodes() -> None:
    """Initialize ComfyUI custom nodes."""
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)
    init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS


def setup_models():
    """Initialize all required models and nodes."""
    nodes = {}
    
    # Initialize all node classes
    nodes['checkpoint_loader'] = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
    nodes['hyper_lora_config'] = NODE_CLASS_MAPPINGS["HyperLoRAConfig"]()
    nodes['load_image'] = NODE_CLASS_MAPPINGS["LoadImage"]()
    nodes['clip_set_last_layer'] = NODE_CLASS_MAPPINGS["CLIPSetLastLayer"]()
    nodes['clip_text_encode'] = NODE_CLASS_MAPPINGS["BNK_CLIPTextEncodeAdvanced"]()
    nodes['hyper_lora_loader'] = NODE_CLASS_MAPPINGS["HyperLoRALoader"]()
    nodes['make_image_batch'] = NODE_CLASS_MAPPINGS["ImpactMakeImageBatch"]()
    nodes['hyper_lora_face_attr'] = NODE_CLASS_MAPPINGS["HyperLoRAFaceAttr"]()
    nodes['hyper_lora_id_cond'] = NODE_CLASS_MAPPINGS["HyperLoRAIDCond"]()
    nodes['hyper_lora_generate_id'] = NODE_CLASS_MAPPINGS["HyperLoRAGenerateIDLoRA"]()
    nodes['hyper_lora_apply'] = NODE_CLASS_MAPPINGS["HyperLoRAApplyLoRA"]()
    nodes['ksampler'] = NODE_CLASS_MAPPINGS["KSampler"]()
    nodes['vae_decode'] = NODE_CLASS_MAPPINGS["VAEDecode"]()
    nodes['face_detailer'] = NODE_CLASS_MAPPINGS["FaceDetailer"]()
    nodes['empty_latent'] = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
    nodes['detector_provider'] = NODE_CLASS_MAPPINGS["UltralyticsDetectorProvider"]()
    
    return nodes


def initialize_static_components(nodes):
    """Initialize components that don't change between generations."""
    checkpoint_result = nodes['checkpoint_loader'].load_checkpoint(
        ckpt_name=CONFIG['model_name']
    )
    
    hyper_config = nodes['hyper_lora_config'].execute(
        image_processor="clip_vit_large_14_processor",
        image_encoder="clip_vit_large_14",
        encoder_types="clip + arcface",
        face_analyzer="antelopev2",
        id_embed_dim=512,
        num_id_tokens=16,
        hyper_dim=128,
        lora_rank=8,
        has_base_lora=False,
    )
    
    clip_result = nodes['clip_set_last_layer'].set_last_layer(
        stop_at_clip_layer=-2, 
        clip=get_value_at_index(checkpoint_result, 1)
    )
    
    latent_result = nodes['empty_latent'].generate(
        width=CONFIG['image_size']['width'], 
        height=CONFIG['image_size']['height'], 
        batch_size=1
    )
    
    detector_result = nodes['detector_provider'].doit(
        model_name=CONFIG['face_detector_model']
    )
    
    hyper_lora_result = nodes['hyper_lora_loader'].execute(
        model=CONFIG['hyper_lora_model'],
        dtype="fp16",
        config=get_value_at_index(hyper_config, 0),
    )
    
    negative_prompt = nodes['clip_text_encode'].encode(
        text=CONFIG['negative_prompt'],
        token_normalization="length+mean",
        weight_interpretation="A1111",
        clip=get_value_at_index(clip_result, 0),
    )
    
    return {
        'checkpoint': checkpoint_result,
        'clip': clip_result,
        'latent': latent_result,
        'detector': detector_result,
        'hyper_lora': hyper_lora_result,
        'negative_prompt': negative_prompt,
    }


def generate_image_with_similarity_check(nodes, static_components, image_path, prompt):
    """Generate image and return the one with highest face similarity."""
    image_result = nodes['load_image'].load_image(image=image_path)
    
    positive_prompt = nodes['clip_text_encode'].encode(
        text=prompt,
        token_normalization="length+mean",
        weight_interpretation="A1111",
        clip=get_value_at_index(static_components['clip'], 0),
    )
    
    image_batch = nodes['make_image_batch'].doit(
        image1=get_value_at_index(image_result, 0)
    )
    
    face_attr = nodes['hyper_lora_face_attr'].execute(
        hyper_lora=get_value_at_index(static_components['hyper_lora'], 0),
        images=get_value_at_index(image_batch, 0),
    )
    
    id_cond = nodes['hyper_lora_id_cond'].execute(
        grayscale=False,
        remove_background=True,
        hyper_lora=get_value_at_index(static_components['hyper_lora'], 0),
        images=get_value_at_index(image_batch, 0),
        face_attr=get_value_at_index(face_attr, 0),
    )
    
    id_lora = nodes['hyper_lora_generate_id'].execute(
        hyper_lora=get_value_at_index(static_components['hyper_lora'], 0),
        id_cond=get_value_at_index(id_cond, 0),
    )
    
    model_with_lora = nodes['hyper_lora_apply'].execute(
        weight=CONFIG['lora_weight'],
        model=get_value_at_index(static_components['checkpoint'], 0),
        lora=get_value_at_index(id_lora, 0),
    )
    
    ref_image = Image.open(image_path)
    ref_feature = get_arc_feature(ref_image)
    
    best_similarity = -1
    best_image = None
    
    for attempt in range(CONFIG['max_generation_attempts']):
        try:
            sample_result = nodes['ksampler'].sample(
                seed=random.randint(1, 2**64),
                steps=CONFIG['generation_params']['steps'],
                cfg=CONFIG['generation_params']['cfg'],
                sampler_name=CONFIG['generation_params']['sampler_name'],
                scheduler=CONFIG['generation_params']['scheduler'],
                denoise=CONFIG['generation_params']['denoise'],
                model=get_value_at_index(model_with_lora, 0),
                positive=get_value_at_index(positive_prompt, 0),
                negative=get_value_at_index(static_components['negative_prompt'], 0),
                latent_image=get_value_at_index(static_components['latent'], 0),
            )
            
            decoded_result = nodes['vae_decode'].decode(
                samples=get_value_at_index(sample_result, 0),
                vae=get_value_at_index(static_components['checkpoint'], 2),
            )
            
            detailed_result = nodes['face_detailer'].doit(
                guide_size=1024,
                guide_size_for=True,
                max_size=1024,
                seed=random.randint(1, 2**64),
                steps=CONFIG['face_detail_params']['steps'],
                cfg=CONFIG['face_detail_params']['cfg'],
                sampler_name=CONFIG['generation_params']['sampler_name'],
                scheduler=CONFIG['generation_params']['scheduler'],
                denoise=CONFIG['face_detail_params']['denoise'],
                feather=5,
                noise_mask=True,
                force_inpaint=True,
                bbox_threshold=0.5,
                bbox_dilation=10,
                bbox_crop_factor=2,
                sam_detection_hint="center-1",
                sam_dilation=0,
                sam_threshold=0.93,
                sam_bbox_expansion=0,
                sam_mask_hint_threshold=0.7,
                sam_mask_hint_use_negative="False",
                drop_size=10,
                wildcard="",
                cycle=1,
                inpaint_model=False,
                noise_mask_feather=0,
                tiled_encode=False,
                tiled_decode=False,
                image=get_value_at_index(decoded_result, 0),
                model=get_value_at_index(model_with_lora, 0),
                clip=get_value_at_index(static_components['clip'], 0),
                vae=get_value_at_index(static_components['checkpoint'], 2),
                positive=get_value_at_index(positive_prompt, 0),
                negative=get_value_at_index(static_components['negative_prompt'], 0),
                bbox_detector=get_value_at_index(static_components['detector'], 0),
            )
            
            image_tensor = detailed_result[0].squeeze(0)
            image_np = image_tensor.cpu().numpy()
            image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
            generated_image = Image.fromarray(image_np)
            
            gen_feature = get_arc_feature(generated_image)
            if ref_feature is not None and gen_feature is not None:
                similarity = torch.cosine_similarity(
                    ref_feature.flatten(), 
                    gen_feature.flatten(), 
                    dim=0
                ).item()
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_image = generated_image
                
                if best_similarity > CONFIG['similarity_threshold']:
                    break
            else:
                best_image = generated_image
                
        except Exception as e:
            print(f"Error during generation attempt {attempt + 1}: {e}")
            continue
    
    return best_image or generated_image, best_similarity


def main():
    """Main function to process images and generate results."""
    parser = argparse.ArgumentParser(description='Generate images using HyperLoRA with ComfyUI')
    parser.add_argument('--input_dir', type=str, 
                       default="path/to/vip200k", 
                       help='Path to the input image directory')
    parser.add_argument('--output_dir', type=str,
                       default="path/to/vip200k", 
                       help='Path to save the output image directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    processed_count = 0
    skipped_count = 0
    
    import_custom_nodes()
    
    with torch.inference_mode():
        print("Initializing models...")
        nodes = setup_models()
        static_components = initialize_static_components(nodes)
        print("Models initialized successfully!")
        
        for i in range(1, 6):
            identity_output_dir = os.path.join(args.output_dir, "i2v", f"id{i:03d}")
            os.makedirs(identity_output_dir, exist_ok=True)
            
            for j in range(1, 6):
                image_path = os.path.join(args.input_dir, "raw", f"id{i:03d}", "image.png")
                prompt_path = os.path.join(args.input_dir, "t2i", f"id{i:03d}", f"prompt{j}.txt")
                output_path = os.path.join(identity_output_dir, f"prompt{j}.jpg")
                
                if not os.path.exists(image_path):
                    print(f"Warning: Image file not found: {image_path}")
                    skipped_count += 1
                    continue
                    
                if not os.path.exists(prompt_path):
                    print(f"Warning: Prompt file not found: {prompt_path}")
                    skipped_count += 1
                    continue
                
                if os.path.exists(output_path):
                    print(f"Skipping existing file: {output_path}")
                    skipped_count += 1
                    continue
                
                try:
                    with open(prompt_path, 'r', encoding='utf-8') as f:
                        prompt = f.read().strip()
                    
                    print(f"Processing id{i:03d}/prompt{j}...")
                    
                    best_image, similarity = generate_image_with_similarity_check(
                        nodes, static_components, image_path, prompt
                    )
                    
                    if best_image:
                        best_image.save(output_path, quality=95)
                        print(f"✓ Saved: {output_path} (similarity: {similarity:.4f})")
                        processed_count += 1
                    else:
                        print(f"✗ Failed to generate image for id{i:03d}/prompt{j}")
                        skipped_count += 1
                        
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Error processing id{i:03d}/prompt{j}: {e}")
                    skipped_count += 1
                    continue
    
    print(f"\nProcessing completed: {processed_count} images generated, {skipped_count} skipped")


if __name__ == "__main__":
    main()
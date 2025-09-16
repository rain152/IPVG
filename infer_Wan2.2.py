#!/usr/bin/env python3
"""
Batch Inference Script - Optimized Version
Iterate through each data entry in CSV file and call Wan2.2 inference command for each entry
"""
import json
import subprocess
import os
import sys
import time
import argparse
from pathlib import Path
import pandas as pd

def escape_prompt(prompt):
    """Escape special characters in prompt"""
    return prompt.replace('"', '\\"').replace('`', '\\`').replace('$', '\\$')

def run_inference_batch(csv_file, output_dir="output", start_idx=0, end_idx=None, task="ti2v-5B"):
    """
    Batch inference function
    
    Args:
        csv_file: CSV file path
        output_dir: Output directory
        start_idx: Start index
        end_idx: End index
        task: Inference task type
    """
    
    # Fixed hyperparameters
    nproc_per_node = 8
    master_port = 29501
    ulysses_size = 8
    timeout = 3000
    
    # Auto-configure model path and size based on task
    if task == 'ti2v-5B':
        ckpt_dir = "path/to/Wan2.2-TI2V-5B"
        size = "1280*704"
    elif task == 'i2v-A14B':
        ckpt_dir = "path/to/Wan2.2-I2V-A14B"
        size = "1280*720"
    else:
        raise ValueError(f"Unsupported task type: {task}")

    # Check CSV file exists
    if not os.path.exists(csv_file):
        print(f"❌ Error: CSV file not found: {csv_file}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read CSV data
    try:
        data = pd.read_csv(csv_file)
    except Exception as e:
        print(f"❌ Failed to read CSV file: {e}")
        return False
    
    # Determine processing range
    total_count = len(data)
    start_idx = max(0, start_idx)
    end_idx = min(total_count, end_idx) if end_idx else total_count
    
    print(f"📊 Total data count: {total_count}")
    print(f"🎯 Processing range: {start_idx} - {end_idx-1} (total {end_idx-start_idx} items)")
    print(f"📁 Output directory: {output_dir}")
    print("="*60)
    
    success_count = 0
    failed_count = 0
    
    # Process each data entry
    for i in range(start_idx, end_idx):
        item = data.iloc[i]
        current_idx = i + 1
        
        print(f"\n🔄 [{current_idx}/{total_count}] Processing...")
        print(f"   Image: {os.path.basename(item['image_path'])}")
        print(f"   Prompt: {item['prompt'][:80]}...")
        
        # Generate output filename
        id_name, prompt_name = item['image_path'].split("/")[-2], item['image_path'].split("/")[-1].replace(".png", "").replace(".jpg", "")
        output_path = os.path.join(output_dir, id_name, f"{prompt_name}.mp4")
        
        # Ensure output directory exists
        output_parent_dir = os.path.dirname(output_path)
        if output_parent_dir:
            os.makedirs(output_parent_dir, exist_ok=True)

        # Check if file already exists
        if os.path.exists(output_path):
            print(f"   ⏭️  File already exists, skipping: {output_path}")
            success_count += 1
            continue
        
        # Build inference command
        cmd = [
            "torchrun", f"--nproc_per_node={nproc_per_node}", f"--master_port={master_port}",
            "generate.py",
            "--task", task,
            "--size", size,
            "--ckpt_dir", ckpt_dir,
            "--dit_fsdp", "--t5_fsdp",
            "--ulysses_size", str(ulysses_size),
            "--prompt", item['prompt'],
            "--image", item['image_path'],
            "--save_file", output_path
        ]
        
        # Record start time
        start_time = time.time()
    
        # Execute inference
        print(f"   🚀 Starting inference...")
        print(f"   📋 Executing command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            
            if result.returncode == 0:
                elapsed_time = time.time() - start_time
                print(f"   ✅ Success! Time elapsed: {elapsed_time:.1f}s")
                print(f"   📁 Output: {output_path}")
                success_count += 1
            else:
                print(f"   ❌ Inference failed! Return code: {result.returncode}")
                print(f"   📄 Standard output:")
                if result.stdout:
                    print(f"      {result.stdout}")
                print(f"   🚨 Error output:")
                if result.stderr:
                    print(f"      {result.stderr}")
                failed_count += 1
                
        except subprocess.TimeoutExpired:
            print(f"   ⏰ Inference timeout (>{timeout} seconds)")
            failed_count += 1
        except Exception as e:
            print(f"   💥 Execution exception: {e}")
            failed_count += 1
    
    # Print final statistics
    print("\n" + "="*60)
    print(f"📊 Batch inference completed!")
    print(f"✅ Success: {success_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"📊 Total: {success_count + failed_count}")
    
    return success_count > 0

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Wan2.2 Batch Inference Script')
    parser.add_argument('--csv_file', type=str,
                       default="./vip200k/data_refined_i2v.csv",
                       help='Input CSV file path')
    parser.add_argument('--output_dir', type=str,
                       default="vip200k/output_refined",
                       help='Output directory')
    parser.add_argument('--start_idx', type=int,
                       default=0,
                       help='Start index')
    parser.add_argument('--end_idx', type=int,
                       default=None,
                       help='End index (default: process to end)')
    parser.add_argument('--task', type=str,
                       choices=['ti2v-5B', 'i2v-A14B'],
                       default='ti2v-5B',
                       help='Inference task type')
    
    args = parser.parse_args()
    
    print("🚀 Wan2.2 Batch Inference Script")
    print(f"📄 CSV file: {args.csv_file}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🎯 Processing range: {args.start_idx} - {args.end_idx if args.end_idx else 'end'}")
    print(f"🔧 Task type: {args.task}")
    print("="*60)
    
    # Execute batch inference
    success = run_inference_batch(
        csv_file=args.csv_file,
        output_dir=args.output_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        task=args.task
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()

import os
import pandas as pd

def collect_version_data(version):
    """Collect data for a specific version"""
    print(f"🔄 Processing {version} version...")
    
    if version == "raw":
        txt_base_dir = "./vip200k/raw"
        image_base_dir = "./vip200k/i2v"
    elif version == "refined":
        txt_base_dir = "./vip200k/i2v"
        image_base_dir = "./vip200k/i2v"
    
    new_data = []
    for id_idx in range(1, 6):
        for prompt_idx in range(1, 6):
            txt_path = os.path.join(txt_base_dir, f"id{id_idx:03d}", f"prompt{prompt_idx}.txt")
            image_path = os.path.join(image_base_dir, f"id{id_idx:03d}", f"prompt{prompt_idx}.jpg")
            try:
                with open(txt_path, 'r', encoding='utf-8') as file:
                    prompt = file.read().strip()
            except Exception as e:
                print(f"❌ Error reading {txt_path}: {e}")
                continue
            new_data.append({"image_path": image_path, "prompt": prompt})
    
    new_data_path = f"./vip200k/data_{version}.csv"
    # save file
    df = pd.DataFrame(new_data)
    df.to_csv(new_data_path, index=False, encoding='utf-8-sig')
    print(f"✅ Saved {len(new_data)} entries to {new_data_path}")
    return new_data_path

# Execute both versions
versions = ["raw", "refined"]
for version in versions:
    collect_version_data(version)

print("🎉 All versions processed successfully!")
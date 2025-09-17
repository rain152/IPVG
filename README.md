# IPVG-STD: Spatial-Temporal Decoupled Identity Preserving Video Generation [MM 2025]

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2507.04705)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Accepted by ACM Multimedia 2025**

## Abstract

This repository contains the official implementation of our paper "Identity-Preserving Text-to-Video Generation Guided by Simple yet Effective Spatial-Temporal Decoupled Representations". Our method addresses the challenge of maintaining identity consistency in text-to-video generation through a novel spatial-temporal decoupled approach.

## 🔧 Environment Setup

Our implementation is built upon two main repositories:
- [Wan2.2](https://github.com/Wan-Video/Wan2.2) - For video generation
- [ComfyUI-HyperLoRA](https://github.com/bytedance/ComfyUI-HyperLoRA) - For identity-preserving image generation

### Prerequisites

1. **Install Wan2.2**
   ```bash
   git clone https://github.com/Wan-Video/Wan2.2.git
   cd Wan2.2
   # Follow the installation instructions in Wan2.2 repository
   ```

2. **Install ComfyUI-HyperLoRA**
   ```bash
   git clone https://github.com/bytedance/ComfyUI-HyperLoRA.git
   cd ComfyUI-HyperLoRA
   # Follow the installation instructions in ComfyUI-HyperLoRA repository
   ```

### Model Checkpoints

Download the required model checkpoints:
- **Wan2.2 Models**: Download from [Wan2.2 model hub](https://github.com/Wan-Video/Wan2.2#model-zoo)
- **HyperLoRA Models**: Download from [HyperLoRA releases](https://github.com/bytedance/ComfyUI-HyperLoRA#models)

### ComfyUI Integration

Our work also supports ComfyUI workflow for a visual interface experience:

1. **Setup ComfyUI Environment** (same as above prerequisites)

2. **Install Custom Node**
   ```bash
   # Copy our custom node to ComfyUI
   cp ./ComfyUI/qwen3_prompt_enhancer.py ./ComfyUI/custom_nodes/
   ```

3. **Load Workflow**
   - Open ComfyUI interface
   - Load the provided workflow file: `HyperLoRA-WAN2.2.json`
   - Experience the complete pipeline through visual interface

## 🚀 Quick Start

You can use our method in two ways: **Command Line Interface** or **ComfyUI Visual Interface**.

### Option 1: Command Line Interface

Our pipeline consists of three main stages:

### Stage 1: Prompt Optimization
Generate optimized T2I and I2V prompts using spatial-temporal decoupled representations:

```bash
python prompt_decoupled.py \
    --input_dir path/to/vip200k/raw/ \
    --output_dir path/to/vip200k \
    --model_name "Qwen/Qwen3-8B" \
    --cache_dir path/to/model
```

### Stage 2: Identity-Preserving First Frame Generation
Generate the first frame using HyperLoRA with identity preservation:

```bash
python hyperlora_t2i.py \
    --input_dir path/to/vip200k \
    --output_dir path/to/vip200k
```

### Stage 3: Video Generation
Generate final videos using Wan2.2:

```bash
python infer_Wan2.2.py \
    --csv_file /path/to/i2v_prompts.csv \
    --output_dir /path/to/videos \
    --task ti2v-5B
```

**Supported Tasks**:
- `ti2v-5B`: Text+Image to Video (5B model)
- `i2v-A14B`: Image to Video (14B model)

## 📊 Data Preparation

Use our data collection script to prepare your dataset:

```bash
python collect_data.py
```

## 🔄 Complete Pipeline Example

Here's a complete example workflow:

```bash
# 1. Generate optimized prompts
python prompt_decoupled.py \
    --input_dir path/to/vip200k/raw/ \
    --output_dir path/to/vip200k

# 2. Generate first frames
python hyperlora_t2i.py \
    --input_dir path/to/vip200k \
    --output_dir path/to/vip200k

# 3. Prepare data for video generation
python collect_data.py

# 4. Generate videos
python infer_Wan2.2.py \
    --csv_file ./vip200k/data_refined.csv \
    --output_dir ./final_videos
```

### Option 2: ComfyUI Visual Interface

For users who prefer a visual workflow:

1. **Load the Custom Node**: Ensure `qwen3_prompt_enhancer.py` is in your ComfyUI `custom_nodes` directory
2. **Open ComfyUI**: Start your ComfyUI interface
3. **Load Workflow**: Import the provided `HyperLoRA-WAN2.2.json` workflow file
4. **Configure Parameters**: Set your input images and prompts in the visual interface
5. **Execute**: Run the complete pipeline through the visual workflow

## 📁 Project Structure

```
MM-IPVG/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── prompt_decoupled.py               # Stage 1: Prompt optimization
├── hyperlora_t2i.py                  # Stage 2: First frame generation
├── infer_Wan2.2.py                   # Stage 3: Video generation
├── collect_data.py                   # Data preparation utility
├── prompt_optimizer.py               # Prompt optimization utilities
├── ComfyUI/
│   ├── qwen3_prompt_enhancer.py      # ComfyUI custom node
│   └── HyperLoRA-WAN2.2.json         # ComfyUI workflow file
├── utils/                            # Utility functions
└── vip200k/                          # Example data and configs
```

## 📄 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{wang2025identity,
  title={Identity-Preserving Text-to-Video Generation Guided by Simple yet Effective Spatial-Temporal Decoupled Representations},
  author={Wang, Yuji and Li, Moran and Hu, Xiaobin and Yi, Ran and Zhang, Jiangning and Feng, Han and Cao, Weijian and Wang, Yabiao and Wang, Chengjie and Ma, Lizhuang},
  journal={arXiv e-prints},
  pages={arXiv--2507},
  year={2025}
}
```

## 🙏 Acknowledgments

This work builds upon:
- [Wan2.2](https://github.com/Wan-Video/Wan2.2) for video generation capabilities
- [ComfyUI-HyperLoRA](https://github.com/bytedance/ComfyUI-HyperLoRA) for identity-preserving image generation
- The open-source community for various tools and libraries

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
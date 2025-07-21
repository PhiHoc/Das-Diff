# DasDiff: Diverse Augmented Single-Step Diffusion

**Official Implementation of the Undergraduate Thesis (2025):**  
*Enhancing Rare Animal Image Classification Performance Using Diffusion-Based Augmentation*  
**Authors:** Ly Phi Hoc - 52000760, Le Quang Duy - 520H0529  
**Advisor:** Mr. Tran Minh Tuan

---

## Abstract

Rare animal classification is a challenging fine-grained visual classification (FGVC) task due to limited labeled data and subtle inter-class differences. This work introduces **DasDiff**, a novel generative augmentation pipeline that uses a **single-step diffusion model** fine-tuned with **Textual Inversion** and **LoRA**, and enhances generation diversity with **LLM-generated environmental contexts**.

DasDiff accelerates synthetic data generation compared to traditional multi-step approaches (e.g., Diff-Aug, Diff-Mix), while achieving competitive or superior classification performance on four curated rare animal datasets: **Bear**, **Panther**, **Python**, and **Turtle**.

---

## Pipeline Overview

The DasDiff framework consists of three main components:

1. **LoRA + Textual Inversion Fine-tuning:** (see `train_lora_dasdiff.py`)  
   Adapt a single-step diffusion model to encode rare species with minimal real samples.

2. **Diverse Synthetic Image Generation:** (see `sample_dasdiff.py`)  
   Generate realistic images using class-specific tokens and diverse textual prompts describing environmental context.

3. **Classifier Training with Synthetic Integration:** (see `train_dasdiff.py`)  
   Train a ResNet-based classifier using both real and synthetic data with flexible control over the mixing ratio.

---

## Directory Structure

```
.
├── train_lora_dasdiff.py        # LoRA + TI fine-tuning of SwiftBrush UNet
├── sample_dasdiff.py            # Synthetic image generation using trained diffusion model
├── train_dasdiff.py             # Classifier training with mixed data
├── dataset/                     # Custom dataset definitions
├── environment.txt              # Optional: text prompts for environmental variation
└── README.md                    # You are here
```

---

## Requirements
```
torch==2.0.1+cu118
scipy==1.14.0
diffusers==0.29.2
transformers==4.42.4
typer==0.12.3
tqdm==4.66.4
natsort==8.4.0
open_clip_torch==2.26.1
wandb==0.17.4
peft==0.11.0
safetensors==0.4.3
hpsv2==1.2.0
datasets==2.16.1 
peft==0.5.0
```

---

## Usage

### 1. Finetune 

Original code checkpoint here

```bash
python train_lora_dasdiff.py \
  --pretrained_model_name_or_path stabilityai/sd-turbo \
  --swiftbrush_checkpoint_path path/to/base/unet \
  --dataset_name bear \
  --train_data_dir /path/to/bear/train_origin \
  --token_name "bear" \
  --output_dir /checkpoints/bear_lora
```

### 2. Generate Synthetic Images

```bash
python sample_dasdiff.py \
  --base_model_path stabilityai/sd-turbo \
  --swiftbrush_unet_path /path/to/base/unet \
  --textual_inversion_embeds_path ./checkpoints/bear_lora/learned_embeds.bin \
  --lora_weights_path /path/to/pytorch_lora_weights.safetensors
  --output_path /path/to/bear/synthetic \
  --dataset bear \
  --train_data_dir /path/to/bear/train_origin \
  --environment_file environment.txt \
  --syn_dataset_mulitiplier 5 \
  --gpu_ids 0
```

### 3. Train the Classifier

```bash
python train_dasdiff.py \
  --real_data_dir /path/to/bear/train_origin \
  --synthetic_data_dir /data/synthetic \
  --test_data_dir /path/to/bear/synthetic \
  --output_dir /results/bear_classifier \
  --dataset_name bear \
  --synthetic_probability 0.2 \
  --model resnet18
```

---

## Results

| Dataset  | Baseline Acc (%) | DasDiff Acc (%) | Speedup over Diff-Mix |
|----------|------------------|-----------------|------------------------|
| Turtle   | 79.5             | **87.4**        | ~4×                    |
| Bear     | 48.6             | **62.3**        | ~4×                    |
| Python   | 55.6             | **66.1**        | ~4×                    |
| Panther  | 43.5             | **59.8**        | ~4×                    |
---

## Datasets

Each dataset (Bear, Turtle, Panther, Python) is structured as:

```
data/
├── train_origin/
│   └── <dataset>/
│       └── class_name/
│           └── *.jpg
├── test/
│   └── <dataset>/
└── synthetic/
    └── data/
        └── class_name/
            └── *.png
```

You must register dataset classes in `dataset/__init__.py`.

---

## 🔮 Citation
```
@thesis{dasdiff2025,
  author       = {Ly Phi Hoc and Le Quang Duy},
  title        = {Enhancing Rare Animal Image Classification Performance Using Diffusion-Based Augmentation},
  school       = {Ton Duc Thang University},
  year         = {2025},
  advisor      = {Tran Minh Tuan}
}
```

---

## Acknowledgments

This codebase is built upon and inspired by two primary sources:

- [**Diff-Mix**](https://github.com/Zhicaiwww/Diff-Mix): Enhance Image Classification via Inter-class Image Mixup with Diffusion Model.
- [**SwiftBrush v2**](https://github.com/VinAIResearch/SwiftBrushV2):  Make Your One-step Diffusion Model Better Than Its Teacher.

We gratefully acknowledge the authors of these works for their contributions, which laid the foundation for DasDiff. This project adapts their techniques with a focus on rare species classification under low-shot settings.
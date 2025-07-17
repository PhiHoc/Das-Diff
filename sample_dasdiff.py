import argparse
import os
import random
import sys
import time
from collections import defaultdict
from multiprocessing import Process, Queue
from queue import Empty

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
import safetensors.torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dataset import DATASET_NAME_MAPPING

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["WANDB_DISABLED"] = "true"

def load_textual_inversion(tokenizer, text_encoder, embeds_path):
    learned_embeds_dict = torch.load(embeds_path, map_location="cpu")
    placeholder_tokens = learned_embeds_dict["placeholder_tokens"]
    learned_embeds = learned_embeds_dict["text_embeds"]
    tokenizer.add_tokens(placeholder_tokens)
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
    for token_id, embed in zip(placeholder_token_ids, learned_embeds):
        token_embeds[token_id] = embed

def apply_lora(unet, lora_path):
    lora_state_dict = safetensors.torch.load_file(lora_path)
    unet.load_state_dict(lora_state_dict, strict=False)

def tokenize_prompt(prompt, tokenizer):
    inputs = tokenizer(prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
    return inputs.input_ids

def generate_image(prompt, tokenizer, text_encoder, unet, vae, scheduler, device):
    input_ids = tokenize_prompt([prompt], tokenizer).to(device)
    encoder_hidden_state = text_encoder(input_ids)[0]

    noise = torch.randn(1, 4, 64, 64, device=device)
    timestep = torch.tensor([scheduler.config.num_train_timesteps - 1], device=device)

    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    alpha_t = (alphas_cumprod[timestep] ** 0.5).view(-1, 1, 1, 1)
    sigma_t = ((1 - alphas_cumprod[timestep]) ** 0.5).view(-1, 1, 1, 1)

    with torch.no_grad():
        model_pred = unet(noise, timestep, encoder_hidden_state).sample
        if model_pred.shape[1] == noise.shape[1] * 2:
            model_pred, _ = torch.split(model_pred, noise.shape[1], dim=1)

        pred_original_sample = (noise - sigma_t * model_pred) / alpha_t
        pred_original_sample = pred_original_sample / vae.config.scaling_factor
        image = (vae.decode(pred_original_sample).sample + 1) / 2
    return image

def sample_func(args, in_queue, gpu_id, process_id, environments):
    if torch.cuda.is_available() and gpu_id is not None:
        device = f"cuda:{gpu_id}"
    else:
        device = "cpu"
    torch.manual_seed(args.seed + process_id)
    np.random.seed(args.seed + process_id)
    random.seed(args.seed + process_id)

    tokenizer = CLIPTokenizer.from_pretrained(args.base_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.base_model_path, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(args.base_model_path, subfolder="vae").to(device)
    scheduler = DDPMScheduler.from_pretrained(args.base_model_path, subfolder="scheduler")
    unet = UNet2DConditionModel.from_pretrained(args.swiftbrush_unet_path).to(device)

    load_textual_inversion(tokenizer, text_encoder, args.textual_inversion_embeds_path)
    if args.lora_weights_path and os.path.exists(args.lora_weights_path):
        apply_lora(unet, args.lora_weights_path)

    train_dataset = DATASET_NAME_MAPPING[args.dataset](split="train", seed=args.seed, examples_per_class=args.examples_per_class, image_train_dir=args.train_data_dir)
    name2placeholder = {name: f"<class_{i}>" for i, name in enumerate(train_dataset.class_names)}

    while True:
        try:
            tasks = [in_queue.get(timeout=5) for _ in range(args.batch_size)]
        except Empty:
            break

        prompts, save_paths = [], []
        for index, target_label in tasks:
            target_name = train_dataset.label2class[target_label]
            target_placeholder = name2placeholder[target_name]
            environment = random.choice(environments) if environments else ""
            prompt = f"a photo of a {target_placeholder}, {environment}".strip()

            prompts.append(prompt)
            print(prompt)
            save_dir = os.path.join(args.output_path, "data", target_name.replace(" ", "_").replace("/", "_"))
            os.makedirs(save_dir, exist_ok=True)
            save_name = f"{target_name.replace(' ', '_').replace('/', '_')}-{index:06d}.png"
            save_paths.append(os.path.join(save_dir, save_name))

        for prompt, save_path in zip(prompts, save_paths):
            image = generate_image(prompt, tokenizer, text_encoder, unet, vae, scheduler, device)
            image = image.mul(255).clamp(0, 255).byte().cpu().squeeze(0).permute(1, 2, 0).numpy()
            Image.fromarray(image).save(save_path)

def main(args):

    os.makedirs(args.output_path, exist_ok=True)

    environments = []
    if args.environment_file and os.path.exists(args.environment_file):
        with open(args.environment_file, "r") as f:
            environments = [line.strip() for line in f if line.strip()]

    train_dataset = DATASET_NAME_MAPPING[args.dataset](split="train", seed=args.seed, examples_per_class=args.examples_per_class, image_train_dir=args.train_data_dir)
    num_classes = len(train_dataset.class_names)
    num_real_images = len(train_dataset) if args.examples_per_class == -1 else args.examples_per_class * num_classes
    num_synthetic_tasks = int(args.syn_dataset_mulitiplier * num_real_images)
    samples_per_class = num_synthetic_tasks // num_classes

    tasks = [(i, target_idx) for target_idx in range(num_classes) for i in range(samples_per_class)]
    remainder = num_synthetic_tasks % num_classes
    tasks += [(len(tasks) + i, random.choice(range(num_classes))) for i in range(remainder)]
    random.shuffle(tasks)

    in_queue = Queue()
    for task in tasks:
        in_queue.put(task)

    processes = []
    with tqdm(total=len(tasks), desc="Generating Images") as pbar:
        for process_id, gpu_id in enumerate(args.gpu_ids if torch.cuda.is_available() else [None] * 1):
            process = Process(target=sample_func, args=(args, in_queue, gpu_id, process_id, environments))
            process.start()
            processes.append(process)

        initial_tasks = len(tasks)
        while any(p.is_alive() for p in processes):
            pbar.n = initial_tasks - in_queue.qsize()
            pbar.refresh()
            time.sleep(1)
        pbar.n = initial_tasks
        pbar.refresh()
        for p in processes: p.join()

    print(f"Generation complete. Images saved under: {os.path.join(args.output_path, 'data')}")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser("Diverse Environment Sampling Script for SwiftBrush")
    parser.add_argument("--base_model_path", type=str, default="stabilityai/sd-turbo")
    parser.add_argument("--swiftbrush_unet_path", type=str, required=True)
    parser.add_argument("--lora_weights_path", type=str, default=None)
    parser.add_argument("--textual_inversion_embeds_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--environment_file", type=str, default=None, help="Path to txt file containing environment descriptions.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--examples_per_class", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0])
    parser.add_argument("--syn_dataset_mulitiplier", type=int, default=1)
    args = parser.parse_args()

    main(args)

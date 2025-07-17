import argparse
import os
import random
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

os.environ["CURL_CA_BUNDLE"] = ""

from collections import defaultdict
from multiprocessing import Process, Queue
from queue import Empty

from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from augmentation import AUGMENT_METHODS
from dataset import DATASET_NAME_MAPPING, IMBALANCE_DATASET_NAME_MAPPING
from utils.misc import parse_finetuned_ckpt
import shutil

def check_args_valid(args):
    if args.sample_strategy == "real-gen":
        args.lora_path = None
        args.embed_path = None
        args.aug_strength = 1
    elif args.sample_strategy == "diff-gen":
        lora_path, embed_path = parse_finetuned_ckpt(args.finetuned_ckpt)
        args.lora_path = lora_path
        args.embed_path = embed_path
        args.aug_strength = 1
    elif args.sample_strategy in ["real-aug", "real-mix"]:
        args.lora_path = None
        args.embed_path = None
    elif args.sample_strategy in ["diff-aug", "diff-mix"]:
        lora_path, embed_path = parse_finetuned_ckpt(args.finetuned_ckpt)
        args.lora_path = lora_path
        args.embed_path = embed_path

def sample_func(args, in_queue, out_queue, gpu_id, process_id):
    os.environ["CURL_CA_BUNDLE"] = ""

    random.seed(args.seed + process_id)
    np.random.seed(args.seed + process_id)
    torch.manual_seed(args.seed + process_id)

    if args.task == "imbalanced":
        train_dataset = IMBALANCE_DATASET_NAME_MAPPING[args.dataset](
            split="train",
            seed=args.seed,
            resolution=args.resolution,
            imbalance_factor=args.imbalance_factor,
            image_train_dir=args.train_data_dir,
            image_test_dir=args.test_data_dir,
        )
    else:
        train_dataset = DATASET_NAME_MAPPING[args.dataset](
            split="train",
            seed=args.seed,
            examples_per_class=args.examples_per_class,
            resolution=args.resolution,
            image_train_dir=args.train_data_dir,
            image_test_dir=args.test_data_dir,
        )

    model = AUGMENT_METHODS[args.sample_strategy](
        model_path=args.model_path,
        embed_path=args.embed_path,
        lora_path=args.lora_path,
        prompt=args.prompt,
        guidance_scale=args.guidance_scale,
        device=f"cuda:{gpu_id}",
    )
    batch_size = args.batch_size

    while True:
        index_list = []
        source_label_list = []
        target_label_list = []
        strength_list = []
        for _ in range(batch_size):
            try:
                index, source_label, target_label, strength = in_queue.get(timeout=1)
                index_list.append(index)
                source_label_list.append(source_label)
                target_label_list.append(target_label)
                strength_list.append(strength)
            except Empty:
                print("Queue is empty, exiting...")
                break

        if not index_list:
            break

        target_label = target_label_list[0]

        if not train_dataset.label_to_indices[target_label]:
            print(f"Warning: No indices found for target label {target_label}. Skipping batch.")
            continue

        target_indice = random.sample(train_dataset.label_to_indices[target_label], 1)[0]
        target_metadata = train_dataset.get_metadata_by_idx(target_indice)
        target_name = target_metadata["name"].replace(" ", "_").replace("/", "_")

        source_images = []
        save_paths = []
        if args.task == "vanilla":
            source_indices = [
                random.sample(train_dataset.label_to_indices[source_label], 1)[0]
                for source_label in source_label_list
            ]
        elif args.task == "imbalanced":
            source_indices = random.sample(range(len(train_dataset)), batch_size)

        for index, source_indice in zip(index_list, source_indices):
            source_images.append(train_dataset.get_image_by_idx(source_indice))
            source_metadata = train_dataset.get_metadata_by_idx(source_indice)
            source_name = source_metadata["name"].replace(" ", "_").replace("/", "_")
            save_name = os.path.join(
                source_name, f"{target_name}-{index:06d}-{strength}.png"
            )
            save_paths.append(os.path.join(args.output_path, "data", save_name))

        if os.path.exists(save_paths[0]):
            print(f"Skipping {save_paths[0]}")
        else:
            image, _ = model(
                image=source_images,
                label=target_label,
                strength=strength,
                metadata=target_metadata,
                resolution=args.resolution,
            )
            for image, save_path in zip(image, save_paths):
                image.save(save_path)
            print(f"Saved {save_path}")

def distribute_samples_cumulatively(master_path, base_output_root, base_folder_name, max_samples_per_class):
    """
    Read from the master folder, create subfolders for different sample sizes,
    and copy data cumulatively.
    """
    print("\n===== Start distributing data cumulatively =====")
    master_csv_path = os.path.join(master_path, "meta.csv")
    if not os.path.exists(master_csv_path):
        print(f"Error: meta.csv not found in master folder: {master_csv_path}")
        return

    master_df = pd.read_csv(master_csv_path)
    master_df.sort_values(by=["Second Directory", "Number"], inplace=True)
    grouped = master_df.groupby("Second Directory")

    steps = [50, 100]
    steps.extend(range(200, max_samples_per_class + 1, 100))

    for num_samples in steps:
        if num_samples > max_samples_per_class:
            continue

        print(f"--- Creating folder for {num_samples} samples per class ---")

        new_folder_name = f"{base_folder_name}_{num_samples}samples"
        output_path = os.path.join(base_output_root, new_folder_name)
        output_data_path = os.path.join(output_path, "data")
        os.makedirs(output_data_path, exist_ok=True)

        final_df_for_step = []

        for class_name, group_df in grouped:
            samples_for_class = group_df.head(num_samples)
            final_df_for_step.append(samples_for_class)

            class_folder_path = os.path.join(output_data_path, class_name.replace(" ", "_").replace("/", "_"))
            os.makedirs(class_folder_path, exist_ok=True)

            for _, row in samples_for_class.iterrows():
                source_image_path = os.path.join(master_path, "data", row["Path"])
                dest_image_path = os.path.join(output_data_path, row["Path"])
                os.makedirs(os.path.dirname(dest_image_path), exist_ok=True)

                if os.path.exists(source_image_path):
                    shutil.copy(source_image_path, dest_image_path)
                else:
                    print(f"Warning: Source file not found: {source_image_path}")

        if final_df_for_step:
            final_df = pd.concat(final_df_for_step).reset_index(drop=True)
            final_csv_path = os.path.join(output_path, "meta.csv")
            final_df.to_csv(final_csv_path, index=False)
            print(f"Created folder '{new_folder_name}' with {len(final_df)} images.")

    print("===== Data distribution completed! =====")

def main(args):
    torch.multiprocessing.set_start_method("spawn")

    base_output_root = os.path.join(args.output_root, args.dataset)
    os.makedirs(base_output_root, exist_ok=True)

    check_args_valid(args)
    if args.task == "vanilla":
        base_folder_name = f"shot{args.examples_per_class}_{args.sample_strategy}_{args.strength_strategy}_{args.aug_strength}"
    else:
        base_folder_name = f"imb{args.imbalance_factor}_{args.sample_strategy}_{args.strength_strategy}_{args.aug_strength}"

    output_name = f"{base_folder_name}_MASTER" if args.create_cumulative_steps else base_folder_name
    args.output_path = os.path.join(base_output_root, output_name)

    os.makedirs(args.output_path, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    gpu_ids = args.gpu_ids
    in_queue = Queue()
    out_queue = Queue()

    if args.task == "imbalanced":
        train_dataset = IMBALANCE_DATASET_NAME_MAPPING[args.dataset](
            split="train",
            seed=args.seed,
            resolution=args.resolution,
            imbalance_factor=args.imbalance_factor,
            image_train_dir=args.train_data_dir,
            image_test_dir=args.test_data_dir,
        )
    else:
        train_dataset = DATASET_NAME_MAPPING[args.dataset](
            split="train",
            seed=args.seed,
            examples_per_class=args.examples_per_class,
            resolution=args.resolution,
            image_train_dir=args.train_data_dir,
            image_test_dir=args.test_data_dir,
        )

    num_classes = len(train_dataset.class_names)
    for name in train_dataset.class_names:
        name = name.replace(" ", "_").replace("/", "_")
        os.makedirs(os.path.join(args.output_path, "data", name), exist_ok=True)

    num_tasks = args.syn_dataset_mulitiplier * len(train_dataset)

    samples_per_class = num_tasks // num_classes
    target_classes = [i for i in range(num_classes) for _ in range(samples_per_class)]
    remainder = num_tasks % num_classes
    if remainder > 0:
        target_classes.extend(random.sample(range(num_classes), remainder))

    random.shuffle(target_classes)

    if args.sample_strategy in ["real-gen", "real-aug", "diff-aug", "diff-gen", "ti-aug"]:
        source_classes = target_classes
    elif args.sample_strategy in ["real-mix", "diff-mix", "ti-mix"]:
        source_classes = random.choices(range(num_classes), k=num_tasks)
    else:
        raise ValueError(f"Unsupported strategy: {args.sample_strategy}")

    if args.strength_strategy == "fixed":
        strength_list = [args.aug_strength] * num_tasks
    elif args.strength_strategy == "uniform":
        strength_list = random.choices([0.3, 0.5, 0.7, 0.9], k=num_tasks)

    for option in zip(range(num_tasks), source_classes, target_classes, strength_list):
        in_queue.put(option)

    sample_config = vars(args)
    sample_config["num_classes"] = num_classes
    sample_config["total_tasks"] = num_tasks

    with open(os.path.join(args.output_path, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(sample_config, f)

    processes = []
    total_tasks = in_queue.qsize()
    print("Total number of tasks:", total_tasks)

    with tqdm(total=total_tasks, desc="Processing") as pbar:
        for process_id, gpu_id in enumerate(gpu_ids):
            process = Process(target=sample_func, args=(args, in_queue, out_queue, gpu_id, process_id))
            process.start()
            processes.append(process)

        while any(p.is_alive() for p in processes):
            pbar.n = total_tasks - in_queue.qsize()
            pbar.refresh()
            time.sleep(1)

        for process in processes:
            process.join()

    # Generate meta.csv
    rootdir = os.path.join(args.output_path, "data")
    pattern_level_1 = r"(.+)"
    pattern_level_2 = r"(.+)-(\d+)-(.+).png"
    data_dict = defaultdict(list)

    for dir in os.listdir(rootdir):
        if not os.path.isdir(os.path.join(rootdir, dir)):
            continue
        first_dir = re.match(pattern_level_1, dir).group(1).replace("_", " ")
        for file in os.listdir(os.path.join(rootdir, dir)):
            match_2 = re.match(pattern_level_2, file)
            second_dir = match_2.group(1).replace("_", " ")
            num = int(match_2.group(2))
            floating_num = float(match_2.group(3))
            data_dict["First Directory"].append(first_dir)
            data_dict["Second Directory"].append(second_dir)
            data_dict["Number"].append(num)
            data_dict["Strength"].append(floating_num)
            data_dict["Path"].append(os.path.join(dir, file))

    df = pd.DataFrame(data_dict)

    valid_rows = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        image_path = os.path.join(args.output_path, "data", row["Path"])
        try:
            img = Image.open(image_path)
            img.close()
            valid_rows.append(row)
        except Exception as e:
            os.remove(image_path)
            print(f"Deleted {image_path} due to error: {str(e)}")

    valid_df = pd.DataFrame(valid_rows)
    csv_path = os.path.join(args.output_path, "meta.csv")
    valid_df.to_csv(csv_path, index=False)
    print("DataFrame:")
    print(df)

    if args.create_cumulative_steps:
        if args.examples_per_class <= 0:
            print("Error: You must provide --examples_per_class to calculate maximum samples.")
            return

        max_samples = args.syn_dataset_mulitiplier * args.examples_per_class
        distribute_samples_cumulatively(args.output_path, base_output_root, base_folder_name, max_samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference script")
    parser.add_argument("--create_cumulative_steps", action="store_true", help="Enable cumulative subfolder creation.")
    parser.add_argument("--finetuned_ckpt", type=str, required=True, help="Path to finetuned checkpoint.")
    parser.add_argument("--output_root", type=str, default="outputs/aug_samples", help="Root output directory.")
    parser.add_argument("--train_data_dir", type=str, default=None, help="Training dataset directory.")
    parser.add_argument("--test_data_dir", type=str, default=None, help="Testing dataset directory.")
    parser.add_argument("--model_path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--dataset", type=str, default="pascal", help="Dataset name.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--examples_per_class", type=int, default=-1, help="Number of examples per class.")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--prompt", type=str, default="a photo of a {name}", help="Prompt template.")
    parser.add_argument("--sample_strategy", type=str, default="ti-mix", choices=["real-gen", "real-aug", "real-mix", "ti-aug", "ti-mix", "diff-aug", "diff-mix", "diff-gen"])
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Guidance scale.")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0], help="List of GPU IDs.")
    parser.add_argument("--task", type=str, default="vanilla", choices=["vanilla", "imbalanced"], help="Task type.")
    parser.add_argument("--imbalance_factor", type=float, default=0.01, choices=[0.01, 0.02, 0.1], help="Imbalance factor.")
    parser.add_argument("--syn_dataset_mulitiplier", type=int, default=5, help="Synthetic dataset size multiplier.")
    parser.add_argument("--strength_strategy", type=str, default="fixed", choices=["fixed", "uniform"])
    parser.add_argument("--aug_strength", type=float, default=0.5, help="Augmentation strength.")
    args = parser.parse_args()

    main(args)

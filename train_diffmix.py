import argparse
import json
import math
import os
import random
import shutil
import sys
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torchvision.models import ViT_B_16_Weights, resnet18, resnet50, vit_b_16
from tqdm import tqdm
import pytorch_warmup

# Add the parent directory to the system path to allow for custom module imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from dataset import DATASET_NAME_MAPPING
from dataset.base import SyntheticDataset
from downstream_tasks.losses import LabelSmoothingLoss
from downstream_tasks.mixup import CutMix, mixup_data
from utils.network import freeze_model
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, ConcatDataset

#######################
##### 1. Settings #####
#######################

# Set the device to CUDA if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def save_metrics_to_json(metrics, filename):
    """Saves a dictionary of metrics to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)


def save_checkpoint(model, optimizer, epoch, metrics, path):
    """Saves the model checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'metrics': metrics
    }
    torch.save(checkpoint, path)


def format_run_name_note(args):
    """Formats a note for the experiment run name based on arguments."""
    args.use_warmup = True
    note = f"{args.note}"
    if args.synthetic_data_dir is not None:
        note = note + f"_{os.path.basename(args.synthetic_data_dir[0])}"
    if args.use_cutmix:
        note = note + "_cutmix"
    if args.use_mixup:
        note = note + "_mixup"
    return note


# --- Argument Parser Setup ---
parser = argparse.ArgumentParser(description="Image Classification Training Script")
parser.add_argument("-d", "--dataset", default="cub", help="Dataset name (e.g., 'cub').")
parser.add_argument("--synthetic_data_dir", type=str, nargs="+", help="Directory/directories for synthetic data.")
parser.add_argument("--synthetic_data_prob", default=0.1, type=float, help="Probability of using a synthetic image.")
parser.add_argument("-m", "--model", default="resnet50",
                    choices=["resnet18", "resnet18pretrain", "resnet50", "vit_b_16"], help="Model architecture.")
parser.add_argument("-b", "--batch_size", default=32, type=int, help="Batch size for training and evaluation.")
parser.add_argument("--lr", default=0.01, type=float, help="Initial learning rate.")
parser.add_argument("--weight_decay", default=5e-4, type=float, help="Weight decay for the optimizer.")
parser.add_argument("--use_cutmix", default=False, action="store_true", help="Enable CutMix augmentation.")
parser.add_argument("--use_mixup", default=False, action="store_true", help="Enable Mixup augmentation.")
parser.add_argument("--criterion", default="ls", type=str, choices=["ce", "ls"],
                    help="Loss function: 'ce' for CrossEntropy, 'ls' for LabelSmoothing.")
parser.add_argument("-g", "--gpu", default=0, type=int, help="GPU ID to use.")
parser.add_argument("-w", "--num_workers", default=4, type=int,
                    help="Number of workers for DataLoader (<=4 recommended for Colab).")
parser.add_argument("-s", "--seed", default=2020, type=int, help="Random seed for reproducibility.")
parser.add_argument("-n", "--note", default="", help="Custom note to append to the experiment folder name.")
parser.add_argument("-p", "--group_note", default="debug", help="Group name for organizing experiments.")
parser.add_argument("-a", "--amp", default=0, type=int, help="AMP mode: 0=off, 1=NVIDIA Apex, 2=PyTorch native AMP.")
parser.add_argument("-rs", "--resize", default=512, type=int, help="Image resize dimension.")
parser.add_argument("--res_mode", default="224", type=str,
                    help="Resolution mode to auto-configure settings (e.g., '224', '384').")
parser.add_argument("-cs", "--crop_size", type=int, default=448, help="Image crop size after resizing.")
parser.add_argument("--examples_per_class", type=int, default=-1,
                    help="Number of examples per class for few-shot learning (-1 for all).")
parser.add_argument("--gamma", type=float, default=1.0, help="Gamma factor for soft labels from synthetic data.")
parser.add_argument("-mp", "--mixup_probability", type=float, default=0.5, help="Probability of applying Mixup/CutMix.")
parser.add_argument("-ne", "--num_epochs", type=int, default=100, help="Total number of training epochs.")
parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw"], help="Optimizer to use.")
parser.add_argument("-fs", "--finetune_strategy", type=str, default=None,
                    help="Strategy for fine-tuning specific layers.")
parser.add_argument("--train_data_dir", type=str, default=None, help="Path to the original training data folder.")
parser.add_argument("--test_data_dir", type=str, default=None, help="Path to the original testing data folder.")
parser.add_argument("--output_root", type=str, default="outputs/result",
                    help="Root directory for all experiment results.")
parser.add_argument("--soft_label_scaler", type=float, default=1.0, help="Scaling factor for soft labels.")
parser.add_argument(
    "--data_mode",
    type=str,
    default="probabilistic",
    choices=["probabilistic", "concat"],
    help="Data loading strategy: 'probabilistic' uses synthetic_data_prob, 'concat' combines datasets."
)

args = parser.parse_args()
run_name = f"{args.dataset}_{format_run_name_note(args)}"
experiment_dir = os.path.join(args.output_root, args.group_note, run_name)

if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)
print(f"Experiment results will be saved to: {experiment_dir}")

# --- Auto-configure settings based on resolution mode ---
if args.optimizer == "sgd":
    base_lr = 0.02
elif args.optimizer == "adamw":
    base_lr = 1e-3
else:
    raise ValueError("Unsupported optimizer specified.")

if args.res_mode == "28":
    args.resize = 32
    args.crop_size = 28
    args.batch_size = 2048
elif args.res_mode == "224":
    args.resize = 256
    args.crop_size = 224
    if args.model in ["resnet50", "resnet18", "resnet18pretrain"]:
        args.batch_size = 256
    elif args.model == "vit_b_16":
        args.batch_size = 128
    else:
        raise ValueError("Unsupported model for this resolution mode.")
elif args.res_mode == "384":
    args.resize = 440
    args.crop_size = 384
    if args.model in ["resnet50", "resnet18", "resnet18pretrain"]:
        args.batch_size = 128
    elif args.model == "vit_b_16":
        args.batch_size = 32
    else:
        raise ValueError("Unsupported model for this resolution mode.")
elif args.res_mode == "448":
    args.resize = 512
    args.crop_size = 448
    if args.model in ["resnet50", "resnet18", "resnet18pretrain"]:
        args.batch_size = 64
    elif args.model == "vit_b_16":
        args.batch_size = 32
    else:
        raise ValueError("Unsupported model for this resolution mode.")
else:
    raise ValueError("Unsupported resolution mode specified.")

# Ensure num_workers is safe for Colab environments
args.num_workers = min(args.num_workers, 4)

# --- Set up CUDA device and random seeds ---
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
print(f"Final device selection: {device}")

random.seed(args.seed)
os.environ["PYTHONHASHSEED"] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


#############################
##### 2. Data Loading #######
#############################

def to_tensor(x):
    """Converts input to a PyTorch tensor."""
    if isinstance(x, int):
        return torch.tensor(x)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        return x
    else:
        raise NotImplementedError(f"Cannot convert type {type(x)} to tensor.")


if args.data_mode == 'probabilistic':
    print("===== Data Loading Mode: Probabilistic (Default) =====")


    # This mode works like the original code, swapping real images for synthetic ones based on a probability.
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.stack([to_tensor(example["label"]) for example in examples])
        # Determine dtype based on whether labels are one-hot (soft) or integer (hard)
        dtype = torch.float32 if len(labels.size()) == 2 else torch.long
        labels = labels.to(dtype=dtype)
        return {"pixel_values": pixel_values, "labels": labels}


    train_set = DATASET_NAME_MAPPING[args.dataset](
        split="train",
        image_size=args.resize,
        crop_size=args.crop_size,
        synthetic_dir=args.synthetic_data_dir,
        synthetic_probability=args.synthetic_data_prob,
        return_onehot=True,
        gamma=args.gamma,
        examples_per_class=args.examples_per_class,
        image_train_dir=args.train_data_dir,
        image_test_dir=args.test_data_dir,
    )
    num_classes = train_set.num_classes

elif args.data_mode == 'concat':
    print("===== Data Loading Mode: Concatenate =====")
    # 1. Load the original dataset with hard labels.
    original_train_set = DATASET_NAME_MAPPING[args.dataset](
        split="train",
        image_size=args.resize,
        crop_size=args.crop_size,
        return_onehot=True,
        synthetic_dir=None,  # Disable synthetic mode to get only original images
        examples_per_class=args.examples_per_class,
        image_train_dir=args.train_data_dir,
        image_test_dir=args.test_data_dir,
    )
    num_classes = original_train_set.num_classes


    # 2. Define a Dataset class for synthetic data by inheriting from SyntheticDataset.
    class SyntheticSoftLabelDataset(SyntheticDataset):
        """Dataset for synthetic images that generates soft labels."""

        def __init__(self, *args, **kwargs):
            self.gamma = kwargs.pop('gamma', 1.0)
            self.soft_label_scaler = kwargs.pop('soft_label_scaler', 1.0)
            super().__init__(*args, **kwargs)  # Call parent constructor

        def __getitem__(self, idx: int) -> dict:
            # Get raw data from the parent class (image path and integer labels)
            path, src_label, tar_label = self.get_syn_item_raw(idx)
            image = Image.open(path).convert("RGB")

            # Read the strength value from the metadata dataframe
            df_data = self.meta_df.iloc[idx]
            strength = df_data["Strength"]

            # Calculate the soft label
            soft_label = torch.zeros(self.num_classes)
            soft_label[src_label] += self.soft_label_scaler * (1 - math.pow(strength, self.gamma))
            soft_label[tar_label] += self.soft_label_scaler * math.pow(strength, self.gamma)

            return {"pixel_values": self.transform(image), "label": soft_label}


    datasets_to_concat = [original_train_set]
    if args.synthetic_data_dir:
        synthetic_soft_label_set = SyntheticSoftLabelDataset(
            synthetic_dir=args.synthetic_data_dir,
            class2label=original_train_set.class2label,
            gamma=args.gamma,
            soft_label_scaler=args.soft_label_scaler,
            image_size=args.resize,
            crop_size=args.crop_size
        )
        datasets_to_concat.append(synthetic_soft_label_set)

    # 3. Combine the datasets.
    train_set = ConcatDataset(datasets_to_concat)
    print(f"Total images in concatenated training set: {len(train_set)}")


    # 4. Define the collate function (both datasets now return tensor labels).
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.stack([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

# Always define the test set for evaluation.
test_set = DATASET_NAME_MAPPING[args.dataset](
    split="val",
    image_size=args.resize,
    crop_size=args.crop_size,
    return_onehot=True,
    image_train_dir=args.train_data_dir,
    image_test_dir=args.test_data_dir
)

batch_size = min(args.batch_size, len(train_set))

# Apply CutMix augmentation if enabled, after the final train_set is created.
if args.use_cutmix:
    if args.data_mode == 'concat':
        print("Note: CutMix is being applied to the combined (concatenated) dataset.")
    train_set = CutMix(
        train_set, num_class=num_classes, prob=args.mixup_probability
    )

# Create the DataLoader for the training set.
train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=args.num_workers,
)


# Collate function for the evaluation loader.
def eval_collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.stack([to_tensor(example["label"]) for example in examples])
    dtype = torch.float32 if len(labels.size()) == 2 else torch.long
    labels = labels.to(dtype=dtype)
    return {"pixel_values": pixel_values, "labels": labels}


eval_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=eval_collate_fn,
    num_workers=args.num_workers,
)

#################################################
##### 3. Model, Optimizer, and Loss Setup #######
#################################################

# --- Model Selection ---
if args.model == "resnet18":
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
elif args.model == "resnet18pretrain":
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
elif args.model == "resnet50":
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
elif args.model == "vit_b_16":
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

model = model.to(device)

# Ensure all parameters are trainable by default
for param in model.parameters():
    param.requires_grad = True

# Apply fine-tuning strategy if specified
if args.finetune_strategy is not None and args.model == "resnet50":
    freeze_model(model, args.finetune_strategy)

# --- Loss Function ---
if args.criterion == "ce":
    criterion = nn.CrossEntropyLoss()
elif args.criterion == "ls":
    criterion = LabelSmoothingLoss(classes=num_classes, smoothing=0.1)
else:
    raise NotImplementedError("The specified loss function is not implemented.")

# --- Optimizer and Scheduler ---
if args.optimizer == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
elif args.optimizer == "adamw":
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
else:
    raise ValueError("The specified optimizer is not supported.")

total_steps = args.num_epochs * len(train_loader.dataset) // batch_size
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
warmup_scheduler = pytorch_warmup.LinearWarmup(optimizer, warmup_period=max(int(0.1 * total_steps), 1))

# --- Prepare Experiment Directory ---
if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)

# Save a copy of this script and the configuration
shutil.copyfile(__file__, os.path.join(experiment_dir, "train_script.py"))
with open(os.path.join(experiment_dir, "config.yaml"), "w", encoding="utf-8") as f:
    yaml.dump(vars(args), f)
with open(os.path.join(experiment_dir, "training_log.csv"), "w", encoding="utf-8") as f:
    f.write("Epoch,LearningRate,TrainLoss,TrainAcc,TestAcc\n")

# --- AMP (Automatic Mixed Precision) Setup ---
use_amp = args.amp
scaler = None
if use_amp == 1:
    print("\n===== Using NVIDIA Apex AMP =====")
    from apex import amp

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
elif use_amp == 2:
    print("\n===== Using PyTorch Native AMP =====")
    scaler = GradScaler()

# --- Resume from Checkpoint ---
checkpoint_path = os.path.join(experiment_dir, "checkpoint.pth")
metrics_path = os.path.join(experiment_dir, "metrics.json")

start_epoch = 0
train_losses, val_accuracies = [], []
train_accuracies = []
best_accuracy = 0.0

if os.path.exists(checkpoint_path):
    print(f"Resuming training from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1

    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        train_losses = metrics.get('train_losses', [])
        train_accuracies = metrics.get('train_accuracies', [])
        val_accuracies = metrics.get('val_accuracies', [])
        best_accuracy = metrics.get('best_accuracy', 0.0)

    # Move optimizer state tensors to the correct device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

################################
##### 4. Training Loop #########
################################
max_eval_acc = best_accuracy

for epoch in range(start_epoch, args.num_epochs):
    print(f"\n===== Epoch: {epoch}/{args.num_epochs - 1} =====")
    model.train()
    current_lr = optimizer.param_groups[0]["lr"]
    total_train_loss = 0.0
    train_correct = 0
    train_total = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=100)
    for batch_idx, batch in enumerate(progress_bar):
        optimizer.zero_grad()
        inputs = batch["pixel_values"].to(device)
        targets = batch["labels"].to(device)

        # Apply Mixup if enabled
        if args.use_mixup and np.random.rand() < args.mixup_probability:
            inputs, targets = mixup_data(inputs, targets, alpha=1.0, num_classes=num_classes)

        # Skip incomplete batches
        if inputs.shape[0] < batch_size:
            continue

        # Forward and backward pass with optional AMP
        if use_amp == 1:  # NVIDIA Apex AMP
            with amp.scale_loss(criterion(model(inputs), targets), optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
        elif use_amp == 2:  # PyTorch Native AMP
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:  # No AMP
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Update training metrics
        _, predicted = torch.max(outputs.data, 1)
        train_total += targets.size(0)

        # Handle both soft and hard labels for accuracy calculation
        if len(targets.shape) == 2:
            targets = torch.argmax(targets, dim=1)

        train_correct += predicted.eq(targets.data).cpu().sum().item()
        total_train_loss += loss.item()

        # Update progress bar
        progress_bar.set_postfix(loss=loss.item(), lr=current_lr)

        # Step the schedulers
        with warmup_scheduler.dampening():
            scheduler.step()

    train_acc = 100.0 * train_correct / train_total
    avg_train_loss = total_train_loss / len(train_loader)
    print(
        f"Train | LR: {current_lr:.6f} | Loss: {avg_train_loss:.4f} | Acc: {train_acc:.3f}% ({train_correct}/{train_total})")

    # --- Evaluation ---
    # Evaluate every 4 epochs or on the final epoch
    if (epoch + 1) % 4 == 0 or (epoch + 1) == args.num_epochs:
        model.eval()
        eval_correct = 0
        eval_total = 0
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating", ncols=100):
                inputs = batch["pixel_values"].to(device)
                targets = batch["labels"].to(device)
                outputs = model(inputs)

                _, predicted = torch.max(outputs.data, 1)
                eval_total += targets.size(0)

                if len(targets.shape) == 2:
                    targets = torch.argmax(targets, dim=1)

                eval_correct += predicted.eq(targets.data).cpu().sum().item()

        eval_acc = 100.0 * eval_correct / eval_total
        print(f"Test  | Acc: {eval_acc:.3f}% ({eval_correct}/{eval_total})")

        # Update and save metrics
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(eval_acc)

        if eval_acc > best_accuracy:
            best_accuracy = eval_acc
            print(f"🎉 New best accuracy: {best_accuracy:.2f}%")
            # Save the best model
            torch.save(model.state_dict(), os.path.join(experiment_dir, "best_model.pth"))

        metrics = {
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies,
            "best_accuracy": best_accuracy
        }

        save_metrics_to_json(metrics, metrics_path)
        save_checkpoint(model, optimizer, epoch, metrics, checkpoint_path)

        # Log to CSV
        with open(os.path.join(experiment_dir, "training_log.csv"), "a", encoding="utf-8") as f:
            f.write(f"{epoch},{current_lr:.6f},{avg_train_loss:.4f},{train_acc:.3f},{eval_acc:.3f}\n")

#######################
##### 5. Final Testing #####
#######################
print("\n\n===== FINAL TESTING =====")
with open(os.path.join(experiment_dir, "training_log.csv"), "a") as f:
    f.write("===== FINAL TESTING =====\n")

# Load the best model for the final test
best_model_path = os.path.join(experiment_dir, "best_model.pth")
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path, map_location=device))
else:
    print("Warning: 'best_model.pth' not found. Testing with the last saved model.")
model.eval()

# Test on both training and validation sets
for dataset_name, data_loader in zip(["Train Set", "Test Set"], [train_loader, eval_loader]):
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Testing on {dataset_name}", ncols=100):
            inputs = batch["pixel_values"].to(device)
            targets = batch["labels"].to(device)

            if len(targets.shape) == 2:
                targets = torch.argmax(targets, dim=1)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets.data).cpu().sum().item()

    test_acc = 100.0 * test_correct / test_total
    print(f"Final Accuracy on {dataset_name}: {test_acc:.2f}%")

    with open(os.path.join(experiment_dir, "training_log.csv"), "a", encoding="utf-8") as f:
        f.write(f"Final Accuracy on {dataset_name}: {test_acc:.2f}%\n")

print("\nTraining and evaluation complete.")

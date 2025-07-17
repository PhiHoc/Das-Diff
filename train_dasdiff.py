import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# Add resnet18 and ResNet18_Weights to imports
from torchvision.models import (vit_b_16, ViT_B_16_Weights,
                                resnet50, ResNet50_Weights,
                                resnet18, ResNet18_Weights)
from tqdm import tqdm

# Reuse the existing dataset system from your codebase
from dataset import DATASET_NAME_MAPPING


def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args):
    """Main function to run the training and evaluation process."""
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    # --- Dataset Loading ---
    DatasetClass = DATASET_NAME_MAPPING[args.dataset_name]
    train_set = DatasetClass(
        split='train',
        image_train_dir=args.real_data_dir,
        examples_per_class=-1,
        return_onehot=False,  # Use integer labels for CrossEntropyLoss
        synthetic_dir=args.synthetic_data_dir,
        synthetic_probability=args.synthetic_probability,
        image_size=args.resize,
        crop_size=args.crop_size,
    )
    test_set = DatasetClass(
        split='val',
        image_test_dir=args.test_data_dir,
        examples_per_class=-1,
        return_onehot=False,
        synthetic_dir=None,  # No synthetic data for testing
        image_size=args.resize,
        crop_size=args.crop_size,
    )

    num_classes = train_set.num_classes
    print(f"Dataset: {args.dataset_name} with {num_classes} classes.")
    print(f"Using model: {args.model}")
    print(f"Training with `synthetic_probability` = {args.synthetic_probability}")
    print(f"Training with `label_smoothing` = {args.label_smoothing}")
    print(f"Testing with {len(test_set)} real samples.")

    # --- Model Selection ---
    if args.model == "resnet18":
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.model == "resnet18pretrain":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.model == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.model == "vit_b_16":
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Model {args.model} is not supported.")

    model.to(device)
    # Compile the model for a potential speed-up if using PyTorch 2.0+
    if torch.__version__ >= "2.0.0":
        model = torch.compile(model)

    # --- Loss, Optimizer, and Scheduler ---
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # --- DataLoader ---
    def collate_fn(batch):
        """Custom collate function to filter out None values from the dataset."""
        batch = [b for b in batch if b is not None and 'pixel_values' in b and 'label' in b]
        if not batch:
            return None, None
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        return pixel_values, labels

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                              collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True,
                             collate_fn=collate_fn)

    # --- Training Loop ---
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_accuracy = 0.0
    patience_counter = 0
    for epoch in range(args.num_epochs):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")

        for inputs, labels in pbar:
            if inputs is None: continue  # Skip empty batches
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            # Use Automatic Mixed Precision
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += inputs.size(0)
            if total_samples > 0:
                pbar.set_postfix(loss=f"{total_loss / total_samples:.4f}", acc=f"{total_correct / total_samples:.4f}")

        # --- Evaluation ---
        model.eval()
        eval_correct, eval_samples = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                if inputs is None: continue
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                eval_correct += (predicted == labels).sum().item()
                eval_samples += inputs.size(0)

        if eval_samples > 0:
            eval_acc = eval_correct / eval_samples
            print(f"\nEpoch {epoch + 1} - Test Accuracy: {eval_acc:.4f}")

            # Save the best model and implement early stopping
            if eval_acc > best_accuracy:
                patience_counter = 0
                best_accuracy = eval_acc
                torch.save(model.state_dict(), output_dir / "best_model.pth")
                print(f"*** New best model saved with Test Acc: {best_accuracy:.4f} ***\n")
            else:
                patience_counter += 1

        if patience_counter >= 20:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

        scheduler.step()

    print(f"Training finished. Best Test Accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Augmentation Experiment with Label Smoothing.")

    # --- Data and Path Arguments ---
    parser.add_argument("--real_data_dir", type=str, required=True, help="Directory containing the real training data.")
    parser.add_argument("--test_data_dir", type=str, required=True, help="Directory containing the test data.")
    parser.add_argument("--synthetic_data_dir", type=str, default=None, help="Directory for synthetic data. Leave empty to not use.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model.")
    parser.add_argument("--dataset_name", type=str, required=True, choices=DATASET_NAME_MAPPING.keys(), help="Name of the dataset as defined in __init__.py.")

    # --- Training Strategy Arguments ---
    parser.add_argument("--synthetic_probability", type=float, default=0.0, help="Probability of sampling a synthetic image. Set to 0.0 to train only on real data.")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Epsilon value for Label Smoothing. Set to 0.0 to disable.")
    parser.add_argument("--model", type=str, default="resnet18pretrain", choices=["resnet18", "resnet18pretrain", "resnet50", "vit_b_16"], help="Choice of model architecture to train.")

    # --- Hyperparameter Arguments ---
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the AdamW optimizer.")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    # Rename 'nepoch' to 'num_epochs' for clarity, as it's used in the main function
    args.num_epochs = args.nepoch
    main(args)

"""
High-Accuracy ResNet-34 Training Pipeline for Galaxy Morphology Classification.
Optimized for NVIDIA GeForce RTX 3050 (4GB VRAM):
- PyTorch Automatic Mixed Precision (AMP)
- Differential Learning Rates (backbone vs head)
- Gradient Accumulation (effective batch size 32)
- Linear Warmup + Cosine Annealing learning rate schedule
- Cross-Entropy with Label Smoothing
- Balanced Accuracy tracking and checkpointing
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from dataset import get_dataloaders
from model import GalaxyResNet34, build_model, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train GalaxyResNet34 on Galaxy Zoo 2")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="GPU batch size (16 for 4GB VRAM)")
    parser.add_argument("--accum-steps", type=int, default=config.GRADIENT_ACCUMULATION_STEPS, help="Gradient accumulation steps")
    parser.add_argument("--lr-backbone", type=float, default=config.LR_BACKBONE, help="Learning rate for ResNet-34 backbone")
    parser.add_argument("--lr-head", type=float, default=config.LR_HEAD, help="Learning rate for GeM pool and classifier head")
    parser.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY, help="Weight decay")
    parser.add_argument("--label-smoothing", type=float, default=config.LABEL_SMOOTHING, help="Label smoothing factor")
    parser.add_argument("--patience", type=int, default=config.EARLY_STOPPING_PATIENCE, help="Early stopping patience")
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS, help="DataLoader worker threads")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device")
    return parser.parse_args()


def get_parameter_groups(
    model: GalaxyResNet34,
    lr_backbone: float,
    lr_head: float,
    weight_decay: float
) -> list:
    """
    Separate parameters into differential learning rate groups:
    - Backbone features: Fine-tuned with low learning rate (3e-5)
    - Head & GeM pool: Trained with higher learning rate (3e-4)
    """
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name or "gem_pool" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    return [
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": head_params, "lr": lr_head, "weight_decay": weight_decay}
    ]


def train_one_epoch(
    model: GalaxyResNet34,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    accum_steps: int = 2
) -> Tuple[float, float]:
    """Train for a single epoch with mixed precision and gradient accumulation."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d} [Train]", leave=False)

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Normalize loss for gradient accumulation
            loss = loss / accum_steps

        # Backward with scaling
        scaler.scale(loss).backward()

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * accum_steps
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            "loss": f"{running_loss / (batch_idx + 1):.4f}",
            "acc": f"{100.0 * correct / total:.2f}%"
        })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_epoch(
    model: GalaxyResNet34,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> Tuple[float, float, float, float]:
    """Evaluate validation split and compute balanced accuracy and macro-F1."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []

    pbar = tqdm(val_loader, desc=f"Epoch {epoch:02d} [Val]  ", leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

    val_loss = running_loss / len(val_loader)
    val_acc = 100.0 * correct / total
    balanced_acc = 100.0 * balanced_accuracy_score(all_targets, all_preds)
    macro_f1 = 100.0 * f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return val_loss, val_acc, balanced_acc, macro_f1


def main():
    args = parse_args()

    # Device selection
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
        print(f"\n🚀 Hardware: NVIDIA GPU ({torch.cuda.get_device_name(0)})")
        print(f"   VRAM Available: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"   Automatic Mixed Precision (AMP): ✅ Enabled")
    else:
        device = torch.device("cpu")
        print("\n⚠️ Hardware: CPU (Training will proceed slower without CUDA)")

    # Data loaders
    print("\n📥 Loading real Galaxy Zoo 2 dataset...")
    train_loader, val_loader, test_loader, class_names, class_weights = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Initialize GalaxyResNet34
    model = build_model(
        num_classes=config.NUM_CLASSES,
        device=device,
        pretrained=config.USE_PRETRAINED
    )

    # Loss with label smoothing and class weights
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=args.label_smoothing)

    # Optimizer with differential learning rates
    param_groups = get_parameter_groups(
        model,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay
    )
    optimizer = torch.optim.AdamW(param_groups)

    # Cosine Annealing scheduler with warmup
    total_epochs = args.epochs
    warmup_epochs = config.WARMUP_EPOCHS

    def lr_lambda(current_epoch: int) -> float:
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(config.MIN_LR / args.lr_head, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Tracking metrics
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_balanced_acc": [],
        "val_macro_f1": [],
        "lr_backbone": [],
        "lr_head": [],
        "gem_p": [],
        "time_seconds": []
    }

    best_score = float("-inf")
    best_epoch = 0
    epochs_no_improve = 0

    print(f"\n🎯 Training Parameters:")
    print(f"   • Epochs:            {total_epochs}")
    print(f"   • Physical Batch:    {args.batch_size}")
    print(f"   • Gradient Accum:    {args.accum_steps} (Effective batch: {args.batch_size * args.accum_steps})")
    print(f"   • LR Backbone:       {args.lr_backbone}")
    print(f"   • LR Head:           {args.lr_head}")
    print(f"   • Label Smoothing:   {args.label_smoothing}")
    print(f"   • Early Stopping:    {args.patience} epochs patience")

    print("\n" + "=" * 80)
    print(f"  STARTING TRAINING PIPELINE")
    print("=" * 80)

    for epoch in range(1, total_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            accum_steps=args.accum_steps
        )

        val_loss, val_acc, val_balanced_acc, val_macro_f1 = evaluate_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch
        )

        scheduler.step()
        epoch_time = time.time() - t0

        curr_lr_bb = optimizer.param_groups[0]["lr"]
        curr_lr_head = optimizer.param_groups[1]["lr"]
        curr_gem_p = model.gem_pool.p.data.item()

        # Log
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_balanced_acc"].append(val_balanced_acc)
        history["val_macro_f1"].append(val_macro_f1)
        history["lr_backbone"].append(curr_lr_bb)
        history["lr_head"].append(curr_lr_head)
        history["gem_p"].append(curr_gem_p)
        history["time_seconds"].append(epoch_time)

        print(
            f"Epoch {epoch:02d}/{total_epochs:02d} [{epoch_time:.1f}s] | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
            f"Balanced Acc: {val_balanced_acc:.2f}% | F1: {val_macro_f1:.2f}% | GeM p: {curr_gem_p:.2f}"
        )

        # Composite selection score ensures both high accuracy and balanced representation
        selection_score = (0.60 * val_acc) + (0.25 * val_balanced_acc) + (0.15 * val_macro_f1)

        # Model Checkpointing on Selection Score
        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            epochs_no_improve = 0

            save_checkpoint(
                model=model,
                save_path=config.BEST_MODEL_PATH,
                epoch=epoch,
                metrics={
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_balanced_accuracy": val_balanced_acc,
                    "val_macro_f1": val_macro_f1,
                    "selection_score": selection_score,
                    "gem_p": curr_gem_p
                },
                optimizer=optimizer,
                scheduler=scheduler
            )
            print(f"   ⭐ New best model saved! Score: {selection_score:.2f} (Acc: {val_acc:.2f}%, BalAcc: {val_balanced_acc:.2f}%)")
        else:
            epochs_no_improve += 1
            print(f"   ⏳ No improvement for {epochs_no_improve}/{args.patience} epochs (Best Score: {best_score:.2f})")
            if epochs_no_improve >= args.patience:
                print(f"\n⛔ Early stopping triggered after {epochs_no_improve} epochs without improvement.")
                break

        # Periodic VRAM Cache Clearing
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Save training log to CSV
    log_df = pd.DataFrame(history)
    log_df.to_csv(config.TRAINING_LOG_PATH, index=False)
    print(f"\n📊 Training history logged to: {config.TRAINING_LOG_PATH}")
    print(f"🏆 Best Validation Selection Score: {best_score:.2f} at Epoch {best_epoch}")


if __name__ == "__main__":
    main()

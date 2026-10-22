"""
Comprehensive Model Evaluation Script for GalaxyResNet34.
Evaluates accuracy on the test set with optional Test-Time Augmentation (TTA).
Outputs:
- Top-1 Accuracy, Balanced Accuracy, Macro-F1
- Per-class precision, recall, and support table
- High-resolution Confusion Matrix saved to outputs/plots/
"""

import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from dataset import get_dataloaders, get_tta_transforms
from model import GalaxyResNet34, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GalaxyResNet34 on Test Set")
    parser.add_argument("--checkpoint", type=str, default=str(config.BEST_MODEL_PATH), help="Path to model checkpoint")
    parser.add_argument("--use-tta", action="store_true", default=config.USE_TTA, help="Use Test-Time Augmentation")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device")
    return parser.parse_args()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    save_path: Path,
    title: str = "Galaxy Morphology Confusion Matrix"
):
    """Generate and save a styled confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(8, 7), facecolor="#0a1628")
    ax.set_facecolor("#0a1628")

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="#cbd5e1")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#cbd5e1")

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Morphology",
        xlabel="Predicted Morphology"
    )

    ax.set_title(title, fontsize=14, fontweight="bold", color="#f8fafc", pad=16)
    ax.tick_params(colors="#cbd5e1", labelsize=11)
    ax.yaxis.label.set_color("#f8fafc")
    ax.xaxis.label.set_color("#f8fafc")

    # Format numbers inside cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = 100.0 * val / np.maximum(cm[i].sum(), 1)
            ax.text(
                j, i, f"{val:,}\n({pct:.1f}%)",
                ha="center", va="center",
                color="white" if val > thresh else "#1e293b",
                fontsize=10, fontweight="bold"
            )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"📊 Confusion matrix saved to: {save_path}")


@torch.no_grad()
def evaluate_standard(
    model: GalaxyResNet34,
    test_loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standard forward evaluation on test dataset."""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    for images, labels in tqdm(test_loader, desc="Testing (Standard)"):
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(images)
            probs = F.softmax(logits, dim=1)

        _, preds = logits.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_targets), np.array(all_preds), np.array(all_probs)


@torch.no_grad()
def evaluate_with_tta(
    model: GalaxyResNet34,
    test_loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Test-Time Augmentation (TTA) evaluation.
    Averages predictions over 4 orthogonal rotations (0°, 90°, 180°, 270°).
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    # 4 rotation transforms directly on tensor batches
    for images, labels in tqdm(test_loader, desc="Testing (TTA 4-View)"):
        images = images.to(device, non_blocking=True)
        batch_probs_list = []

        # View 1: 0 degrees
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            probs_0 = F.softmax(model(images), dim=1)
            batch_probs_list.append(probs_0)

        # View 2: 90 degrees rotation
        img_90 = torch.rot90(images, k=1, dims=[2, 3])
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            probs_90 = F.softmax(model(img_90), dim=1)
            batch_probs_list.append(probs_90)

        # View 3: 180 degrees rotation
        img_180 = torch.rot90(images, k=2, dims=[2, 3])
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            probs_180 = F.softmax(model(img_180), dim=1)
            batch_probs_list.append(probs_180)

        # View 4: 270 degrees rotation
        img_270 = torch.rot90(images, k=3, dims=[2, 3])
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            probs_270 = F.softmax(model(img_270), dim=1)
            batch_probs_list.append(probs_270)

        # Average probabilities across the 4 rotated orientations
        mean_probs = torch.stack(batch_probs_list, dim=0).mean(dim=0)
        preds = mean_probs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.numpy())
        all_probs.extend(mean_probs.cpu().numpy())

    return np.array(all_targets), np.array(all_preds), np.array(all_probs)


def main():
    args = parse_args()

    # Device
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"\n📡 Evaluating on device: {device}")

    # Checkpoint
    checkpoint_path = Path(args.checkpoint)
    model, checkpoint = load_checkpoint(checkpoint_path, device=device)

    # Load test dataset
    _, _, test_loader, class_names, _ = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=config.NUM_WORKERS
    )

    # Run evaluation
    if args.use_tta:
        print("\n🔄 Running Test-Time Augmentation (4-Orientation TTA)...")
        y_true, y_pred, y_probs = evaluate_with_tta(model, test_loader, device)
        eval_tag = "TTA"
    else:
        print("\n⚡ Running Standard Single-Pass Evaluation...")
        y_true, y_pred, y_probs = evaluate_standard(model, test_loader, device)
        eval_tag = "Standard"

    # Compute Metrics
    acc = 100.0 * accuracy_score(y_true, y_pred)
    balanced_acc = 100.0 * balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    print("\n" + "=" * 65)
    print(f"  FINAL TEST RESULTS ({eval_tag})")
    print("=" * 65)
    print(f"  • Top-1 Accuracy:     {acc:.2f}%")
    print(f"  • Balanced Accuracy:  {balanced_acc:.2f}%")
    print(f"  • Macro-F1 Score:     {report_dict['macro avg']['f1-score']*100:.2f}%")
    print("\nDetailed Classification Report:")
    print(report_text)
    print("=" * 65)

    # Plot & save confusion matrix
    cm_path = config.PLOTS_DIR / f"confusion_matrix_{eval_tag.lower()}.png"
    plot_confusion_matrix(cm, class_names, cm_path, title=f"Galaxy Zoo 2 Test Set ({eval_tag})")

    # Save metrics table
    res_df = pd.DataFrame([{
        "mode": eval_tag,
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "macro_f1": report_dict["macro avg"]["f1-score"] * 100,
        "macro_precision": report_dict["macro avg"]["precision"] * 100,
        "macro_recall": report_dict["macro avg"]["recall"] * 100,
    }])
    metrics_path = config.LOGS_DIR / f"test_metrics_{eval_tag.lower()}.csv"
    res_df.to_csv(metrics_path, index=False)
    print(f"📁 Metrics exported to: {metrics_path}")


if __name__ == "__main__":
    main()

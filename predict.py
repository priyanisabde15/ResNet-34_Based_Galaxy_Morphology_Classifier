"""
Inference CLI Tool for Galaxy Morphology Classification.
Classify individual galaxy images or entire directories using GalaxyResNet34.
Supports Test-Time Augmentation (TTA).
"""

import os
import argparse
from pathlib import Path
from PIL import Image
import pandas as pd
import torch

import config
from dataset import get_astronomical_transforms, get_tta_transforms
from model import GalaxyResNet34, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Predict galaxy morphology using GalaxyResNet34")
    parser.add_argument("--image", type=str, default=None, help="Path to single galaxy image (.jpg, .png)")
    parser.add_argument("--dir", type=str, default=None, help="Path to directory of galaxy images")
    parser.add_argument("--checkpoint", type=str, default=str(config.BEST_MODEL_PATH), help="Path to model checkpoint")
    parser.add_argument("--use-tta", action="store_true", default=True, help="Enable Test-Time Augmentation")
    parser.add_argument("--output", type=str, default="predictions.csv", help="CSV output path for directory inference")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def predict_single_image(
    image_path: Path,
    model: GalaxyResNet34,
    device: torch.device,
    use_tta: bool = True
) -> dict:
    """Classify a single galaxy image and return class probabilities."""
    img = Image.open(image_path).convert("RGB")

    if use_tta:
        tta_transforms = get_tta_transforms()
        pred_idx, probs = model.predict_with_tta(img, tta_transforms, device)
        probs_np = probs.cpu().numpy()
    else:
        _, eval_transform = get_astronomical_transforms()
        tensor = eval_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = model.predict_proba(tensor).squeeze(0)
            pred_idx = int(probs.argmax().item())
            probs_np = probs.cpu().numpy()

    return {
        "filename": image_path.name,
        "predicted_class": config.CLASS_NAMES[pred_idx],
        "confidence": float(probs_np[pred_idx]),
        "probabilities": {
            config.CLASS_NAMES[i]: float(probs_np[i]) for i in range(len(config.CLASS_NAMES))
        },
        "description": config.CLASS_DESCRIPTIONS[pred_idx],
        "icon": config.CLASS_ICONS[pred_idx]
    }


def main():
    args = parse_args()

    if not args.image and not args.dir:
        print("❌ Error: You must supply either --image <path> or --dir <path>.")
        return

    # Device
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        model, _ = load_checkpoint(checkpoint_path, device=device)
    else:
        print(f"\n⚠️ Notice: Trained checkpoint not found at: {checkpoint_path}")
        print("   Running with ImageNet pretrained ResNet-34 backbone for demonstration.")
        print("   (Run 'python train.py' to train and generate the fine-tuned best_model.pth checkpoint.)\n")
        from model import build_model
        model = build_model(device=device, pretrained=True)

    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"❌ Image not found: {img_path}")
            return

        res = predict_single_image(img_path, model, device, use_tta=args.use_tta)

        print("\n" + "=" * 65)
        print(f"🌌 GALAXY MORPHOLOGY PREDICTION: {res['filename']}")
        print("=" * 65)
        print(f"  Classification: {res['icon']} {res['predicted_class']}")
        print(f"  Confidence:     {res['confidence'] * 100:.2f}%")
        print(f"  Description:    {res['description']}")
        print("\nProbability Distribution:")
        for cname, prob in res["probabilities"].items():
            bar_len = int(prob * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"  {cname:<15} [{bar}] {prob * 100:5.2f}%")
        print("=" * 65)

    elif args.dir:
        dir_path = Path(args.dir)
        if not dir_path.exists() or not dir_path.is_dir():
            print(f"❌ Directory not found: {dir_path}")
            return

        image_files = [
            f for f in dir_path.iterdir()
            if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        if not image_files:
            print(f"❌ No valid image files (.jpg, .png) found in {dir_path}")
            return

        print(f"\n📂 Processing {len(image_files):,} images from {dir_path.name}...")
        records = []

        for f in image_files:
            try:
                res = predict_single_image(f, model, device, use_tta=args.use_tta)
                row = {
                    "filename": res["filename"],
                    "predicted_class": res["predicted_class"],
                    "confidence_pct": round(res["confidence"] * 100, 2)
                }
                for cname, prob in res["probabilities"].items():
                    row[f"prob_{cname.lower().replace('/', '_')}"] = round(prob, 4)
                records.append(row)
            except Exception as e:
                print(f"⚠️ Error processing {f.name}: {e}")

        df = pd.DataFrame(records)
        df.to_csv(args.output, index=False)
        print(f"✅ Predictions exported to: {args.output}")


if __name__ == "__main__":
    main()

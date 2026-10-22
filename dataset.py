"""
Galaxy Zoo 2 Dataset and Astronomical Preprocessing Pipeline.
Strictly uses REAL Galaxy Zoo 2 data. NO synthetic/simulated data.
Features:
- Dieleman 207x207 center cropping (crops out empty SDSS black borders)
- Space symmetry augmentations: full continuous 360° rotation + dual-axis flipping
- Test-Time Augmentation (TTA) transforms generator
- Stratified train/val/test splits
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from sklearn.model_selection import train_test_split

import config


class GalaxyZoo2Dataset(Dataset):
    """
    Real Galaxy Zoo 2 Dataset for Galaxy Morphology Classification.
    Parses SDSS Galaxy Zoo 2 volunteer consensus votes from training_solutions_rev1.csv
    and loads matching images from images_training_rev1.
    """

    def __init__(
        self,
        samples: List[Tuple[str, int, float]],
        transform: Optional[transforms.Compose] = None
    ):
        """
        Args:
            samples: List of (image_path, class_label, confidence_score)
            transform: PyTorch torchvision transforms
        """
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label, _ = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_astronomical_transforms(
    center_crop_size: int = config.CENTER_CROP_SIZE,
    input_size: int = config.INPUT_SIZE
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Create training and validation/test transforms tailored for astronomical imagery.
    
    Training pipeline:
    1. CenterCrop(207): Focuses on the galactic center, removing ~70% empty SDSS background.
    2. Resize(224): Standard ResNet input dimension.
    3. RandomHorizontalFlip & RandomVerticalFlip: Invariance to telescope orientation.
    4. RandomRotation(180): Continuous full 360° rotational invariance.
    5. Light ColorJitter: Accounts for minor variations in SDSS camera filter throughput.
    6. Normalization: ImageNet statistics.
    """
    train_transform = transforms.Compose([
        transforms.CenterCrop(center_crop_size),
        transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180, interpolation=InterpolationMode.BILINEAR),
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.CenterCrop(center_crop_size),
        transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, eval_transform


def get_tta_transforms(
    center_crop_size: int = config.CENTER_CROP_SIZE,
    input_size: int = config.INPUT_SIZE
) -> List[transforms.Compose]:
    """
    Generates 4 distinct orthogonal views for Test-Time Augmentation (TTA).
    Averages predictions across rotations to boost test accuracy without retraining.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    base = [
        transforms.CenterCrop(center_crop_size),
        transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BILINEAR),
    ]

    tta_transforms = [
        # 1. Standard view (0°)
        transforms.Compose(base + [transforms.ToTensor(), normalize]),
        # 2. 90° rotation
        transforms.Compose(base + [
            transforms.RandomRotation(degrees=(90, 90)),
            transforms.ToTensor(),
            normalize
        ]),
        # 3. 180° rotation
        transforms.Compose(base + [
            transforms.RandomRotation(degrees=(180, 180)),
            transforms.ToTensor(),
            normalize
        ]),
        # 4. 270° rotation
        transforms.Compose(base + [
            transforms.RandomRotation(degrees=(270, 270)),
            transforms.ToTensor(),
            normalize
        ]),
    ]
    return tta_transforms


def load_galaxy_zoo_samples(
    data_dir: Path = config.DATA_DIR,
    csv_file: Path = config.CSV_FILE,
    image_dir: Path = config.IMAGE_DIR,
    num_classes: int = config.NUM_CLASSES,
    min_confidence: float = config.MIN_VOTE_CONFIDENCE
) -> List[Tuple[str, int, float]]:
    """
    Loads and matches real Galaxy Zoo 2 image files with CSV consensus labels.
    STRICTLY REAL DATA ONLY.
    """
    if not csv_file.exists():
        raise FileNotFoundError(
            f"Galaxy Zoo 2 label CSV not found at: {csv_file}\n"
            "Please ensure training_solutions_rev1.csv is present."
        )

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Galaxy Zoo 2 image directory not found at: {image_dir}\n"
            "Please ensure images_training_rev1 folder is present."
        )

    print(f"📊 Reading morphology labels from: {csv_file}")
    df = pd.read_csv(csv_file)
    df["GalaxyID"] = df["GalaxyID"].astype(str)

    # Map labels:
    # Class1.1 = Smooth
    # Class1.2 = Featured / Disk
    # Class1.3 = Star / Artifact
    labels_dict: Dict[str, Tuple[int, float]] = {}

    for _, row in df.iterrows():
        gid = row["GalaxyID"]
        p_smooth = float(row.get("Class1.1", 0.0))
        p_featured = float(row.get("Class1.2", 0.0))
        p_artifact = float(row.get("Class1.3", 0.0))

        if num_classes == 3:
            probs = [p_smooth, p_featured, p_artifact]
            class_idx = int(np.argmax(probs))
            conf = float(probs[class_idx])
            if conf >= min_confidence:
                labels_dict[gid] = (class_idx, conf)
        else:
            # 2-class setup (ignoring artifacts or ambiguous entries)
            if p_artifact > 0.5:
                continue
            if p_smooth > p_featured:
                conf = p_smooth
                class_idx = 0
            else:
                conf = p_featured
                class_idx = 1
            if conf >= min_confidence:
                labels_dict[gid] = (class_idx, conf)

    print(f"📂 Scanning image directory: {image_dir}")
    available_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"📸 Found {len(available_files):,} image files in folder")

    samples: List[Tuple[str, int, float]] = []
    for fname in available_files:
        gid = os.path.splitext(fname)[0]
        if gid in labels_dict:
            class_idx, conf = labels_dict[gid]
            full_path = str(image_dir / fname)
            samples.append((full_path, class_idx, conf))

    if len(samples) == 0:
        raise ValueError("Zero images matched the CSV labels. Verify image filenames match GalaxyID values.")

    print(f"✅ Successfully matched {len(samples):,} real Galaxy Zoo 2 images to verified labels.")
    
    # Class distribution summary
    counts = np.bincount([s[1] for s in samples], minlength=num_classes)
    for idx, c in enumerate(counts):
        cname = config.CLASS_NAMES[idx]
        pct = 100.0 * c / len(samples)
        print(f"   • {cname} (Class {idx}): {c:,} samples ({pct:.2f}%)")

    return samples


def get_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    data_dir: Path = config.DATA_DIR,
    pin_memory: bool = config.PIN_MEMORY
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], np.ndarray]:
    """
    Build stratified training, validation, and test dataloaders for Galaxy Zoo 2.
    """
    samples = load_galaxy_zoo_samples(data_dir=data_dir)
    labels = np.array([s[1] for s in samples])
    indices = np.arange(len(samples))

    # Stratified split: 70% train, 15% val, 15% test
    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        indices,
        labels,
        test_size=(config.VAL_RATIO + config.TEST_RATIO),
        stratify=labels,
        random_state=config.RANDOM_SEED
    )

    test_relative_ratio = config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_relative_ratio,
        stratify=temp_labels,
        random_state=config.RANDOM_SEED
    )

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]

    train_transform, eval_transform = get_astronomical_transforms()

    train_dataset = GalaxyZoo2Dataset(train_samples, transform=train_transform)
    val_dataset = GalaxyZoo2Dataset(val_samples, transform=eval_transform)
    test_dataset = GalaxyZoo2Dataset(test_samples, transform=eval_transform)

    print(f"\n📦 Data Splits Summary:")
    print(f"   • Training set:   {len(train_dataset):,} samples")
    print(f"   • Validation set: {len(val_dataset):,} samples")
    print(f"   • Test set:       {len(test_dataset):,} samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Balanced weights: Smooth and Featured/Disk are nearly balanced (~43% vs ~57%).
    # Artifact is tiny (~0.10%), so we assign it a modest weight (2.0) to prevent gradient explosion
    # and avoid degenerate minority class collapse.
    if config.NUM_CLASSES == 3:
        class_weights = np.array([1.0, 1.0, 2.0], dtype=np.float32)
    else:
        class_weights = np.array([1.0, 1.0], dtype=np.float32)

    return train_loader, val_loader, test_loader, config.CLASS_NAMES, class_weights

"""
Configuration settings for ResNet-34 Galaxy Morphology Classifier
Optimized for NVIDIA GeForce RTX 3050 (4GB VRAM) and real Galaxy Zoo 2 dataset.
"""

import os
from pathlib import Path
import torch

# ============================================================================
# PROJECT & DATA PATHS
# ============================================================================

# Base project root
PROJECT_ROOT = Path(__file__).resolve().parent

# Source path to the real Galaxy Zoo 2 dataset in sibling folder
DEFAULT_DATA_DIR = Path(r"C:\MEGA PROJECTS\MACHINE UNLEARNING USING GALAXY MORPHOLGY USING CNN (RESNET34)_Project\data")
DATA_DIR = Path(os.environ.get("GALAXY_DATA_DIR", str(DEFAULT_DATA_DIR)))

# Image directory and CSV labels file
IMAGE_DIR = DATA_DIR / "images_training_rev1"
CSV_FILE = DATA_DIR / "training_solutions_rev1.csv"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOGS_DIR = OUTPUT_DIR / "logs"
PLOTS_DIR = OUTPUT_DIR / "plots"

for d in [OUTPUT_DIR, CHECKPOINT_DIR, LOGS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Best model checkpoint path
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"
TRAINING_LOG_PATH = LOGS_DIR / "training_log.csv"

# ============================================================================
# HARDWARE & ACCELERATION (RTX 3050 4GB VRAM)
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()  # Automatic Mixed Precision for 4GB VRAM
NUM_WORKERS = 2  # Balanced for Windows DataLoader
PIN_MEMORY = torch.cuda.is_available()

# Physical batch size on GPU (16 fits comfortably in ~2.2 GB VRAM)
BATCH_SIZE = 16

# Gradient accumulation steps (16 * 2 = effective batch size of 32)
GRADIENT_ACCUMULATION_STEPS = 2

# ============================================================================
# DATASET & MORPHOLOGY CLASSES
# ============================================================================

# Number of classes: 3 (Smooth, Featured/Disk, Artifact) or 2 (Smooth, Featured/Disk)
NUM_CLASSES = 3

if NUM_CLASSES == 3:
    CLASS_NAMES = ["Smooth", "Featured/Disk", "Artifact"]
    CLASS_DESCRIPTIONS = [
        "Completely smooth, rounded elliptical galaxy with no spiral structure",
        "Disk galaxy with visible features: spiral arms, central bar, or ring structure",
        "Star, cosmic ray defect, or unclassifiable astronomical artifact"
    ]
    CLASS_COLORS = ["#60a5fa", "#34d399", "#f87171"]
    CLASS_ICONS = ["🔵", "🌀", "⭐"]
else:
    CLASS_NAMES = ["Smooth", "Featured/Disk"]
    CLASS_DESCRIPTIONS = [
        "Completely smooth, rounded elliptical galaxy with no spiral structure",
        "Disk galaxy with visible features: spiral arms, central bar, or ring structure"
    ]
    CLASS_COLORS = ["#60a5fa", "#34d399"]
    CLASS_ICONS = ["🔵", "🌀"]

# Astronomical Preprocessing
RAW_IMAGE_SIZE = 424          # Original SDSS image resolution
CENTER_CROP_SIZE = 207        # Center crop removing outer black background
INPUT_SIZE = 224              # Final input resolution to ResNet-34

# Dataset Split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Confidence filter threshold (Optional: set > 0.0 to filter low-confidence votes, e.g., 0.50)
MIN_VOTE_CONFIDENCE = 0.0

# ============================================================================
# MODEL ARCHITECTURE (ResNet-34 + GeM Pooling)
# ============================================================================

BACKBONE = "resnet34"
USE_PRETRAINED = True
GEM_P_INIT = 3.0              # Initial power p for Generalized Mean (GeM) Pooling
GEM_EPS = 1e-6
DROPOUT_RATE_1 = 0.3
DROPOUT_RATE_2 = 0.2
HEAD_HIDDEN_DIM = 256

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

NUM_EPOCHS = 35
LR_BACKBONE = 3e-5            # Gentle fine-tuning for pretrained features
LR_HEAD = 3e-4                # Higher learning rate for new classification head
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2             # Linear learning rate warmup
MIN_LR = 1e-6                 # Minimum LR for Cosine Annealing
EARLY_STOPPING_PATIENCE = 10  # Epochs to wait before early stopping
LABEL_SMOOTHING = 0.08        # Regularization for crowd-sourced ambiguous votes

# ============================================================================
# TEST-TIME AUGMENTATION (TTA)
# ============================================================================

USE_TTA = True
TTA_STEPS = 4                 # 4-rotation averaging (0°, 90°, 180°, 270°)

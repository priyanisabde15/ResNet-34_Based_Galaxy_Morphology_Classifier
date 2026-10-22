"""
Galaxy Morphology ResNet-34 Architecture with GeM Pooling.
Enhancements:
- Learnable Generalized Mean (GeM) Pooling replaces standard AvgPool2d.
- Layered normalization and GELU non-linearities in the classification head.
- Mixed-precision compatible.
- Integrated Test-Time Augmentation (TTA) forward method.
"""

from pathlib import Path
from typing import Optional, Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image

import config


class GeMPooling(nn.Module):
    """
    Generalized Mean (GeM) Pooling Layer:
    f(x) = (1/|X| * sum(x^p))^(1/p)
    
    When p = 1: Standard Average Pooling (effective for diffuse elliptical galaxies).
    When p -> inf: Max Pooling (effective for high-contrast spiral knots and bars).
    By making p a learnable parameter initialized to 3.0, the network dynamically
    learns how to pool galaxy features across different morphological structures.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super(GeMPooling, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clamp for numerical stability with power operations
        x_clamped = x.clamp(min=self.eps)
        # Average pool of x^p
        pooled = F.avg_pool2d(
            x_clamped.pow(self.p),
            (x.size(-2), x.size(-1))
        )
        # Result^(1/p)
        return pooled.pow(1.0 / self.p)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class GalaxyResNet34(nn.Module):
    """
    ResNet-34 based deep CNN tailored for Galaxy Morphology Classification.
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        pretrained: bool = config.USE_PRETRAINED,
        p_init: float = config.GEM_P_INIT
    ):
        super(GalaxyResNet34, self).__init__()
        self.num_classes = num_classes

        # Load ResNet-34 backbone
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        base_resnet = models.resnet34(weights=weights)

        # Feature extractor layers
        self.conv1 = base_resnet.conv1
        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool
        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2
        self.layer3 = base_resnet.layer3
        self.layer4 = base_resnet.layer4

        # Replace standard AdaptiveAvgPool with learnable GeM Pooling
        self.gem_pool = GeMPooling(p=p_init, eps=config.GEM_EPS)

        # Modern high-capacity classification head
        in_features = base_resnet.fc.in_features  # 512 for ResNet-34
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=config.DROPOUT_RATE_1),
            nn.Linear(in_features, config.HEAD_HIDDEN_DIM),
            nn.GELU(),
            nn.BatchNorm1d(config.HEAD_HIDDEN_DIM),
            nn.Dropout(p=config.DROPOUT_RATE_2),
            nn.Linear(config.HEAD_HIDDEN_DIM, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pooled 512-dimensional feature embedding."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.gem_pool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw class logits."""
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute normalized softmax probabilities."""
        self.eval()
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    @torch.no_grad()
    def predict_with_tta(
        self,
        pil_image: Image.Image,
        tta_transforms: list,
        device: torch.device
    ) -> Tuple[int, np_array_probs := torch.Tensor]:
        """
        Run multi-view Test-Time Augmentation on a single PIL image.
        Returns the ensemble predicted class index and class probabilities.
        """
        self.eval()
        probs_list = []
        for transform in tta_transforms:
            tensor = transform(pil_image).unsqueeze(0).to(device)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = self.forward(tensor)
                probs = F.softmax(logits, dim=1)
                probs_list.append(probs)

        # Average probabilities across all TTA views
        mean_probs = torch.stack(probs_list, dim=0).mean(dim=0).squeeze(0)
        pred_class = int(mean_probs.argmax().item())
        return pred_class, mean_probs


def build_model(
    num_classes: int = config.NUM_CLASSES,
    device: Optional[torch.device] = None,
    pretrained: bool = config.USE_PRETRAINED
) -> GalaxyResNet34:
    """
    Factory to construct and initialize GalaxyResNet34 on the target device.
    """
    if device is None:
        device = config.DEVICE

    model = GalaxyResNet34(num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n🧠 Initialized GalaxyResNet34:")
    print(f"   • Total parameters:     {total_params:,}")
    print(f"   • Trainable parameters: {trainable_params:,}")
    print(f"   • Learnable GeM p-init: {config.GEM_P_INIT}")
    print(f"   • Target device:        {device}")

    return model


def save_checkpoint(
    model: GalaxyResNet34,
    save_path: Path,
    epoch: int,
    metrics: Dict,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None
) -> None:
    """Save full training checkpoint."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "gem_p": model.gem_pool.p.data.cpu().item(),
        "num_classes": model.num_classes,
        "metrics": metrics
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, save_path)
    print(f"💾 Checkpoint saved to: {save_path} (Epoch {epoch})")


def load_checkpoint(
    checkpoint_path: Path,
    model: Optional[GalaxyResNet34] = None,
    device: Optional[torch.device] = None
) -> Tuple[GalaxyResNet34, Dict]:
    """Load model weights and metadata from checkpoint."""
    if device is None:
        device = config.DEVICE

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = checkpoint.get("num_classes", config.NUM_CLASSES)

    if model is None:
        model = GalaxyResNet34(num_classes=num_classes, pretrained=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    gem_p = checkpoint.get("gem_p", None)
    gem_info = f" (GeM p={gem_p:.3f})" if gem_p is not None else ""
    print(f"✅ Loaded GalaxyResNet34 from {checkpoint_path.name}{gem_info}")

    return model, checkpoint

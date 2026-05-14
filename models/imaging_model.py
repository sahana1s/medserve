"""
models/imaging_model.py — Chest X-ray classifier using torchxrayvision or ResNet18.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import Any
from models.base import BaseInferenceEngine, ModelMetadata
from system.request import ModelType


# ---------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------

LABELS = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass","Nodule",
    "Pneumonia","Pneumothorax","Consolidation","Edema","Emphysema",
    "Fibrosis","Pleural_Thickening","Hernia",
]


# ---------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ---------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------

class _ImagingBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        try:
            import torchxrayvision as xrv

            self.net = xrv.models.DenseNet(weights="densenet121-res224-all")
            self.use_txrv = True
            self.txrv_pathologies = self.net.pathologies

            print("[Imaging] torchxrayvision DenseNet121 loaded")

        except ImportError:
            import torchvision.models as tv

            self.net = tv.resnet18(weights=tv.ResNet18_Weights.DEFAULT)

            for p in self.net.parameters():
                p.requires_grad = False

            self.net.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(512, 14)
            )

            self.use_txrv = False
            print("[Imaging] Using ResNet18 fallback")

    def forward(self, x):
        if self.use_txrv:
            gray = x.mean(dim=1, keepdim=True)
            gray = (gray - 0.5) * 2048.0

            raw = self.net(gray)

            result = torch.zeros(x.shape[0], 14, device=x.device)

            for i, lbl in enumerate(LABELS):
                key = lbl.replace("_", " ").lower()

                for j, p in enumerate(self.txrv_pathologies or []):
                    if p and key in p.lower():
                        result[:, i] = torch.sigmoid(raw[:, j])
                        break

            return result

        return torch.sigmoid(self.net(x))


# ---------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------

class ImagingInferenceEngine(BaseInferenceEngine):

    # ---------------------------------------------------------
    # LOAD MODEL
    # ---------------------------------------------------------
    def _load_model(self):
        self.model = _ImagingBackbone()

        if self.model_path:
            try:
                state = torch.load(self.model_path, map_location=self.device)
                self.model.net.load_state_dict(state, strict=False)
                print(f"[Imaging] Loaded weights from {self.model_path}")
            except Exception as e:
                print(f"[Imaging] Failed to load weights: {e}")

        self.model.to(self.device)

    # ---------------------------------------------------------
    # FORWARD
    # ---------------------------------------------------------
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        # IMPORTANT: force FP32 for stability (txrv breaks in fp16)
        x = x.to(torch.float32)
        return self.model(x)

    # ---------------------------------------------------------
    # PREPROCESS (SINGLE SAMPLE ONLY)
    # ---------------------------------------------------------
    def preprocess(self, raw_input: Any) -> torch.Tensor:
        from PIL import Image

        if isinstance(raw_input, str):
            raw_input = Image.open(raw_input).convert("RGB")

        if hasattr(raw_input, "convert"):  # PIL image
            return TRANSFORM(raw_input)

        if isinstance(raw_input, torch.Tensor):
            return raw_input.float()

        return torch.tensor(raw_input, dtype=torch.float32)

    # ---------------------------------------------------------
    # CRITICAL FIX: SAFE BATCHING (FIXED SHAPE)
    # ---------------------------------------------------------
    def _prepare_batch(self, inputs):
        # Imaging is fixed-shape → safe concat
        tensors = []

        for x in inputs:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            else:
                x = x.float()

            tensors.append(x)

        return torch.cat(tensors, dim=0).to(self.device)

    # ---------------------------------------------------------
    # METADATA
    # ---------------------------------------------------------
    @property
    def metadata(self):
        name = (
            "DenseNet121-TXRVision"
            if getattr(getattr(self, "model", None), "use_txrv", False)
            else "ResNet18-pretrained"
        )

        return ModelMetadata(
            model_type=ModelType.IMAGING,
            model_name=name,
            input_shape=(3, 224, 224),
        )


# ---------------------------------------------------------------------
# TRAINING (unchanged but safe)
# ---------------------------------------------------------------------

def train_imaging_model(train_loader, val_loader, epochs=10, lr=1e-4,
                        save_path="results/weights/imaging_model.pt", device="cuda"):

    import os

    model = _ImagingBackbone().to(device)

    if model.use_txrv:
        for p in model.net.parameters():
            p.requires_grad = False
        for p in list(model.net.parameters())[-20:]:
            p.requires_grad = True

    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    crit = nn.BCELoss()
    best = float("inf")

    for ep in range(epochs):
        model.train()
        tl = 0.0

        for x, y in train_loader:
            opt.zero_grad()
            loss = crit(model(x.to(device)), y.float().to(device))
            loss.backward()
            opt.step()
            tl += loss.item()

        model.eval()
        vl = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                vl += crit(model(x.to(device)), y.float().to(device)).item()

        tl /= len(train_loader)
        vl /= len(val_loader)

        print(f"Epoch {ep+1}/{epochs}  train={tl:.4f}  val={vl:.4f}")

        if vl < best:
            best = vl
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print("  -> saved")

    return model

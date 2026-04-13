"""
models/imaging_model.py — Chest X-ray classifier using torchxrayvision pretrained weights.
No training required. Uses DenseNet121 trained on NIH+CheXpert+MIMIC-CXR.
Cite: Cohen et al., TorchXRayVision, MIDL 2022.
Install: pip install torchxrayvision
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import Any
from models.base import BaseInferenceEngine, ModelMetadata
from system.request import ModelType

LABELS = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass","Nodule",
    "Pneumonia","Pneumothorax","Consolidation","Edema","Emphysema",
    "Fibrosis","Pleural_Thickening","Hernia",
]

TRANSFORM = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])


class _ImagingBackbone(nn.Module):
    """
    Tries torchxrayvision first (best pretrained clinical weights).
    Falls back to pretrained ResNet18 if txrv not installed.
    """
    def __init__(self):
        super().__init__()
        try:
            import torchxrayvision as xrv
            self.net      = xrv.models.DenseNet(weights="densenet121-res224-all")
            self.use_txrv = True
            self.txrv_pathologies = self.net.pathologies
            print("[Imaging] torchxrayvision DenseNet121 loaded (NIH+CheXpert+MIMIC-CXR pretrained)")
        except ImportError:
            import torchvision.models as tv
            b = tv.resnet18(weights=tv.ResNet18_Weights.DEFAULT)
            for p in b.parameters(): p.requires_grad = False
            b.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 14))
            self.net      = b
            self.use_txrv = False
            print("[Imaging] torchxrayvision not found — using ResNet18 (pip install torchxrayvision for better weights)")

    def forward(self, x):
        if self.use_txrv:
            # txrv expects (B,1,H,W) in [-1024, 1024]
            gray   = x.mean(dim=1, keepdim=True)
            gray   = (gray - 0.5) * 2048.0
            raw    = self.net(gray)                   # (B, n_txrv_pathologies)
            result = torch.zeros(x.shape[0], 14, device=x.device)
            for i, lbl in enumerate(LABELS):
                key = lbl.replace("_"," ").lower()
                for j, p in enumerate(self.txrv_pathologies or []):
                    if p and key in (p or "").lower():
                        result[:, i] = torch.sigmoid(raw[:, j]); break
            return result
        else:
            return torch.sigmoid(self.net(x))


class ImagingInferenceEngine(BaseInferenceEngine):
    """ResNet/DenseNet chest X-ray classifier. Input:(B,3,224,224) Output:(B,14) SLA:500ms"""

    def _load_model(self):
        self.model = _ImagingBackbone()
        if self.model_path:
            try:
                self.model.net.load_state_dict(torch.load(self.model_path, map_location=self.device))
                print(f"[Imaging] Loaded fine-tuned weights from {self.model_path}")
            except Exception as e:
                print(f"[Imaging] Could not load {self.model_path} ({e}) — using pretrained")
        self.model.to(self.device)

    # def _forward(self, x): return self.model(x).cpu().float()
    def _forward(self, x):
        # FORCE FP32 — torchxrayvision breaks in FP16
        x = x.to(torch.float32)
        return self.model(x).cpu().float()

    def preprocess(self, raw_input):
        from PIL import Image
        if isinstance(raw_input, str): raw_input = Image.open(raw_input).convert("RGB")
        if hasattr(raw_input, "convert"): return TRANSFORM(raw_input).unsqueeze(0)
        t = raw_input if isinstance(raw_input, torch.Tensor) else torch.tensor(raw_input, dtype=torch.float32)
        return t.unsqueeze(0) if t.dim() == 3 else t

    @property
    def metadata(self):
        name = "DenseNet121-TXRVision" if getattr(getattr(self,'model',None),'use_txrv',False) else "ResNet18-pretrained"
        return ModelMetadata(model_type=ModelType.IMAGING, model_name=name, input_shape=(3,224,224))


def train_imaging_model(train_loader, val_loader, epochs=10, lr=1e-4,
                        save_path="results/weights/imaging_model.pt", device="cuda"):
    """Optional fine-tuning. Only needed if you want domain-specific weights."""
    import os
    from models.imaging_model import _ImagingBackbone
    model = _ImagingBackbone().to(device)
    if model.use_txrv:
        # Unfreeze last block of DenseNet for fine-tuning
        for p in model.net.parameters(): p.requires_grad = False
        for p in list(model.net.parameters())[-20:]: p.requires_grad = True
    opt  = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    crit = nn.BCELoss()
    best = float("inf")
    for ep in range(epochs):
        model.train(); tl=0.0
        for x,y in train_loader:
            opt.zero_grad(); loss=crit(model(x.to(device)),y.float().to(device))
            loss.backward(); opt.step(); tl+=loss.item()
        model.eval(); vl=0.0
        with torch.no_grad():
            for x,y in val_loader: vl+=crit(model(x.to(device)),y.float().to(device)).item()
        tl/=len(train_loader); vl/=len(val_loader)
        print(f"Epoch {ep+1}/{epochs}  train={tl:.4f}  val={vl:.4f}")
        if vl<best:
            best=vl; os.makedirs(str(save_path).rsplit("/",1)[0],exist_ok=True)
            torch.save(model.state_dict(),save_path); print("  -> saved")
    return model

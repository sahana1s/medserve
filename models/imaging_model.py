"""
models/imaging_model.py — ResNet18 for chest X-ray classification.

Task:    Multilabel classification — detect which of 14 pathologies
         are present in a chest X-ray.

Dataset: NIH ChestX-ray14 (freely available, no credentialing needed)
         https://nihcc.app.box.com/v/ChestXray-NIHCC
         OR MIMIC-CXR (requires PhysioNet access, same as MIMIC-III)

Input:   (batch_size, 3, 224, 224) — standard ImageNet preprocessing
         Normalize with ImageNet mean/std (torchvision handles this)

Output:  (batch_size, 14) — independent probabilities for each pathology
         Labels: Atelectasis, Cardiomegaly, Effusion, Infiltration,
                 Mass, Nodule, Pneumonia, Pneumothorax, Consolidation,
                 Edema, Emphysema, Fibrosis, Pleural Thickening, Hernia

SLA:     500ms (LOW priority — batch-friendly, radiology pre-screening)
Target:  ~0.75 mean AUC across 14 labels (reasonable ResNet18 baseline)

Architecture note: We use a pretrained ResNet18 and replace only the
final FC layer. Fine-tuning the last 2 blocks + classifier is enough
for a research baseline. Don't train from scratch — it wastes time.
"""

import time
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Tuple, List


# ---------------------------------------------------------------------------
# Label definitions
# ---------------------------------------------------------------------------

CHESTXRAY_LABELS: List[str] = [
    "Atelectasis", "Cardiomegaly", "Effusion",    "Infiltration",
    "Mass",        "Nodule",        "Pneumonia",   "Pneumothorax",
    "Consolidation","Edema",        "Emphysema",   "Fibrosis",
    "Pleural_Thickening",           "Hernia",
]
NUM_CLASSES = len(CHESTXRAY_LABELS)  # 14


# ---------------------------------------------------------------------------
# Standard preprocessing transform
# ---------------------------------------------------------------------------

IMAGING_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet statistics
        std=[0.229, 0.224, 0.225],
    ),
])


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class ImagingModel(nn.Module):
    """
    Pretrained ResNet18 adapted for chest X-ray multilabel classification.

    Fine-tuning strategy (week 1: use this, don't overthink):
    1. Load ImageNet pretrained weights
    2. Freeze all layers except layer4 and the new classifier
    3. Replace final FC (1000 classes → 14 classes)
    4. Use sigmoid (not softmax) — these are independent binary predictions

    Why ResNet18 over ResNet50/ViT?
    - Faster inference → easier to hit 500ms SLA
    - Smaller memory footprint → more room for other models on same GPU
    - Performance difference is small for a research baseline
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Freeze all layers first
        for param in backbone.parameters():
            param.requires_grad = False

        # Unfreeze the last residual block (layer4) for fine-tuning
        for param in backbone.layer4.parameters():
            param.requires_grad = True

        # Replace the final classifier
        in_features = backbone.fc.in_features  # 512 for ResNet18
        backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
            # No sigmoid here — use BCEWithLogitsLoss during training
            # Sigmoid applied at inference time for probabilities
        )

        self.backbone    = backbone
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 3, 224, 224)
        Returns:
            logits: (batch_size, 14) — raw logits (no sigmoid)
        """
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

class ImagingInferenceEngine:
    """
    Wraps ImagingModel for production inference.
    Imaging is the LOW priority workload — designed for batching.
    Batch sizes of 8–32 are expected and efficient.
    """

    def __init__(
        self,
        model_path: str = None,
        device:     str = "cuda" if torch.cuda.is_available() else "cpu",
        use_fp16:   bool = True,
    ):
        self.device  = torch.device(device)
        self.use_fp16 = use_fp16 and (device == "cuda")

        self.model = ImagingModel(pretrained=True)

        if model_path:
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"[Imaging] Loaded weights from {model_path}")
        else:
            print("[Imaging] No weights path — using pretrained backbone only (testing)")

        self.model.to(self.device)
        self.model.eval()

        if self.use_fp16:
            self.model = self.model.half()
            print("[Imaging] Running in FP16 mode")

    @torch.no_grad()
    def infer(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Run inference on a batch of chest X-rays.

        Args:
            input_tensor: (batch_size, 3, 224, 224) — preprocessed images
                          Apply IMAGING_TRANSFORM before passing here.

        Returns:
            probs:             (batch_size, 14) — probabilities for each pathology
            inference_time_ms: wall-clock time for this inference call

        Example:
            engine = ImagingInferenceEngine()
            x = torch.randn(8, 3, 224, 224)    # batch of 8 X-rays
            probs, latency = engine.infer(x)
            # probs[i, j] = probability of pathology j in patient i
            # e.g. probs[0, 6] = probability of Pneumonia in first patient
        """
        t_start = time.perf_counter()

        x = input_tensor.to(self.device)
        if self.use_fp16:
            x = x.half()

        logits = self.model(x)               # (batch, 14) raw logits
        probs  = torch.sigmoid(logits).float()  # (batch, 14) probabilities

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        inference_time_ms = (time.perf_counter() - t_start) * 1000

        return probs.cpu(), inference_time_ms

    def predict_labels(
        self, input_tensor: torch.Tensor, threshold: float = 0.5
    ) -> Tuple[List[List[str]], float]:
        """
        Convenience method: returns predicted pathology names above threshold.

        Example:
            labels, latency = engine.predict_labels(x)
            # labels[0] = ["Atelectasis", "Effusion"] for first patient
        """
        probs, latency = self.infer(input_tensor)
        batch_labels   = []
        for sample_probs in probs:
            detected = [
                CHESTXRAY_LABELS[i]
                for i, p in enumerate(sample_probs)
                if p.item() > threshold
            ]
            batch_labels.append(detected)
        return batch_labels, latency


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def train_imaging_model(
    train_loader,
    val_loader,
    epochs:    int = 10,
    lr:        float = 1e-4,     # Lower LR — we're fine-tuning, not training from scratch
    save_path: str = "results/imaging_model.pt",
    device:    str = "cuda" if torch.cuda.is_available() else "cpu",
) -> ImagingModel:
    """
    Fine-tune ImagingModel on chest X-ray data.

    DataLoader should yield:
        x: (batch, 3, 224, 224) — preprocessed with IMAGING_TRANSFORM
        y: (batch, 14)          — multilabel binary targets (0 or 1)

    Class imbalance note: most labels are rare (Hernia ~0.2%).
    Use pos_weight or weighted sampling — or just accept it for a research baseline.
    Target AUC: 0.74–0.78 mean across 14 labels.
    """
    model     = ImagingModel(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()  # handles multilabel well
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y.float()).item()

        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"train={train_loss/len(train_loader):.4f} | "
              f"val={val_loss/len(val_loader):.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved to {save_path}")

    return model


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Imaging Model Smoke Test ===\n")

    engine = ImagingInferenceEngine(model_path=None)

    # Test with batch_size=1 (single image, as HIGH priority urgent dispatch would send)
    single = torch.randn(1, 3, 224, 224)
    probs, latency = engine.infer(single)
    print(f"Single image | Output: {probs.shape} | Latency: {latency:.2f}ms")
    print(f"SLA (500ms): {'PASS' if latency < 500 else 'FAIL'}")
    print()

    # Test with batch_size=16 (normal batch dispatch)
    batch = torch.randn(16, 3, 224, 224)
    probs, latency = engine.infer(batch)
    print(f"Batch of 16  | Output: {probs.shape} | Latency: {latency:.2f}ms")
    print(f"SLA (500ms): {'PASS' if latency < 500 else 'FAIL'}")
    print()

    # Show label predictions for one sample
    labels, _ = engine.predict_labels(single, threshold=0.3)
    print(f"Detected pathologies (threshold=0.3): {labels[0] or ['None detected']}")

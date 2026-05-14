"""
models/icu_model.py — BiLSTM for ICU sepsis prediction.

Now supports variable-length ICU sequences with proper padding-based batching.
"""

import torch
import torch.nn as nn
from typing import Any, List
from models.base import BaseInferenceEngine, ModelMetadata
from system.request import ModelType


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

class _BiLSTM(nn.Module):
    def __init__(self, input_size=34, hidden_size=64, num_layers=2,
                 dropout=0.3, bidirectional=True):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        directions = 2 if bidirectional else 1

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * directions, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.classifier(out)   # logits (NO sigmoid)


# ---------------------------------------------------------------------
# ICU Engine
# ---------------------------------------------------------------------

class ICUInferenceEngine(BaseInferenceEngine):

    INPUT_SIZE = 34
    HIDDEN_SIZE = 64

    # -------------------------------------------------------------
    # MODEL LOADING
    # -------------------------------------------------------------
    def _load_model(self):
        self.model = _BiLSTM(
            input_size=self.INPUT_SIZE,
            hidden_size=self.HIDDEN_SIZE
        )

        if self.model_path:
            try:
                state = torch.load(self.model_path, map_location=self.device)

                if isinstance(state, dict):
                    if "state_dict" in state:
                        state = state["state_dict"]
                    elif "model" in state:
                        state = state["model"]

                self.model.load_state_dict(state, strict=False)
                print(f"[ICU] Loaded weights from {self.model_path}")

            except Exception as e:
                print(f"[ICU] Failed to load weights: {e}")
                print("[ICU] Using random initialization")

        else:
            print("[ICU] Random initialization (benchmark mode)")

        self.model.to(self.device)

    # -------------------------------------------------------------
    # FORWARD
    # -------------------------------------------------------------
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # -------------------------------------------------------------
    # PREPROCESS
    # -------------------------------------------------------------
    def preprocess(self, raw_input: Any) -> torch.Tensor:
        import numpy as np

        if isinstance(raw_input, np.ndarray):
            return torch.from_numpy(raw_input).float()

        return torch.tensor(raw_input, dtype=torch.float32)

    # -------------------------------------------------------------
    # CRITICAL FIX: VARIABLE-LENGTH BATCHING
    # -------------------------------------------------------------
    def _prepare_batch(self, inputs: List[torch.Tensor]):

        from torch.nn.utils.rnn import pad_sequence

        tensors = []

        for x in inputs:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            else:
                x = x.float()

            tensors.append(x)

        # Pad ICU sequences → (B, T_max, F)
        batch = pad_sequence(
            tensors,
            batch_first=True,
            padding_value=0.0
        )

        return batch.to(self.device)

    # -------------------------------------------------------------
    # METADATA
    # -------------------------------------------------------------
    @property
    def metadata(self) -> ModelMetadata:
        has_weights = self.model_path is not None
        name = f"BiLSTM-{'pretrained' if has_weights else 'randominit'}-v1"

        return ModelMetadata(
            model_type=ModelType.ICU,
            model_name=name,
            input_shape=None,   # IMPORTANT: variable-length ICU
        )


# ---------------------------------------------------------------------
# TRAINING (fixed logits/sigmoid inconsistency)
# ---------------------------------------------------------------------

def train_icu_model(train_loader, val_loader, epochs=10, lr=1e-3,
                    save_path="results/weights/icu_model.pt", device="cuda"):

    import os

    model = _BiLSTM().to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    best = float("inf")

    for epoch in range(epochs):

        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            logits = model(x).squeeze(-1)  # FIXED
            loss = criterion(logits, y)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x).squeeze(-1)
                val_loss += criterion(logits, y).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        sched.step(val_loss)

        print(f"Epoch {epoch+1:02d}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best:
            best = val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print("  -> saved best model")

    return model

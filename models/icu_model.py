"""
models/icu_model.py — LSTM for ICU time-series prediction.

Task:    Binary classification — will this patient develop sepsis
         in the next 6 hours? (Label from MIMIC-III Sepsis Challenge)

Dataset: PhysioNet Sepsis Challenge 2019 (pre-processed CSVs)
         or MIMIC-III CHARTEVENTS (raw vitals, more preprocessing needed)

Input:   (batch_size, seq_len=48, n_features=34) — 48 hours of hourly vitals
         Features: HR, MAP, O2Sat, Temp, SBP, DBP, Resp, EtCO2,
                   BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN,
                   Alkalinephos, Calcium, Chloride, Creatinine, Glucose,
                   Lactate, Magnesium, Phosphate, Potassium, Bilirubin,
                   Hct, Hgb, PTT, WBC, Fibrinogen, Platelets,
                   Age, Gender (2 non-time-varying features)

Output:  (batch_size, 1) — probability of sepsis onset [0, 1]

SLA:     100ms  (HIGH priority — critical alert workload)
Target:  ~0.75–0.80 AUROC on validation set (acceptable for a research baseline)

Week 1 goal: get this running end-to-end on dummy data, then
             swap in real MIMIC data once PhysioNet access arrives.
"""

import time
import torch
import torch.nn as nn
from typing import Tuple


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class ICUModel(nn.Module):
    """
    Bidirectional LSTM for ICU sepsis prediction.

    Architecture choices are deliberately simple:
    - Bidirectional LSTM: captures patterns in both temporal directions
    - Single hidden layer: keeps inference fast (critical for 100ms SLA)
    - Dropout: regularization for small clinical datasets
    - Final sigmoid: output is a probability (not logits)

    This is not state-of-the-art. That's intentional. Your contribution
    is the serving system, not the model. Don't waste time here.
    """

    def __init__(
        self,
        input_size:   int = 34,    # number of clinical features
        hidden_size:  int = 64,    # LSTM hidden dimension
        num_layers:   int = 2,     # stacked LSTM layers
        dropout:      float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        self.directions    = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,       # input shape: (batch, seq, features)
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * self.directions, 1)
        self.sigmoid    = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            out: (batch_size, 1) — sepsis probability
        """
        # lstm_out: (batch, seq_len, hidden * directions)
        lstm_out, _ = self.lstm(x)

        # Use the last time step's output for classification
        last_step = lstm_out[:, -1, :]          # (batch, hidden * directions)
        dropped   = self.dropout(last_step)
        logit     = self.classifier(dropped)    # (batch, 1)
        return self.sigmoid(logit)


# ---------------------------------------------------------------------------
# Inference function — this is what the serving system calls
# ---------------------------------------------------------------------------

class ICUInferenceEngine:
    """
    Wraps ICUModel for production inference.
    Handles device placement, timing, and FP16 (optional).
    """

    def __init__(
        self,
        model_path: str = None,
        device:     str = "cuda" if torch.cuda.is_available() else "cpu",
        use_fp16:   bool = True,
    ):
        self.device  = torch.device(device)
        self.use_fp16 = use_fp16 and (device == "cuda")

        # Build model
        self.model = ICUModel()

        # Load weights if provided, else use random init (for testing)
        if model_path:
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"[ICU] Loaded weights from {model_path}")
        else:
            print("[ICU] No weights path given — using random init (testing only)")

        self.model.to(self.device)
        self.model.eval()

        # Convert to FP16 for faster inference if on GPU
        if self.use_fp16:
            self.model = self.model.half()
            print("[ICU] Running in FP16 mode")

    @torch.no_grad()
    def infer(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Run inference on a single batch.

        Args:
            input_tensor: (batch_size, seq_len=48, n_features=34)
                          Expected dtype: float32 (we handle FP16 casting internally)

        Returns:
            result:           (batch_size, 1) — sepsis probabilities [0, 1]
            inference_time_ms: wall-clock time for this inference call

        Example:
            engine = ICUInferenceEngine()
            x = torch.randn(1, 48, 34)          # 1 patient, 48 time steps
            probs, latency = engine.infer(x)
            print(f"Sepsis probability: {probs[0].item():.3f} in {latency:.1f}ms")
        """
        t_start = time.perf_counter()

        # Move to device
        x = input_tensor.to(self.device)

        # Cast to FP16 if enabled
        if self.use_fp16:
            x = x.half()

        # Forward pass
        result = self.model(x)

        # Cast result back to FP32 for downstream processing
        result = result.float()

        # Sync GPU before timing (important for accurate GPU latency measurement)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        inference_time_ms = (t_end - t_start) * 1000

        return result.cpu(), inference_time_ms


# ---------------------------------------------------------------------------
# Training helper (week 1: just get it running, don't over-tune)
# ---------------------------------------------------------------------------

def train_icu_model(
    train_loader,
    val_loader,
    epochs:    int = 10,
    lr:        float = 1e-3,
    save_path: str = "results/icu_model.pt",
    device:    str = "cuda" if torch.cuda.is_available() else "cpu",
) -> ICUModel:
    """
    Minimal training loop. Returns trained model.

    For MIMIC-III Sepsis Challenge data:
        - Each sample: (seq_len=48, n_features=34) vitals + label (0/1)
        - Class imbalance ~10:1 (non-sepsis:sepsis) — use pos_weight
        - Target AUROC: 0.75–0.80 in 10 epochs

    Args:
        train_loader: DataLoader yielding (x, y) where
                      x: (batch, 48, 34) float32
                      y: (batch, 1)      float32
        val_loader:   DataLoader for validation
        epochs:       Training epochs (10 is enough for a baseline)
        lr:           Learning rate
        save_path:    Where to save the trained weights
        device:       "cuda" or "cpu"
    """
    model = ICUModel().to(device)

    # Weighted BCE for class imbalance (sepsis is rare — ~10:1 ratio)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # Use raw logits for BCEWithLogitsLoss (numerically stable)
            logits = model.classifier(model.dropout(model.lstm(x)[0][:, -1, :]))
            loss   = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits   = model.classifier(model.dropout(model.lstm(x)[0][:, -1, :]))
                val_loss += criterion(logits, y).item()

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"train_loss={train_loss/len(train_loader):.4f} | "
              f"val_loss={val_loss/len(val_loader):.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model to {save_path}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    return model


# ---------------------------------------------------------------------------
# Quick smoke test — run this file directly to verify everything works
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== ICU Model Smoke Test ===\n")

    engine = ICUInferenceEngine(model_path=None)  # random weights

    # Simulate a batch of 4 ICU patients, 48 hourly time steps, 34 features
    dummy_input = torch.randn(4, 48, 34)
    results, latency = engine.infer(dummy_input)

    print(f"Input shape:   {dummy_input.shape}")
    print(f"Output shape:  {results.shape}")
    print(f"Predictions:   {results.squeeze().tolist()}")
    print(f"Inference time: {latency:.2f}ms")
    print(f"SLA (100ms):   {'PASS' if latency < 100 else 'FAIL'}")
    print()

    # Verify single-patient inference (batch_size=1, as scheduler would call it)
    single = torch.randn(1, 48, 34)
    result, latency = engine.infer(single)
    print(f"Single patient sepsis probability: {result.item():.4f}")
    print(f"Latency: {latency:.2f}ms")

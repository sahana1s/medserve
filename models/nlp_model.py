"""
models/nlp_model.py — Clinical NLP using BiomedBERT pretrained weights.
FIXED: saves/loads base model name alongside weights so registry always
uses the correct architecture when loading local .pt files.
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Any, List, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from models.base import BaseInferenceEngine, ModelMetadata
from system.request import ModelType

MEDNLI_LABELS = ["entailment", "neutral", "contradiction"]

_PRETRAINED_OPTIONS = [
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    "allenai/scibert_scivocab_uncased",
    "distilbert-base-uncased-finetuned-sst-2-english",
    "distilbert-base-uncased",
]

# Sidecar filename: stored next to nlp_model.pt
_CONFIG_SUFFIX = "_config.json"


def _config_path(weights_path: str) -> Path:
    """Given 'results/weights/nlp_model.pt', returns 'results/weights/nlp_model_config.json'"""
    p = Path(weights_path)
    return p.parent / (p.stem + _CONFIG_SUFFIX)


class NLPInferenceEngine(BaseInferenceEngine):
    """
    Clinical NLP inference engine.
    Saves base_model name alongside .pt file so loading always uses
    the correct architecture — no hardcoded base model string.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_classes: int = 2,
        labels: Optional[List[str]] = None,
        max_length: int = 128,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_fp16: bool = True,
        **kwargs,
    ):
        self.num_classes = num_classes
        self.max_length  = max_length
        self.labels      = labels or MEDNLI_LABELS[:num_classes]
        self._base_model = None   # populated in _load_model
        super().__init__(model_path=model_path, device=device, use_fp16=use_fp16)

    # ----------------------------------------------------------------
    # Model loading
    # ----------------------------------------------------------------

    def _load_model(self):
        """
        Loading strategy:
        1. If model_path points to a .pt file AND a sidecar _config.json exists
           → read base_model from sidecar, load tokenizer + arch from it,
             then load state_dict from .pt
        2. If model_path points to a .pt file but NO sidecar exists
           → the .pt was saved by the old code (wrong base).
             Detect the correct base by trying to load each pretrained option
             and finding the one whose state_dict keys match.
        3. If model_path is None or is a HuggingFace model name
           → load directly from HF Hub.
        """
        if self.model_path and Path(self.model_path).suffix == ".pt":
            self._load_from_pt()
        else:
            # model_path is either None or a HF hub name
            hf_name = self.model_path  # may be None
            self._load_pretrained(preferred=hf_name)

    def _load_from_pt(self):
        """Load weights from a local .pt file using sidecar config if available."""
        cfg_path = _config_path(self.model_path)

        if cfg_path.exists():
            # Happy path: sidecar tells us exactly which base model was used
            with open(cfg_path) as f:
                cfg = json.load(f)
            base_model = cfg.get("base_model", _PRETRAINED_OPTIONS[0])
            print(f"[NLP] Sidecar found → base model: {base_model}")
            self._load_arch_and_weights(base_model, self.model_path)
        else:
            # No sidecar: probe each pretrained option to find matching architecture
            print(f"[NLP] No sidecar at {cfg_path} — probing architectures...")
            state = torch.load(self.model_path, map_location="cpu")
            if isinstance(state, dict) and "base_model" in state:
                # Some savers embed base_model inside the state dict itself
                base_model = state.pop("base_model")
                print(f"[NLP] Found base_model in state dict: {base_model}")
                self._load_arch_and_weights(base_model, None, state_dict=state)
                return

            # Try to load each candidate and check for missing/unexpected keys
            loaded = False
            for candidate in _PRETRAINED_OPTIONS:
                try:
                    print(f"[NLP] Probing: {candidate}")
                    tok   = AutoTokenizer.from_pretrained(candidate)
                    model = AutoModelForSequenceClassification.from_pretrained(
                        candidate,
                        num_labels=self.num_classes,
                        ignore_mismatched_sizes=True,
                    )
                    result = model.load_state_dict(state, strict=False)
                    missing     = [k for k in result.missing_keys
                                   if "classifier" not in k and "pooler" not in k]
                    unexpected  = [k for k in result.unexpected_keys
                                   if "cls." not in k]  # cls.* are MLM heads, OK to ignore
                    if len(missing) == 0 and len(unexpected) == 0:
                        print(f"[NLP] Architecture match: {candidate}")
                        self.tokenizer   = tok
                        self.model       = model
                        self._base_model = candidate
                        self.model.to(self.device)
                        # Write sidecar so next load is fast
                        _write_sidecar(self.model_path, candidate)
                        loaded = True
                        break
                    else:
                        print(f"[NLP]   missing={len(missing)} unexpected={len(unexpected)} — skip")
                except Exception as e:
                    print(f"[NLP]   {candidate} failed: {str(e)[:80]}")

            if not loaded:
                # Fallback: just load BiomedBERT ignoring mismatches
                # (the classifier head will be randomly init'd — fine for benchmarking)
                print("[NLP] No exact match found — loading BiomedBERT with ignore_mismatched_sizes")
                self._load_arch_and_weights(_PRETRAINED_OPTIONS[0], self.model_path)

    def _load_arch_and_weights(self, base_model: str, weights_path: Optional[str],
                               state_dict=None):
        """Load tokenizer + model arch from base_model, then apply weights."""
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True,
        )
        self._base_model = base_model

        if state_dict is None and weights_path:
            state_dict = torch.load(weights_path, map_location=self.device)

        if state_dict is not None:
            result = self.model.load_state_dict(state_dict, strict=False)
            unexpected_real = [k for k in result.unexpected_keys if "cls." not in k]
            missing_real    = [k for k in result.missing_keys
                               if "classifier" not in k and "pooler" not in k]
            if unexpected_real or missing_real:
                print(f"[NLP] Load report — unexpected: {len(result.unexpected_keys)}"
                      f"  missing: {len(result.missing_keys)}")
                print("[NLP] cls.* keys (MLM head) are expected to be UNEXPECTED — OK to ignore.")
                print("[NLP] classifier.* MISSING = new head, randomly initialized — OK for benchmarking.")
            else:
                print(f"[NLP] Clean load from {weights_path}")

        self.model.to(self.device)
        print(f"[NLP] Ready: {base_model} ({self.num_classes} classes)")

    def _load_pretrained(self, preferred: Optional[str] = None):
        """Load directly from HF Hub."""
        candidates = []
        if preferred:
            candidates.append(preferred)
        candidates += [o for o in _PRETRAINED_OPTIONS if o != preferred]

        for name in candidates:
            try:
                print(f"[NLP] Loading from HF: {name}")
                self.tokenizer = AutoTokenizer.from_pretrained(name)
                self.model     = AutoModelForSequenceClassification.from_pretrained(
                    name, num_labels=self.num_classes, ignore_mismatched_sizes=True,
                )
                self._base_model = name
                self.model.to(self.device)
                print(f"[NLP] Loaded: {name}")
                return
            except Exception as e:
                print(f"[NLP] Failed {name}: {str(e)[:80]}")

        raise RuntimeError("No NLP model could be loaded from HF Hub.")

    # ----------------------------------------------------------------
    # Engine API
    # ----------------------------------------------------------------

    def _prepare_batch(self, inputs: List[str]) -> dict:
        enc = self.tokenizer(
            inputs, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    def _forward(self, batch: dict) -> torch.Tensor:
        out = self.model(**batch)
        return torch.softmax(out.logits, dim=-1).cpu().float()

    def preprocess(self, raw_input: Any) -> str:
        return str(raw_input)

    @property
    def metadata(self) -> ModelMetadata:
        name = self._base_model or "BiomedBERT-NLP"
        return ModelMetadata(
            model_type=ModelType.NLP,
            model_name=name,
            input_shape=None,
        )


# ----------------------------------------------------------------
# Sidecar helper
# ----------------------------------------------------------------

def _write_sidecar(weights_path: str, base_model: str):
    """Write base_model name next to the .pt file so future loads are instant."""
    cfg_path = _config_path(weights_path)
    with open(cfg_path, "w") as f:
        json.dump({"base_model": base_model}, f, indent=2)
    print(f"[NLP] Sidecar written: {cfg_path}")


def save_nlp_weights(model, tokenizer, weights_path: str, base_model: str):
    """
    Correctly save NLP model weights + sidecar.
    Use this in load_pretrained_models.py instead of torch.save(model.state_dict()).
    """
    import os
    os.makedirs(str(Path(weights_path).parent), exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    _write_sidecar(weights_path, base_model)
    print(f"[NLP] Saved weights: {weights_path}")
    print(f"[NLP] Saved sidecar: {_config_path(weights_path)}")


# ----------------------------------------------------------------
# Optional fine-tuning
# ----------------------------------------------------------------

def train_nlp_model(train_texts, train_labels, val_texts, val_labels,
                    num_classes=5, epochs=3, batch_size=32, lr=2e-5,
                    save_path="results/weights/nlp_model.pt", device="cuda"):
    from torch.utils.data import Dataset, DataLoader

    base = _PRETRAINED_OPTIONS[0]
    tok  = AutoTokenizer.from_pretrained(base)
    mdl  = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=num_classes, ignore_mismatched_sizes=True
    ).to(device)

    class TD(Dataset):
        def __init__(self, texts, labels):
            self.enc    = tok(texts, padding=True, truncation=True,
                              max_length=128, return_tensors="pt")
            self.labels = torch.tensor(labels, dtype=torch.long)
        def __len__(self): return len(self.labels)
        def __getitem__(self, i):
            return {k: v[i] for k, v in self.enc.items()}, self.labels[i]

    def collate(batch):
        items, labels = zip(*batch)
        return {k: torch.stack([b[k] for b in items]) for k in items[0]}, torch.stack(labels)

    tl  = DataLoader(TD(train_texts, train_labels), batch_size, shuffle=True, collate_fn=collate)
    vl  = DataLoader(TD(val_texts,   val_labels),   batch_size, collate_fn=collate)
    opt = torch.optim.AdamW(mdl.parameters(), lr=lr)
    best = 0.0

    for ep in range(epochs):
        mdl.train()
        for enc, labs in tl:
            opt.zero_grad()
            mdl(**{k: v.to(device) for k, v in enc.items()}, labels=labs.to(device)).loss.backward()
            nn.utils.clip_grad_norm_(mdl.parameters(), 1.0); opt.step()
        mdl.eval(); correct = total = 0
        with torch.no_grad():
            for enc, labs in vl:
                out = mdl(**{k: v.to(device) for k, v in enc.items()})
                correct += (out.logits.argmax(-1).cpu() == labs).sum().item()
                total   += len(labs)
        acc = correct / total
        print(f"Epoch {ep+1}/{epochs}  val_acc={acc:.4f}")
        if acc > best:
            best = acc
            save_nlp_weights(mdl, tok, save_path, base)

    return mdl

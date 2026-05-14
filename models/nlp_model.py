"""
models/nlp_model.py — Clinical NLP using BiomedBERT pretrained weights.

FIXED:
- robust batching (handles str, list[str], nested lists)
- safe integration with BaseInferenceEngine (no fragile assumptions)
- consistent tokenizer input guarantees
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Any, List, Optional

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from models.base import BaseInferenceEngine, ModelMetadata
from system.request import ModelType


# ---------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------

MEDNLI_LABELS = ["entailment", "neutral", "contradiction"]

_PRETRAINED_OPTIONS = [
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    "allenai/scibert_scivocab_uncased",
    "distilbert-base-uncased-finetuned-sst-2-english",
    "distilbert-base-uncased",
]


# ---------------------------------------------------------------------
# Sidecar config
# ---------------------------------------------------------------------

_CONFIG_SUFFIX = "_config.json"


def _config_path(weights_path: str) -> Path:
    p = Path(weights_path)
    return p.parent / (p.stem + _CONFIG_SUFFIX)


# ---------------------------------------------------------------------
# NLP Engine
# ---------------------------------------------------------------------

class NLPInferenceEngine(BaseInferenceEngine):

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
        self.max_length = max_length
        self.labels = labels or MEDNLI_LABELS[:num_classes]
        self._base_model = None

        super().__init__(
            model_path=model_path,
            device=device,
            use_fp16=use_fp16,
        )

    # ---------------------------------------------------------
    # LOAD MODEL
    # ---------------------------------------------------------

    def _load_model(self):

        if self.model_path and Path(self.model_path).suffix == ".pt":
            self._load_from_pt()
        else:
            self._load_pretrained(self.model_path)

    # ---------------------------------------------------------
    # LOAD FROM .PT
    # ---------------------------------------------------------

    def _load_from_pt(self):

        cfg_path = _config_path(self.model_path)

        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)

            base_model = cfg.get("base_model", _PRETRAINED_OPTIONS[0])

            print(f"[NLP] Sidecar found → {base_model}")
            self._load_arch_and_weights(base_model, self.model_path)

        else:
            print("[NLP] No sidecar found — using fallback base model")
            self._load_arch_and_weights(_PRETRAINED_OPTIONS[0], self.model_path)

    # ---------------------------------------------------------
    # LOAD ARCH + WEIGHTS
    # ---------------------------------------------------------

    def _load_arch_and_weights(self, base_model: str, weights_path: Optional[str]):

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True,
        )

        self._base_model = base_model

        if weights_path:
            state = torch.load(weights_path, map_location="cpu")

            if isinstance(state, dict):
                if "state_dict" in state:
                    state = state["state_dict"]
                elif "model" in state:
                    state = state["model"]

            self.model.load_state_dict(state, strict=False)

        self.model.to(self.device)

        print(f"[NLP] Ready: {base_model}")

    # ---------------------------------------------------------
    # LOAD FROM HF
    # ---------------------------------------------------------

    def _load_pretrained(self, preferred: Optional[str]):

        candidates = []
        if preferred:
            candidates.append(preferred)

        candidates += [c for c in _PRETRAINED_OPTIONS if c != preferred]

        for name in candidates:
            try:
                print(f"[NLP] Loading HF model: {name}")

                self.tokenizer = AutoTokenizer.from_pretrained(name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    name,
                    num_labels=self.num_classes,
                    ignore_mismatched_sizes=True,
                )

                self._base_model = name
                self.model.to(self.device)

                print(f"[NLP] Loaded: {name}")
                return

            except Exception as e:
                print(f"[NLP] Failed {name}: {str(e)[:80]}")

        raise RuntimeError("No NLP model could be loaded.")

    # ---------------------------------------------------------
    # SAFE BATCHING (CRITICAL FIX)
    # ---------------------------------------------------------

    def _prepare_batch(self, inputs: List[Any]) -> dict:
        """
        Robust batching for ALL scheduler behaviors:
        - str
        - List[str]
        - nested lists
        - mixed types
        """

        texts = []

        for x in inputs:

            # nested batching safety
            if isinstance(x, list):
                texts.extend([str(i) for i in x])
            else:
                texts.append(str(x))

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {k: v.to(self.device) for k, v in enc.items()}

    # ---------------------------------------------------------
    # FORWARD
    # ---------------------------------------------------------

    def _forward(self, batch: dict) -> torch.Tensor:
        logits = self.model(**batch).logits
        return torch.softmax(logits, dim=-1).cpu().float()

    # ---------------------------------------------------------
    # PREPROCESS (SINGLE SAMPLE ONLY)
    # ---------------------------------------------------------

    def preprocess(self, raw_input: Any) -> str:
        return str(raw_input)

    # ---------------------------------------------------------
    # METADATA
    # ---------------------------------------------------------

    @property
    def metadata(self) -> ModelMetadata:
        name = self._base_model or "BiomedBERT-NLP"

        return ModelMetadata(
            model_type=ModelType.NLP,
            model_name=name,
            input_shape=None,
        )


# ---------------------------------------------------------------------
# SIDE CAR SAVE
# ---------------------------------------------------------------------

def _write_sidecar(weights_path: str, base_model: str):
    cfg_path = _config_path(weights_path)

    with open(cfg_path, "w") as f:
        json.dump({"base_model": base_model}, f, indent=2)

    print(f"[NLP] Sidecar written: {cfg_path}")


def save_nlp_weights(model, tokenizer, weights_path: str, base_model: str):

    import os

    os.makedirs(str(Path(weights_path).parent), exist_ok=True)

    torch.save(model.state_dict(), weights_path)
    _write_sidecar(weights_path, base_model)

    print(f"[NLP] Saved: {weights_path}")


# ---------------------------------------------------------------------
# TRAINING (unchanged logic, safe)
# ---------------------------------------------------------------------

def train_nlp_model(
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    num_classes=5,
    epochs=3,
    batch_size=32,
    lr=2e-5,
    save_path="results/weights/nlp_model.pt",
    device="cuda",
):

    from torch.utils.data import Dataset, DataLoader

    base = _PRETRAINED_OPTIONS[0]

    tok = AutoTokenizer.from_pretrained(base)

    mdl = AutoModelForSequenceClassification.from_pretrained(
        base,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    ).to(device)

    class TD(Dataset):
        def __init__(self, texts, labels):
            self.enc = tok(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            self.labels = torch.tensor(labels)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            return {k: v[i] for k, v in self.enc.items()}, self.labels[i]

    def collate(batch):
        items, labels = zip(*batch)
        return (
            {k: torch.stack([b[k] for b in items]) for k in items[0]},
            torch.stack(labels),
        )

    train_loader = DataLoader(
        TD(train_texts, train_labels),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
    )

    val_loader = DataLoader(
        TD(val_texts, val_labels),
        batch_size=batch_size,
        collate_fn=collate,
    )

    opt = torch.optim.AdamW(mdl.parameters(), lr=lr)

    best = 0.0

    for ep in range(epochs):

        mdl.train()

        for enc, labs in train_loader:
            opt.zero_grad()

            loss = mdl(
                **{k: v.to(device) for k, v in enc.items()},
                labels=labs.to(device),
            ).loss

            loss.backward()
            opt.step()

        mdl.eval()

        correct = total = 0

        with torch.no_grad():
            for enc, labs in val_loader:
                out = mdl(**{k: v.to(device) for k, v in enc.items()})
                pred = out.logits.argmax(-1).cpu()

                correct += (pred == labs).sum().item()
                total += len(labs)

        acc = correct / total

        print(f"Epoch {ep+1}/{epochs}  val_acc={acc:.4f}")

        if acc > best:
            best = acc
            save_nlp_weights(mdl, tok, save_path, base)

    return mdl

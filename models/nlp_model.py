"""
models/nlp_model.py — Clinical NLP using BiomedBERT pretrained weights.
No training required. Uses microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract
— a BERT model pretrained on 21M PubMed abstracts + 3M full-text articles.
Falls back to distilbert-base-uncased if BiomedBERT unavailable.
"""

import time
import torch
import torch.nn as nn
from typing import Any, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from models.base import BaseInferenceEngine, ModelMetadata
from system.request import ModelType

MEDNLI_LABELS = ["entailment", "neutral", "contradiction"]
PUBMED_LABELS = ["background", "objective", "methods", "results", "conclusions"]

# Best pretrained options in priority order
_PRETRAINED_OPTIONS = [
    # BiomedBERT: pretrained on PubMed, strong clinical text understanding
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    # SciBERT: pretrained on scientific papers
    "allenai/scibert_scivocab_uncased",
    # General fallback: SST-2 fine-tuned DistilBERT (binary, just for smoke testing)
    "distilbert-base-uncased-finetuned-sst-2-english",
]


class NLPInferenceEngine(BaseInferenceEngine):
    """
    BiomedBERT-based clinical text classifier.
    Input: string or list of strings   Output: (batch, num_classes)   SLA: 300ms

    For benchmarking purposes, the model runs sequence classification.
    The exact task/labels don't matter for scheduler benchmarking —
    what matters is the compute profile (transformer forward pass on variable-length text).
    """

    def __init__(self, model_path=None, num_classes=2, labels=None,
                 max_length=128, device="cuda" if torch.cuda.is_available() else "cpu",
                 use_fp16=True):
        self.num_classes = num_classes
        self.max_length  = max_length
        self.labels      = labels or MEDNLI_LABELS[:num_classes]
        self._model_name = None
        # Tokenizer loaded in _load_model so device is available
        super().__init__(model_path=model_path, device=device, use_fp16=use_fp16)

    def _load_model(self):
        # If a local fine-tuned path is given, use it directly
        if self.model_path and not any(self.model_path.startswith(opt) for opt in _PRETRAINED_OPTIONS):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    "distilbert-base-uncased", num_labels=self.num_classes
                )
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self._model_name = f"fine-tuned:{self.model_path}"
                print(f"[NLP] Loaded fine-tuned weights from {self.model_path}")
                self.model.to(self.device)
                return
            except Exception as e:
                print(f"[NLP] Could not load {self.model_path}: {e} — trying pretrained")

        # Try pretrained options in order
        hf_name = self.model_path or None
        loaded  = False

        candidates = [hf_name] + _PRETRAINED_OPTIONS if hf_name else _PRETRAINED_OPTIONS

        for candidate in candidates:
            if candidate is None:
                continue
            try:
                print(f"[NLP] Trying: {candidate}")
                self.tokenizer = AutoTokenizer.from_pretrained(candidate)
                self.model     = AutoModelForSequenceClassification.from_pretrained(
                    candidate,
                    num_labels=self.num_classes,
                    ignore_mismatched_sizes=True,  # allows loading with different num_labels
                )
                self._model_name = candidate
                loaded = True
                print(f"[NLP] Loaded: {candidate}")
                break
            except Exception as e:
                print(f"[NLP]   Failed ({type(e).__name__}: {str(e)[:80]})")
                continue

        if not loaded:
            raise RuntimeError(
                "Could not load any NLP model. Check internet connectivity on Kaggle/Colab."
            )

        self.model.to(self.device)

    def _prepare_batch(self, inputs: List[str]) -> dict:
        """Tokenize full batch — captures padding overhead in latency measurement."""
        encoded = self.tokenizer(
            inputs, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in encoded.items()}

    def _forward(self, batch: dict) -> torch.Tensor:
        out = self.model(**batch)
        return torch.softmax(out.logits, dim=-1).cpu().float()

    def preprocess(self, raw_input: Any) -> str:
        return str(raw_input)

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_type=ModelType.NLP,
            model_name=self._model_name or "BiomedBERT-pretrained",
            input_shape=None,
        )


def train_nlp_model(train_texts, train_labels, val_texts, val_labels,
                    num_classes=5, epochs=3, batch_size=32, lr=2e-5,
                    save_path="results/weights/nlp_model.pt", device="cuda"):
    """Optional fine-tuning on top of pretrained BiomedBERT."""
    import os
    from torch.utils.data import Dataset, DataLoader

    base_model = _PRETRAINED_OPTIONS[0]  # start from BiomedBERT
    tokenizer  = AutoTokenizer.from_pretrained(base_model)
    model      = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=num_classes, ignore_mismatched_sizes=True
    ).to(device)

    class TD(Dataset):
        def __init__(self, texts, labels):
            self.enc    = tokenizer(texts, padding=True, truncation=True,
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
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best = 0.0

    for ep in range(epochs):
        model.train()
        for enc, labs in tl:
            opt.zero_grad()
            loss = model(**{k:v.to(device) for k,v in enc.items()}, labels=labs.to(device)).loss
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        model.eval(); correct=total=0
        with torch.no_grad():
            for enc, labs in vl:
                out = model(**{k:v.to(device) for k,v in enc.items()})
                correct += (out.logits.argmax(-1).cpu()==labs).sum().item(); total+=len(labs)
        acc = correct/total; print(f"Epoch {ep+1}/{epochs}  val_acc={acc:.4f}")
        if acc>best:
            best=acc; os.makedirs(str(save_path).rsplit("/",1)[0], exist_ok=True)
            torch.save(model.state_dict(), save_path); print("  -> saved")
    return model

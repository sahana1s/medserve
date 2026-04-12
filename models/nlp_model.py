"""
models/nlp_model.py — Clinical NLP using BiomedBERT pretrained weights.
FIXED: registry compatibility + safer HF loading on Kaggle
"""

import torch
import torch.nn as nn
from typing import Any, List, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from models.base import BaseInferenceEngine, ModelMetadata
from system.request import ModelType


MEDNLI_LABELS = ["entailment", "neutral", "contradiction"]

_PRETRAINED_OPTIONS = [
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    "allenai/scibert_scivocab_uncased",
    "distilbert-base-uncased-finetuned-sst-2-english",
]


class NLPInferenceEngine(BaseInferenceEngine):
    """
    Clinical NLP inference engine (BiomedBERT / SciBERT / DistilBERT fallback)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_classes: int = 2,
        labels: Optional[List[str]] = None,
        max_length: int = 128,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_fp16: bool = True,
        **kwargs,   # ✅ FIX: absorbs registry mismatches safely
    ):
        self.num_classes = num_classes
        self.max_length = max_length
        self.labels = labels or MEDNLI_LABELS[:num_classes]

        super().__init__(
            model_path=model_path,
            device=device,
            use_fp16=use_fp16
        )

    # ---------------------------------------------------------
    # Model loading
    # ---------------------------------------------------------

    def _load_model(self):

        # If local fine-tuned weights exist
        if self.model_path and not self.model_path.startswith("http"):
            try:
                base = "distilbert-base-uncased"

                self.tokenizer = AutoTokenizer.from_pretrained(base)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    base,
                    num_labels=self.num_classes,
                    ignore_mismatched_sizes=True,
                )

                state = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state)

                print(f"[NLP] Loaded local weights: {self.model_path}")

            except Exception as e:
                print(f"[NLP] Local load failed: {e}")
                self._load_pretrained()
            return

        self._load_pretrained()

    # ---------------------------------------------------------
    # Pretrained fallback
    # ---------------------------------------------------------

    def _load_pretrained(self):
        for name in _PRETRAINED_OPTIONS:
            try:
                print(f"[NLP] Trying: {name}")

                self.tokenizer = AutoTokenizer.from_pretrained(name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    name,
                    num_labels=self.num_classes,
                    ignore_mismatched_sizes=True,
                )

                print(f"[NLP] Loaded: {name}")
                break

            except Exception as e:
                print(f"[NLP] Failed {name}: {str(e)[:80]}")
                continue
        else:
            raise RuntimeError("No NLP model could be loaded")

        self.model.to(self.device)

    # ---------------------------------------------------------
    # Engine API
    # ---------------------------------------------------------

    def _prepare_batch(self, inputs: List[str]):
        enc = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    def _forward(self, batch):
        out = self.model(**batch)
        return torch.softmax(out.logits, dim=-1).cpu().float()

    def preprocess(self, raw_input: Any) -> str:
        return str(raw_input)

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_type=ModelType.NLP,
            model_name="BiomedBERT-NLP",
            input_shape=None,
        )

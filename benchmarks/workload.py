"""
benchmarks/workload.py — Reproducible hospital workload generator.
FIXED: ICU input shape is now deterministic to match model contract.
"""

import random
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


class LoadLevel(Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


LOAD_RATES = {
    LoadLevel.LOW:    10.0,
    LoadLevel.MEDIUM: 30.0,
    LoadLevel.HIGH:   60.0,
}

DEFAULT_MIX  = {"icu": 0.20, "nlp": 0.30, "imaging": 0.50}

SLA_MS = {
    "icu":     250,
    "nlp":     500,
    "imaging": 1200,
}


@dataclass
class WorkloadRequest:
    request_id: str
    model_type: str
    input_data: object
    arrival_offset_ms: float
    sla_ms: float
    sent_at_ms: Optional[float] = None
    result_at_ms: Optional[float] = None
    inference_ms: Optional[float] = None


class WorkloadGenerator:

    CLINICAL_TEXTS = [
        "Patient is hemodynamically stable.",
        "Chest X-ray shows bilateral infiltrates.",
        "Labs: WBC elevated, sepsis suspected.",
        "Vitals stable. Continue monitoring.",
    ]

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def generate(self, n: int, arrival_rate: float,
                 mix: Dict[str, float] = None) -> List[WorkloadRequest]:

        mix   = mix or DEFAULT_MIX
        types = list(mix.keys())
        wts   = list(mix.values())

        gaps = self.np_rng.exponential(1.0 / arrival_rate, size=n)
        offsets = np.cumsum(gaps) * 1000.0

        return [
            WorkloadRequest(
                request_id=f"req_{i:05d}",
                model_type=(mt := self.rng.choices(types, weights=wts)[0]),
                input_data=self._make_input(mt),
                arrival_offset_ms=float(offsets[i]),
                sla_ms=SLA_MS[mt],
            )
            for i in range(n)
        ]

    def _make_input(self, model_type: str):

        # ---------------------------------------------------------
        # ICU: FIXED SHAPE (CRITICAL FIX)
        # ---------------------------------------------------------
        if model_type == "icu":
            seq_len = 48   # MUST MATCH MODEL CONTRACT
            return np.clip(
                self.np_rng.normal(0.5, 0.15, (seq_len, 34)),
                0.0,
                1.0
            ).astype(np.float32)

        # ---------------------------------------------------------
        # Imaging: fixed shape
        # ---------------------------------------------------------
        elif model_type == "imaging":
            return self.np_rng.normal(
                loc=[[[0.485]], [[0.456]], [[0.406]]],
                scale=[[[0.229]], [[0.224]], [[0.225]]],
                size=(3, 224, 224)
            ).astype(np.float32)

        # ---------------------------------------------------------
        # NLP: unchanged
        # ---------------------------------------------------------
        elif model_type == "nlp":
            text = self.rng.choice(self.CLINICAL_TEXTS)
            return text

        raise ValueError(f"Unknown model_type: {model_type}")

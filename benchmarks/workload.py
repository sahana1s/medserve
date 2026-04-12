"""
benchmarks/workload.py — Reproducible hospital workload generator.

Produces an identical request stream used across ALL three benchmark modes.
Same seed = same requests = fair comparison between schedulers.

Workload model: Poisson arrivals, three model types, configurable mix.
"""

import random, time
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


class LoadLevel(Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"

LOAD_RATES   = {LoadLevel.LOW: 5.0, LoadLevel.MEDIUM: 20.0, LoadLevel.HIGH: 50.0}
DEFAULT_MIX  = {"icu": 0.20, "nlp": 0.30, "imaging": 0.50}
ICU_HEAVY    = {"icu": 0.50, "nlp": 0.25, "imaging": 0.25}
BALANCED_MIX = {"icu": 0.33, "nlp": 0.34, "imaging": 0.33}
SLA_MS       = {"icu": 100,  "nlp": 300,  "imaging": 500}


@dataclass
class WorkloadRequest:
    request_id:        str
    model_type:        str
    input_data:        object        # np.ndarray for icu/imaging, str for nlp
    arrival_offset_ms: float
    sla_ms:            float
    sent_at_ms:        Optional[float] = None
    result_at_ms:      Optional[float] = None
    inference_ms:      Optional[float] = None

    @property
    def total_latency_ms(self):
        if self.sent_at_ms and self.result_at_ms:
            return self.result_at_ms - self.sent_at_ms
        return None

    @property
    def sla_violated(self):
        if self.total_latency_ms is None: return None
        return self.total_latency_ms > self.sla_ms


class WorkloadGenerator:
    """Reproducible synthetic workloads. Same seed = same requests every time."""

    CLINICAL_TEXTS = [
        "Patient is hemodynamically stable. No acute distress noted.",
        "Chest X-ray shows bilateral infiltrates consistent with pneumonia.",
        "Labs: WBC 14.2, lactate 3.1, procalcitonin elevated. Sepsis protocol initiated.",
        "No fever. Vitals within normal limits. Continue current management.",
        "ECG shows ST elevation in V1-V4. Cardiology consult requested.",
        "Patient reports worsening shortness of breath over past 24 hours.",
        "Post-operative day 2. Patient tolerating oral intake. Wound healing well.",
        "Acute onset confusion in elderly patient. Urine culture pending.",
        "BP 80/50, HR 120, RR 28. Starting vasopressors. ICU transfer.",
        "Routine follow-up. All labs within normal limits. Discharge home.",
    ]

    def __init__(self, seed: int = 42):
        self.seed   = seed
        self.rng    = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def generate(self, n: int, arrival_rate: float,
                 mix: Dict[str, float] = None) -> List[WorkloadRequest]:
        """Generate n Poisson-arrival requests with pre-built inputs."""
        mix    = mix or DEFAULT_MIX
        types  = list(mix.keys())
        wts    = list(mix.values())
        gaps   = self.np_rng.exponential(1.0 / arrival_rate, size=n)
        offsets = np.cumsum(gaps) * 1000.0

        return [
            WorkloadRequest(
                request_id        = f"req_{i:05d}",
                model_type        = (mt := self.rng.choices(types, weights=wts)[0]),
                input_data        = self._make_input(mt),
                arrival_offset_ms = float(offsets[i]),
                sla_ms            = SLA_MS[mt],
            )
            for i in range(n)
        ]

    def generate_levels(self, n_per_level: int = 500,
                        mix: Dict = None) -> Dict[str, List[WorkloadRequest]]:
        return {lv.value: self.generate(n_per_level, LOAD_RATES[lv], mix)
                for lv in LoadLevel}

    def _make_input(self, model_type: str):
        if model_type == "icu":
            return np.clip(
                self.np_rng.normal(0.5, 0.15, (48, 34)), 0.0, 1.0
            ).astype(np.float32)
        elif model_type == "imaging":
            return self.np_rng.normal(
                loc=[[[0.485]], [[0.456]], [[0.406]]],
                scale=[[[0.229]], [[0.224]], [[0.225]]],
                size=(3, 224, 224)
            ).astype(np.float32)
        elif model_type == "nlp":
            return self.rng.choice(self.CLINICAL_TEXTS)
        raise ValueError(f"Unknown model_type: {model_type}")

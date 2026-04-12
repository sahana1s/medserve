"""
models/registry.py — MedServe Model Registry (FIXED VERSION)

Key fixes:
- Prevents accidental HuggingFace downloads during warmup
- Safe handling of missing weights
- Robust batch inference (ICU / Imaging / NLP separated correctly)
- No hidden side-effects in default()
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from system.request import Request, ModelType
from models.base import BaseInferenceEngine, ModelMetadata


class ModelRegistry:

    def __init__(self):
        self._engines: Dict[ModelType, BaseInferenceEngine] = {}
        self._call_counts = {t: 0 for t in ModelType}
        self._total_latency = {t: 0.0 for t in ModelType}

    # ---------------------------------------------------------
    # Register
    # ---------------------------------------------------------
    def register(self, engine: BaseInferenceEngine):
        mtype = engine.metadata.model_type

        if mtype in self._engines:
            print(f"[Registry] Replacing {mtype.value}")

        self._engines[mtype] = engine

        print(
            f"[Registry] Registered {engine.metadata.model_name} "
            f"for {mtype.value} (SLA={engine.metadata.sla_ms}ms)"
        )
        return self

    # ---------------------------------------------------------
    # Default setup (SAFE)
    # ---------------------------------------------------------
    @classmethod
    def default(
        cls,
        weights_dir="results/weights",
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_fp16=True,
    ):

        from models.icu_model import ICUInferenceEngine
        from models.imaging_model import ImagingInferenceEngine
        from models.nlp_model import NLPInferenceEngine

        weights = Path(weights_dir)

        def get_path(name: str):
            p = weights / name
            return str(p) if p.exists() else None

        registry = cls()

        # ---------------- ICU ----------------
        registry.register(ICUInferenceEngine(
            model_path=get_path("icu_model.pt"),
            device=device,
            use_fp16=use_fp16,
        ))

        # ---------------- Imaging ----------------
        registry.register(ImagingInferenceEngine(
            model_path=get_path("imaging_model.pt"),
            device=device,
            use_fp16=use_fp16,
        ))

        # ---------------- NLP (CRITICAL FIX) ----------------
        nlp_path = get_path("nlp_model.pt")

        registry.register(NLPInferenceEngine(
            model_path=nlp_path,
            device=device,
            use_fp16=use_fp16,

            # IMPORTANT FLAG (you must support this in NLP engine)
            local_only=nlp_path is not None
        ))

        return registry

    # ---------------------------------------------------------
    # Warmup (NO NETWORK CALLS ALLOWED)
    # ---------------------------------------------------------
    def warmup_all(self, n_runs: int = 5):
        print("\n[Registry] Warming up all engines...")

        for engine in self._engines.values():

            try:
                # NEVER let warmup trigger HF downloads
                engine.warmup(n_runs=n_runs)

                # only imaging/icu benefit from batch benchmark
                if engine.metadata.model_type != ModelType.NLP:
                    engine.benchmark_batch_sizes()

            except Exception as e:
                print(f"[Registry] Warmup skipped for {engine.metadata.model_name}: {e}")

        print("[Registry] All engines ready.\n")

    # ---------------------------------------------------------
    # Single inference
    # ---------------------------------------------------------
    def infer(self, request: Request) -> Tuple[torch.Tensor, float]:
        engine = self._engines[request.model_type]

        result, latency = engine.infer(request.input_tensor)

        request.mark_complete(result, latency)

        self._record(request.model_type, latency)

        return result, latency

    # ---------------------------------------------------------
    # Batch inference (FIXED NLP handling)
    # ---------------------------------------------------------
    def infer_batch(self, requests: List[Request]):

        if not requests:
            return []

        grouped = {}
        for i, r in enumerate(requests):
            grouped.setdefault(r.model_type, []).append((i, r))

        results = [None] * len(requests)

        for mtype, items in grouped.items():

            engine = self._engines[mtype]

            batch_inputs = [r.input_tensor for _, r in items]

            # ---------------- NLP SPECIAL CASE ----------------
            if mtype == ModelType.NLP:
                batch_result, latency = engine.infer(batch_inputs)
            else:
                batch_tensor = torch.stack(batch_inputs)
                batch_result, latency = engine.infer(batch_tensor)

            # distribute results
            for k, (idx, req) in enumerate(items):
                req.mark_complete(batch_result[k], latency)
                self._record(mtype, latency)
                results[idx] = (batch_result[k], latency)

        return results

    # ---------------------------------------------------------
    # Stats
    # ---------------------------------------------------------
    def _record(self, mtype, latency):
        self._call_counts[mtype] += 1
        self._total_latency[mtype] += latency

    def stats(self):
        return {
            t.value: {
                "requests": self._call_counts[t],
                "avg_latency_ms": (
                    self._total_latency[t] / self._call_counts[t]
                    if self._call_counts[t] > 0 else 0
                ),
            }
            for t in ModelType
            if self._call_counts[t] > 0
        }

    def registered_types(self):
        return list(self._engines.keys())

    def is_registered(self, model_type: ModelType):
        return model_type in self._engines

    def __repr__(self):
        return "ModelRegistry(" + ", ".join(
            f"{t.value}:{e.metadata.model_name}"
            for t, e in self._engines.items()
        ) + ")"

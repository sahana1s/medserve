"""
models/registry.py — MedServe model registry (FIXED for NLP compatibility)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch

from system.request import Request, ModelType
from models.base import BaseInferenceEngine, ModelMetadata


class ModelRegistry:

    def __init__(self):
        self._engines: Dict[ModelType, BaseInferenceEngine] = {}
        self._calls = {t: 0 for t in ModelType}
        self._latency = {t: 0.0 for t in ModelType}

    # ---------------------------------------------------------
    # Register
    # ---------------------------------------------------------

    def register(self, engine: BaseInferenceEngine):
        mtype = engine.metadata.model_type

        if mtype in self._engines:
            print(f"[Registry] Replacing {mtype.value}")
        else:
            print(f"[Registry] Registered {engine.metadata.model_name} for {mtype.value}")

        self._engines[mtype] = engine
        return self

    def swap(self, model_type: ModelType, engine: BaseInferenceEngine):
        return self.register(engine)

    # added
    def get_metadata(self, model_type: ModelType) -> ModelMetadata:
        return self._engines[model_type].metadata

    # ---------------------------------------------------------
    # Default registry
    # ---------------------------------------------------------

    @classmethod
    def default(
        cls,
        weights_dir: str = "results/weights",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_fp16: bool = True,
    ):

        from models.icu_model import ICUInferenceEngine
        from models.imaging_model import ImagingInferenceEngine
        from models.nlp_model import NLPInferenceEngine

        w = Path(weights_dir)

        def p(name): 
            return str(w / name) if (w / name).exists() else None

        r = cls()

        # ICU
        r.register(ICUInferenceEngine(
            model_path=p("icu_model.pt"),
            device=device,
            use_fp16=use_fp16,
        ))

        # Imaging
        r.register(ImagingInferenceEngine(
            model_path=p("imaging_model.pt"),
            device=device,
            use_fp16=False,
        ))

        # NLP (FIXED: no extra args like local_only)
        r.register(NLPInferenceEngine(
            model_path=p("nlp_model.pt"),
            device=device,
            use_fp16=use_fp16,
        ))

        return r

    # ---------------------------------------------------------
    # Warmup
    # ---------------------------------------------------------

    def warmup_all(self, n_runs=20):
        print("\n[Registry] Warming up and measuring actual latencies...")
        for mtype, engine in self._engines.items():
            if not hasattr(engine, "warmup"):
                continue
            latencies = engine.warmup(n_runs=n_runs)
            if not latencies:
                continue
            latencies.sort()
            avg = sum(latencies) / len(latencies)
            p99 = latencies[min(len(latencies)-1, int(0.99 * len(latencies)))]
            print(
                f"  {mtype.value:8s}  "
                f"avg={avg:.1f}ms  p99={p99:.1f}ms"
            )
        print("[Registry] Ready.\n")

    # ---------------------------------------------------------
    # Inference
    # ---------------------------------------------------------

    def infer(self, request: Request):
        engine = self._engines[request.model_type]
        out, latency = engine.infer(request.input_tensor)

        request.mark_complete(out, latency)

        self._calls[request.model_type] += 1
        self._latency[request.model_type] += latency

        return out, latency

    def infer_batch(self, requests: List[Request]):
        if not requests:
            return []

        grouped = {}
        for i, r in enumerate(requests):
            grouped.setdefault(r.model_type, []).append((i, r))

        results = [None] * len(requests)

        for mtype, items in grouped.items():
            engine = self._engines[mtype]

            idxs = [i for i, _ in items]
            reqs = [r for _, r in items]

            inputs = [r.input_tensor for r in reqs]

            batch_out, latency = engine.infer(inputs)

            for k, req in enumerate(reqs):
                out = batch_out[k] if batch_out.ndim > 1 else batch_out
                req.mark_complete(out, latency)

                results[idxs[k]] = (out, latency)

                self._calls[mtype] += 1
                self._latency[mtype] += latency

        return results

    # ---------------------------------------------------------
    # Stats
    # ---------------------------------------------------------

    def stats(self):
        return {
            t.value: {
                "requests": self._calls[t],
                "avg_latency_ms": (
                    self._latency[t] / self._calls[t]
                    if self._calls[t] else 0
                )
            }
            for t in self._engines
        }

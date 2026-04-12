"""
models/registry.py — Pluggable model registry for MedServe.

The registry is the ONLY thing the scheduler and serving system import.
Models are registered by name, not imported directly. This means:

  1. Adding a new model = one function call, no changes to scheduler code
  2. Swapping a model = re-register under the same ModelType
  3. Running ablations = register multiple variants, benchmark all

Usage:

    # --- Register built-in models ---
    registry = ModelRegistry.default()
    registry.warmup_all()

    # --- Register a custom model ---
    from my_new_model import MyECGEngine
    registry.register(MyECGEngine(model_path="weights/ecg.pt"))

    # --- Single inference ---
    result, latency_ms = registry.infer(request)

    # --- Batch inference ---
    results = registry.infer_batch(requests)

    # --- Swap model at runtime (for ablation studies) ---
    registry.swap(ModelType.ICU, NewICUEngine(model_path="weights/new_icu.pt"))
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type
import torch

from system.request import Request, ModelType
from models.base import BaseInferenceEngine, ModelMetadata


class ModelRegistry:
    """
    Central store for all active inference engines.

    Key design decisions:
    - Engines are stored by ModelType (one active engine per workload)
    - infer() dispatches to the right engine automatically
    - infer_batch() handles mixed-type batches by grouping by type
    - All latency measurements flow through here for uniform logging
    """

    def __init__(self):
        self._engines: Dict[ModelType, BaseInferenceEngine] = {}
        self._call_counts: Dict[ModelType, int] = {t: 0 for t in ModelType}
        self._total_latency: Dict[ModelType, float] = {t: 0.0 for t in ModelType}

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    def register(self, engine: BaseInferenceEngine) -> "ModelRegistry":
        """
        Register an inference engine for its model_type.
        Overwrites any existing engine for that type (allows hot-swapping).

        Returns self for chaining:
            registry = (ModelRegistry()
                .register(ICUEngine())
                .register(ImagingEngine())
                .register(NLPEngine()))
        """
        mtype = engine.metadata.model_type
        if mtype in self._engines:
            print(f"[Registry] Replacing {self._engines[mtype].metadata.model_name} "
                  f"with {engine.metadata.model_name} for {mtype.value}")
        else:
            print(f"[Registry] Registered {engine.metadata.model_name} "
                  f"for {mtype.value} (SLA={engine.metadata.sla_ms}ms)")
        self._engines[mtype] = engine
        return self

    def swap(self, model_type: ModelType, new_engine: BaseInferenceEngine):
        """Hot-swap a model at runtime. Used for ablation studies."""
        self.register(new_engine)

    def unregister(self, model_type: ModelType):
        """Remove an engine from the registry."""
        if model_type in self._engines:
            del self._engines[model_type]

    # -------------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------------

    @classmethod
    def default(
        cls,
        weights_dir: str = "results/weights",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_fp16: bool = True,
    ) -> "ModelRegistry":
        """
        Build the standard three-workload registry.
        Loads weights from weights_dir if .pt files exist there,
        otherwise falls back to pretrained/random init for testing.

        weights_dir layout expected after Kaggle training:
            results/weights/
                icu_model.pt
                imaging_model.pt
                nlp_model.pt
        """
        # Import here to avoid circular imports
        from models.icu_model import ICUInferenceEngine
        from models.imaging_model import ImagingInferenceEngine
        from models.nlp_model import NLPInferenceEngine

        weights = Path(weights_dir)

        def maybe_path(name: str) -> Optional[str]:
            p = weights / name
            return str(p) if p.exists() else None

        registry = cls()
        registry.register(ICUInferenceEngine(
            model_path=maybe_path("icu_model.pt"),
            device=device, use_fp16=use_fp16,
        ))
        registry.register(ImagingInferenceEngine(
            model_path=maybe_path("imaging_model.pt"),
            device=device, use_fp16=use_fp16,
        ))
        registry.register(NLPInferenceEngine(
            model_path=maybe_path("nlp_model.pt"),
            device=device, use_fp16=use_fp16,
        ))
        return registry

    @classmethod
    def from_config(cls, config_path: str) -> "ModelRegistry":
        """
        Load registry from a JSON config file.
        Useful for switching between experiment configurations.

        Config format:
        {
          "device": "cuda",
          "use_fp16": true,
          "models": [
            {
              "class": "ICUInferenceEngine",
              "model_path": "results/weights/icu_model.pt"
            },
            ...
          ]
        }
        """
        import importlib
        with open(config_path) as f:
            config = json.load(f)

        registry = cls()
        device   = config.get("device", "cpu")
        use_fp16 = config.get("use_fp16", True)

        for m in config.get("models", []):
            module = importlib.import_module("models." + m["module"])
            cls_   = getattr(module, m["class"])
            engine = cls_(
                model_path=m.get("model_path"),
                device=device,
                use_fp16=use_fp16,
            )
            registry.register(engine)

        return registry

    # -------------------------------------------------------------------------
    # Warmup
    # -------------------------------------------------------------------------

    def warmup_all(self, n_runs: int = 20):
        """
        Warm up all registered engines and benchmark batch sizes.
        Call once at server startup before serving any requests.
        """
        print("\n[Registry] Warming up all engines...")
        for engine in self._engines.values():
            engine.warmup(n_runs=n_runs)
            engine.benchmark_batch_sizes()
        print("[Registry] All engines ready.\n")

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    def infer(self, request: Request) -> Tuple[torch.Tensor, float]:
        """
        Dispatch a single request to its engine.
        Updates request in-place via request.mark_complete().
        """
        engine = self._get_engine(request.model_type)
        result, latency = engine.infer(request.input_tensor)
        request.mark_complete(result, latency)
        self._record(request.model_type, latency)
        return result, latency

    def infer_batch(self, requests: List[Request]) -> List[Tuple[torch.Tensor, float]]:
        """
        Dispatch a batch of requests. Handles mixed model types by grouping.

        The scheduler typically sends same-type batches for efficiency,
        but this method handles mixed batches gracefully for robustness.

        Returns list of (result, latency_ms) in same order as input.
        """
        if not requests:
            return []

        # Group by model type while preserving order
        groups: Dict[ModelType, List[Tuple[int, Request]]] = {}
        for i, req in enumerate(requests):
            groups.setdefault(req.model_type, []).append((i, req))

        results = [None] * len(requests)

        for model_type, indexed_reqs in groups.items():
            engine     = self._get_engine(model_type)
            indices    = [i for i, _ in indexed_reqs]
            batch_reqs = [r for _, r in indexed_reqs]

            # Collect inputs
            if model_type == ModelType.NLP:
                inputs = [r.input_tensor for r in batch_reqs]   # list of strings
            else:
                inputs = [r.input_tensor for r in batch_reqs]   # list of tensors

            # Run batch inference
            batch_result, latency = engine.infer(inputs)

            # Distribute results back to individual requests
            for k, req in enumerate(batch_reqs):
                single_result = (
                    batch_result[k].unsqueeze(0)
                    if batch_result.dim() > 0
                    else batch_result
                )
                req.mark_complete(single_result, latency)
                self._record(model_type, latency)
                results[indices[k]] = (single_result, latency)

        return results

    # -------------------------------------------------------------------------
    # Introspection
    # -------------------------------------------------------------------------

    def get_metadata(self, model_type: ModelType) -> ModelMetadata:
        return self._get_engine(model_type).metadata

    def all_metadata(self) -> Dict[str, dict]:
        """Serialize all engine metadata. Used for paper's system description table."""
        return {
            mtype.value: engine.metadata.to_dict()
            for mtype, engine in self._engines.items()
        }

    def stats(self) -> Dict[str, dict]:
        """Runtime statistics per model type."""
        out = {}
        for mtype, count in self._call_counts.items():
            if count == 0:
                continue
            out[mtype.value] = {
                "requests_served": count,
                "avg_latency_ms":  round(self._total_latency[mtype] / count, 2),
            }
        return out

    def registered_types(self) -> List[ModelType]:
        return list(self._engines.keys())

    def is_registered(self, model_type: ModelType) -> bool:
        return model_type in self._engines

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _get_engine(self, model_type: ModelType) -> BaseInferenceEngine:
        if model_type not in self._engines:
            raise KeyError(
                f"No engine registered for {model_type.value}. "
                f"Registered: {[t.value for t in self._engines]}"
            )
        return self._engines[model_type]

    def _record(self, model_type: ModelType, latency_ms: float):
        self._call_counts[model_type]  += 1
        self._total_latency[model_type] += latency_ms

    def __repr__(self) -> str:
        engines = ", ".join(
            f"{t.value}={e.metadata.model_name}"
            for t, e in self._engines.items()
        )
        return f"ModelRegistry([{engines}])"

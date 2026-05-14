"""
models/base.py — Abstract base class for all MedServe inference engines.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, Any, Dict, Optional, List
import torch

from system.request import ModelType, Priority, SLA_DEADLINES_MS


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

@dataclass
class ModelMetadata:
    model_type: ModelType
    model_name: str
    input_shape: Optional[Tuple]   # None for variable-length models
    framework: str = "pytorch"

    priority: Priority = field(init=False)
    sla_ms: float = field(init=False)

    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_batch_size: int = 1

    def __post_init__(self):
        from system.request import WORKLOAD_PRIORITY
        self.priority = WORKLOAD_PRIORITY[self.model_type]
        self.sla_ms = SLA_DEADLINES_MS[self.priority]

    @property
    def dispatch_threshold_ms(self) -> float:
        ALPHA = 1.5
        return ALPHA * max(self.avg_latency_ms, 1.0)

    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "model_type": self.model_type.value,
            "priority": self.priority.name,
            "sla_ms": self.sla_ms,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "max_batch_size": self.max_batch_size,
            "framework": self.framework,
        }


# ---------------------------------------------------------------------------
# Base Engine
# ---------------------------------------------------------------------------

class BaseInferenceEngine(ABC):

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_fp16: bool = True,
    ):
        self.model_path = model_path
        self.device = torch.device(device)
        self.use_fp16 = use_fp16 and (self.device.type == "cuda")

        self.model = None
        self._is_loaded = False

        self._load_model()
        self._apply_fp16()
        self._is_loaded = True

    # ---------------- REQUIRED METHODS ----------------

    @abstractmethod
    def _load_model(self):
        ...

    @abstractmethod
    def _forward(self, x: Any) -> torch.Tensor:
        ...

    @abstractmethod
    def preprocess(self, raw_input: Any) -> Any:
        ...

    @property
    @abstractmethod
    def metadata(self) -> ModelMetadata:
        ...

    # ---------------- BATCHING (MUST OVERRIDE) ----------------

    def _prepare_batch(self, inputs: List[Any]) -> Any:
        """
        IMPORTANT:
        Base class no longer assumes tensor shapes match.

        Every engine MUST override this.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _prepare_batch()"
        )

    # ---------------- INTERNAL ----------------

    def _apply_fp16(self):
        if self.model is None:
            return
        self.model.eval()
        if self.use_fp16:
            try:
                self.model = self.model.half()
            except Exception:
                self.use_fp16 = False

    # ---------------- MAIN INFERENCE ----------------

    @torch.no_grad()
    def infer(self, inputs: Any) -> Tuple[torch.Tensor, float]:

        if not isinstance(inputs, list):
            inputs = [inputs]

        t_start = time.perf_counter()

        batch = self._prepare_batch(inputs)

        if self.use_fp16 and isinstance(batch, torch.Tensor):
            batch = batch.half()

        result = self._forward(batch)

        if isinstance(result, torch.Tensor):
            result = result.float().cpu()

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        latency_ms = (time.perf_counter() - t_start) * 1000

        return result, latency_ms

    # ---------------- WARMUP (FIXED) ----------------

    def warmup(self, n_runs: int = 10):

        if self.metadata.input_shape is None:
            dummy = ["warmup text"] * 4
        else:
            dummy = [
                torch.zeros(1, *self.metadata.input_shape)
                for _ in range(4)
            ]

        latencies = []

        for _ in range(n_runs):
            # FIX: test full batch path, not single sample
            _, ms = self.infer(dummy)
            latencies.append(ms)

        latencies.sort()

        self.metadata.avg_latency_ms = sum(latencies) / len(latencies)
        self.metadata.p99_latency_ms = latencies[int(0.99 * len(latencies))]

        print(
            f"[{self.metadata.model_name}] Warmup complete: "
            f"avg={self.metadata.avg_latency_ms:.1f}ms  "
            f"p99={self.metadata.p99_latency_ms:.1f}ms  "
            f"dispatch_threshold={self.metadata.dispatch_threshold_ms:.1f}ms"
        )

    # ---------------- OPTIONAL BENCHMARK ----------------

    def benchmark_batch_sizes(self, sizes: list = None) -> Dict[int, float]:

        if self.metadata.input_shape is None:
            print(f"[{self.metadata.model_name}] Skipping batch benchmark")
            return {}

        sizes = sizes or [1, 2, 4, 8, 16, 32]
        results = {}

        for bs in sizes:
            dummy = [
                torch.zeros(1, *self.metadata.input_shape)
                for _ in range(bs)
            ]

            lat = []
            for _ in range(5):
                _, ms = self.infer(dummy)
                lat.append(ms)

            avg = sum(lat) / len(lat)
            results[bs] = avg

            print(
                f"  batch={bs:3d}: {avg:7.2f}ms "
                f"({'OK' if avg < self.metadata.sla_ms else 'SLA FAIL'})"
            )

        self.metadata.max_batch_size = max(
            (b for b, v in results.items() if v < self.metadata.sla_ms),
            default=1
        )

        return results

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "unloaded"
        fp = "FP16" if self.use_fp16 else "FP32"
        return (
            f"{self.__class__.__name__}("
            f"model={self.metadata.model_name}, "
            f"device={self.device}, {fp}, {status})"
        )

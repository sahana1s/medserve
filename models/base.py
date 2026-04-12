"""
models/base.py — Abstract base class for all MedServe inference engines.

Every model you plug into MedServe must implement this interface.
The scheduler, batcher, and registry ONLY speak this language.

This is what makes the system model-agnostic:
  - Swap LSTM → Transformer for ICU? Only the model file changes.
  - Add a new workload (e.g. ECG classification)? Subclass BaseInferenceEngine.
  - Compare ResNet18 vs ViT for imaging? Both implement the same interface.

The scheduler never imports a concrete model class directly.
It only ever calls engine.infer(tensor) and reads engine.metadata.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, Any, Dict, Optional
import torch

from system.request import ModelType, Priority, SLA_DEADLINES_MS


# ---------------------------------------------------------------------------
# Model metadata — what the scheduler needs to know about a model
# ---------------------------------------------------------------------------

@dataclass
class ModelMetadata:
    """
    Static description of a model's resource profile and SLA contract.
    The scheduler reads this to make batching and dispatch decisions.

    Fields:
        model_type      — which workload this handles (ICU / IMAGING / NLP)
        priority        — clinical urgency tier (derived from model_type)
        sla_ms          — SLA deadline in milliseconds
        avg_latency_ms  — measured average single-sample inference latency
                          (populated after warmup, used for dispatch threshold)
        max_batch_size  — largest batch this model can handle within SLA
        input_shape     — expected shape of a single input sample (no batch dim)
                          e.g. (48, 34) for ICU, (3, 224, 224) for imaging
                          For text models: None (variable length)
        model_name      — human-readable name for logging and paper tables
        framework       — "pytorch", "onnx", "tensorrt" — for paper's methods section
    """
    model_type:     ModelType
    model_name:     str
    input_shape:    Optional[Tuple]   # None for variable-length (NLP)
    framework:      str = "pytorch"

    # Derived automatically
    priority:       Priority          = field(init=False)
    sla_ms:         float             = field(init=False)

    # Populated after warmup (set by BaseInferenceEngine.warmup())
    avg_latency_ms: float             = field(default=0.0)
    p99_latency_ms: float             = field(default=0.0)
    max_batch_size: int               = field(default=1)

    def __post_init__(self):
        from system.request import WORKLOAD_PRIORITY
        self.priority = WORKLOAD_PRIORITY[self.model_type]
        self.sla_ms   = SLA_DEADLINES_MS[self.priority]

    @property
    def dispatch_threshold_ms(self) -> float:
        """
        If a request has less than this much time remaining,
        dispatch it immediately regardless of batch fill level.
        ALPHA=1.5 means: dispatch when 1.5x the avg inference time remains.
        """
        ALPHA = 1.5
        return ALPHA * max(self.avg_latency_ms, 1.0)

    def to_dict(self) -> Dict:
        """Serialize for logging and paper tables."""
        return {
            "model_name":     self.model_name,
            "model_type":     self.model_type.value,
            "priority":       self.priority.name,
            "sla_ms":         self.sla_ms,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "max_batch_size": self.max_batch_size,
            "framework":      self.framework,
        }


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseInferenceEngine(ABC):
    """
    Contract that every MedServe model must implement.

    Subclass this for any new model. The only required methods are:
        _load_model()   — load weights, build model, move to device
        _forward()      — raw forward pass, returns tensor
        preprocess()    — convert raw input to model-ready tensor

    Everything else (timing, FP16, warmup, metadata population) is
    handled here so subclasses stay clean and focused on the model.

    Example — adding a new ECG model:

        class ECGInferenceEngine(BaseInferenceEngine):
            def _load_model(self):
                self.model = ECGTransformer(...)
                self.model.load_state_dict(torch.load(self.model_path))

            def _forward(self, x):
                return self.model(x)

            def preprocess(self, raw_input):
                # raw_input: numpy array of shape (5000,) — 10s at 500Hz
                return torch.tensor(raw_input).float().unsqueeze(0)

            @property
            def metadata(self):
                return ModelMetadata(
                    model_type=ModelType.ECG,
                    model_name="ECGTransformer-v1",
                    input_shape=(5000,),
                )
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device:     str = "cuda" if torch.cuda.is_available() else "cpu",
        use_fp16:   bool = True,
    ):
        self.model_path = model_path
        self.device     = torch.device(device)
        self.use_fp16   = use_fp16 and (self.device.type == "cuda")
        self.model      = None
        self._is_loaded = False

        self._load_model()
        self._apply_fp16()
        self._is_loaded = True

    # --- Abstract interface (subclasses must implement) ---

    @abstractmethod
    def _load_model(self):
        """
        Load the model into self.model.
        Move to self.device. Do not call .eval() here — base class handles it.
        """
        ...

    @abstractmethod
    def _forward(self, x: Any) -> torch.Tensor:
        """
        Raw forward pass. Receives whatever _prepare_batch() returns.
        Must return a torch.Tensor on CPU in float32.
        Do NOT add timing here — base class handles timing.
        """
        ...

    @abstractmethod
    def preprocess(self, raw_input: Any) -> Any:
        """
        Convert a single raw input sample into model-ready format.
        Called by the request generator and test harness.

        For ICU:     numpy array (48, 34)  → torch.Tensor (1, 48, 34)
        For Imaging: PIL.Image or filepath → torch.Tensor (1, 3, 224, 224)
        For NLP:     string               → string (tokenization happens in _forward)
        """
        ...

    @property
    @abstractmethod
    def metadata(self) -> ModelMetadata:
        """Return this model's static metadata descriptor."""
        ...

    # --- Provided by base class (do not override) ---

    def _apply_fp16(self):
        """Convert model to FP16 if GPU is available and use_fp16 is set."""
        if self.model is None:
            return
        self.model.eval()
        if self.use_fp16:
            try:
                self.model = self.model.half()
            except Exception:
                self.use_fp16 = False  # fallback silently

    def _prepare_batch(self, inputs: list) -> Any:
        """
        Stack a list of preprocessed inputs into a batch.
        Override this if your model needs custom batching (e.g. NLP padding).
        Default: torch.cat along dim=0.
        """
        return torch.cat(inputs, dim=0).to(self.device)

    @torch.no_grad()
    def infer(self, inputs: Any) -> Tuple[torch.Tensor, float]:
        """
        Main entry point. Called by ModelRegistry for every request.

        Args:
            inputs: Either a single preprocessed sample OR a list of samples.
                    For NLP: a string or list of strings.
                    For others: tensor(s).

        Returns:
            result:            (batch_size, output_dim) tensor, float32, on CPU
            inference_time_ms: wall-clock time including device transfer
        """
        # Normalize to list
        if not isinstance(inputs, list):
            inputs = [inputs]

        t_start = time.perf_counter()

        # Prepare batch (handles stacking + device transfer)
        batch = self._prepare_batch(inputs)

        # Cast to FP16 if enabled (for tensor inputs)
        if self.use_fp16 and isinstance(batch, torch.Tensor):
            batch = batch.half()

        # Forward pass
        result = self._forward(batch)

        # Ensure float32 output on CPU
        if isinstance(result, torch.Tensor):
            result = result.float().cpu()

        # GPU sync before stopping timer (critical for accurate GPU timing)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        inference_time_ms = (time.perf_counter() - t_start) * 1000

        return result, inference_time_ms

    def warmup(self, n_runs: int = 10):
        """
        Run n_runs warmup inferences and populate metadata latency fields.
        Call this once at startup before serving real requests.
        The scheduler uses avg_latency_ms to compute dispatch thresholds.
        """
        if self.metadata.input_shape is None:
            # NLP / variable-length: use a short dummy string
            dummy_inputs = ["warmup text"] * 4
        else:
            # Fixed-shape models: create dummy tensors
            dummy_inputs = [
                torch.zeros(1, *self.metadata.input_shape)
                for _ in range(4)
            ]

        latencies = []
        for _ in range(n_runs):
            _, ms = self.infer(dummy_inputs[0])
            latencies.append(ms)

        latencies.sort()
        self.metadata.avg_latency_ms = sum(latencies) / len(latencies)
        self.metadata.p99_latency_ms = latencies[int(0.99 * len(latencies))]

        print(f"[{self.metadata.model_name}] Warmup complete: "
              f"avg={self.metadata.avg_latency_ms:.1f}ms  "
              f"p99={self.metadata.p99_latency_ms:.1f}ms  "
              f"dispatch_threshold={self.metadata.dispatch_threshold_ms:.1f}ms")

    def benchmark_batch_sizes(self, sizes: list = None) -> Dict[int, float]:
        """
        Measure latency at different batch sizes.
        Used to set max_batch_size and for paper's batch size analysis.

        Returns: {batch_size: avg_latency_ms}
        """
        if self.metadata.input_shape is None:
            print(f"[{self.metadata.model_name}] Skipping batch benchmark (variable-length input)")
            return {}

        sizes = sizes or [1, 2, 4, 8, 16, 32]
        results = {}

        for bs in sizes:
            dummy = [torch.zeros(1, *self.metadata.input_shape) for _ in range(bs)]
            latencies = []
            for _ in range(5):
                _, ms = self.infer(dummy)
                latencies.append(ms)
            avg_ms = sum(latencies) / len(latencies)
            results[bs] = avg_ms
            within_sla = avg_ms < self.metadata.sla_ms
            print(f"  batch_size={bs:3d}: {avg_ms:7.2f}ms  "
                  f"({'within' if within_sla else 'EXCEEDS'} SLA={self.metadata.sla_ms}ms)")

        # Set max_batch_size to largest batch that fits within SLA
        self.metadata.max_batch_size = max(
            (bs for bs, ms in results.items() if ms < self.metadata.sla_ms),
            default=1
        )
        print(f"  → max_batch_size set to {self.metadata.max_batch_size}\n")
        return results

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "unloaded"
        fp     = "FP16" if self.use_fp16 else "FP32"
        return (f"{self.__class__.__name__}("
                f"model={self.metadata.model_name}, "
                f"device={self.device}, {fp}, {status})")

"""
request.py — Core data structures for MedServe.

Every request flowing through the system is represented as a Request object.
This is the single most important dataclass in the project — everything
the scheduler, batcher, and metrics logger touches comes from here.
"""

import uuid
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import torch


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ModelType(Enum):
    """The three healthcare AI workloads in our system."""
    ICU       = "icu"       # LSTM: sepsis/mortality prediction from vitals
    IMAGING   = "imaging"   # ResNet18: chest X-ray classification
    NLP       = "nlp"       # DistilBERT: clinical text classification


class Priority(Enum):
    """
    Clinical urgency tier. Maps directly to SLA deadlines.
    Higher integer = higher priority (used for heap ordering).
    """
    LOW    = 1   # Imaging: routine, batchable, 500ms SLA
    MID    = 2   # NLP: EHR assistant, interactive, 300ms SLA
    HIGH   = 3   # ICU: critical alerts, 100ms SLA


# ---------------------------------------------------------------------------
# SLA Deadlines (milliseconds) — grounded in clinical literature
# ---------------------------------------------------------------------------

SLA_DEADLINES_MS = {
    Priority.HIGH: 100,   # Must beat bedside monitor refresh rate (~100ms)
    Priority.MID:  300,   # Nielsen's interaction threshold for "immediate" UX
    Priority.LOW:  500,   # Must load before radiologist opens the image in PACS
}

# Default model type → priority mapping
WORKLOAD_PRIORITY = {
    ModelType.ICU:     Priority.HIGH,
    ModelType.NLP:     Priority.MID,
    ModelType.IMAGING: Priority.LOW,
}


# ---------------------------------------------------------------------------
# Request dataclass
# ---------------------------------------------------------------------------

@dataclass
class Request:
    """
    A single inference request flowing through the MedServe system.

    Fields set at creation time:
        request_id      — unique ID for tracking and logging
        model_type      — which model handles this request
        priority        — clinical urgency tier (derived from model_type)
        deadline_ms     — absolute wall-clock deadline (ms since epoch)
        input_tensor    — preprocessed input ready for model inference
        arrival_time_ms — wall-clock time when request entered the system

    Fields set after inference:
        result          — model output (probabilities or class predictions)
        inference_time_ms — how long inference actually took
        total_latency_ms  — end-to-end time from arrival to result
        sla_violated    — True if total_latency_ms > SLA_DEADLINES_MS[priority]
    """

    # --- Set at creation ---
    model_type:      ModelType
    input_tensor:    torch.Tensor
    request_id:      str             = field(default_factory=lambda: str(uuid.uuid4())[:8])
    arrival_time_ms: float           = field(default_factory=lambda: time.time() * 1000)
    priority:        Priority        = field(init=False)
    deadline_ms:     float           = field(init=False)

    # --- Set after inference ---
    result:             Optional[torch.Tensor] = field(default=None, init=False)
    inference_time_ms:  Optional[float]        = field(default=None, init=False)
    total_latency_ms:   Optional[float]        = field(default=None, init=False)
    sla_violated:       Optional[bool]         = field(default=None, init=False)

    def __post_init__(self):
        self.priority    = WORKLOAD_PRIORITY[self.model_type]
        self.deadline_ms = self.arrival_time_ms + SLA_DEADLINES_MS[self.priority]

    # --- Computed properties ---

    @property
    def time_remaining_ms(self) -> float:
        """Milliseconds until this request's SLA deadline expires."""
        return self.deadline_ms - (time.time() * 1000)

    @property
    def is_critical(self) -> bool:
        """
        True when less than ALPHA * estimated_inference_time remains.
        Used by the scheduler to force immediate dispatch.
        Set ALPHA = 1.5 (scheduler can tune this).
        """
        return self.time_remaining_ms < 0

    @property
    def urgency_score(self) -> float:
        """
        Core scheduler metric. Higher = serve sooner.

        urgency = tier_weight / max(time_remaining_ms, ε)

        As deadline approaches → time_remaining → 0 → urgency → ∞
        Higher priority tier → larger base weight → higher urgency at all times
        """
        TIER_WEIGHTS = {
            Priority.HIGH: 3.0,
            Priority.MID:  1.5,
            Priority.LOW:  1.0,
        }
        epsilon = 0.001  # prevent division by zero
        remaining = max(self.time_remaining_ms, epsilon)
        return TIER_WEIGHTS[self.priority] / remaining

    # --- Lifecycle methods ---

    def mark_complete(self, result: torch.Tensor, inference_time_ms: float):
        """Call this immediately after inference completes."""
        self.result            = result
        self.inference_time_ms = inference_time_ms
        self.total_latency_ms  = (time.time() * 1000) - self.arrival_time_ms
        self.sla_violated      = self.total_latency_ms > SLA_DEADLINES_MS[self.priority]

    # --- Comparison (for priority queue ordering) ---

    def __lt__(self, other: "Request") -> bool:
        """Higher urgency score = higher priority in the heap."""
        return self.urgency_score > other.urgency_score

    def __repr__(self) -> str:
        return (
            f"Request(id={self.request_id}, model={self.model_type.value}, "
            f"priority={self.priority.name}, remaining={self.time_remaining_ms:.1f}ms)"
        )

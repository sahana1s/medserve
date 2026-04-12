"""
benchmarks/mode_medserve.py — MedServe SLA-aware scheduler benchmark.

This is your main contribution. Runs the same workload through your
urgency-score + adaptive-batching scheduler and measures whether
it protects ICU SLAs while maintaining imaging throughput.

Scheduler algorithm (paper section 3.2):
    Every TICK_MS milliseconds:
    1. Score each queued request: urgency = tier_weight / time_remaining
    2. If any request has time_remaining < ALPHA * avg_inference_ms: dispatch immediately
    3. Otherwise: collect top-k by urgency, dispatch when batch is full or timeout
    4. Adaptive batching: if HIGH queue pressure, halve batch size for LOW/MID work
    5. Aging: multiply urgency by AGING_FACTOR every AGING_INTERVAL_S of waiting

Hyperparameters (tune these in week 7):
    TICK_MS           = 10       scheduling interval
    ALPHA             = 1.5      dispatch threshold multiplier
    MAX_BATCH         = 16       maximum batch size
    HIGH_PRESSURE_THR = 3        ICU queue depth that triggers batch reduction
    AGING_FACTOR      = 1.2      urgency multiplier per aging interval
    AGING_INTERVAL_S  = 2.0      how often aging applies

Run on Kaggle/Colab:
    from benchmarks.mode_medserve import run_medserve_benchmark
    results = run_medserve_benchmark(load_level="medium")
"""

import sys, time, copy, threading
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from benchmarks.workload import WorkloadGenerator, WorkloadRequest, LoadLevel, LOAD_RATES, DEFAULT_MIX
from benchmarks.metrics  import MetricsCollector


# ---------------------------------------------------------------------------
# Scheduler hyperparameters — tune these later
# ---------------------------------------------------------------------------

class SchedulerConfig:
    """
    All tunable hyperparameters in one place.
    Change these for sensitivity analysis experiments.
    """
    TICK_MS:            float = 10.0    # scheduling loop interval
    ALPHA:              float = 1.5     # dispatch when time_remaining < ALPHA * avg_inf_ms
    MAX_BATCH:          int   = 16      # max requests per dispatch
    HIGH_PRESSURE_THR:  int   = 3       # ICU queue depth → halve batch size
    AGING_FACTOR:       float = 1.2     # urgency multiplier per aging interval
    AGING_INTERVAL_S:   float = 2.0     # how often aging is applied
    TIER_WEIGHTS: Dict  = None          # HIGH/MID/LOW urgency base weights

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        if self.TIER_WEIGHTS is None:
            self.TIER_WEIGHTS = {"icu": 3.0, "nlp": 1.5, "imaging": 1.0}

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Queued request wrapper (adds scheduling metadata)
# ---------------------------------------------------------------------------

class QueuedRequest:
    def __init__(self, req: WorkloadRequest, config: SchedulerConfig):
        self.req          = req
        self.config       = config
        self.queued_at_ms = time.perf_counter() * 1000.0
        self.aging_mult   = 1.0     # increases over time via aging

    @property
    def time_remaining_ms(self) -> float:
        deadline = self.req.sent_at_ms + self.req.sla_ms
        return deadline - (time.perf_counter() * 1000.0)

    @property
    def urgency(self) -> float:
        weight = self.config.TIER_WEIGHTS.get(self.req.model_type, 1.0)
        remaining = max(self.time_remaining_ms, 0.001)
        return (weight / remaining) * self.aging_mult

    def apply_aging(self):
        self.aging_mult *= self.config.AGING_FACTOR


# ---------------------------------------------------------------------------
# SLA-aware scheduler
# ---------------------------------------------------------------------------

class MedServeScheduler:
    """
    The core research contribution.

    Three priority queues (icu / nlp / imaging).
    Scheduling loop runs every TICK_MS milliseconds.
    Dispatches batches based on urgency score + SLA deadline proximity.
    """

    def __init__(self, registry, config: SchedulerConfig):
        self.registry      = registry
        self.config        = config
        self.queues: Dict[str, List[QueuedRequest]] = {
            "icu": [], "nlp": [], "imaging": []
        }
        self.lock          = threading.Lock()
        self._stop_event   = threading.Event()
        self._results: Dict[str, tuple] = {}    # request_id → (result, inf_ms)
        self._result_events: Dict[str, threading.Event] = {}
        self._last_aging   = time.perf_counter()
        self._avg_inf_ms: Dict[str, float] = {"icu": 20.0, "nlp": 80.0, "imaging": 150.0}

    def submit(self, req: WorkloadRequest) -> threading.Event:
        """Add a request to the appropriate priority queue. Returns an Event that fires on completion."""
        event = threading.Event()
        with self.lock:
            self._result_events[req.request_id] = event
            self.queues[req.model_type].append(QueuedRequest(req, self.config))
        return event

    def get_result(self, request_id: str, timeout_s: float = 2.0):
        """Block until the request's result is ready. Returns (result, inf_ms)."""
        event = self._result_events.get(request_id)
        if event:
            event.wait(timeout=timeout_s)
        return self._results.get(request_id, (None, 9999.0))

    def start(self):
        """Start the scheduling loop in a background thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._scheduling_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the scheduling loop."""
        self._stop_event.set()
        self._thread.join(timeout=5.0)

    # -------------------------------------------------------------------------
    # Core scheduling loop
    # -------------------------------------------------------------------------

    def _scheduling_loop(self):
        """Runs every TICK_MS. Implements the paper's Algorithm 1."""
        while not self._stop_event.is_set():
            loop_start = time.perf_counter()

            with self.lock:
                self._apply_aging()
                self._dispatch_one_cycle()

            # Sleep for remainder of tick
            elapsed_ms = (time.perf_counter() - loop_start) * 1000.0
            sleep_ms   = max(0.0, self.config.TICK_MS - elapsed_ms)
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)

    def _apply_aging(self):
        """Increase urgency of long-waiting requests to prevent starvation."""
        now = time.perf_counter()
        if now - self._last_aging >= self.config.AGING_INTERVAL_S:
            for queue in self.queues.values():
                for qr in queue:
                    qr.apply_aging()
            self._last_aging = now

    def _dispatch_one_cycle(self):
        """
        One scheduling decision cycle. Called every TICK_MS.

        Algorithm:
        1. Check for CRITICAL requests (deadline imminent) → dispatch immediately
        2. Compute adaptive batch size based on ICU queue pressure
        3. Select top-k by urgency score → dispatch as batch
        """
        # --- Step 1: Emergency dispatch for critical requests ---
        critical = self._find_critical()
        if critical:
            self._dispatch(critical)
            return

        # --- Step 2: Compute adaptive batch size ---
        batch_size = self._compute_batch_size()

        # --- Step 3: Select highest-urgency candidates ---
        all_queued = [qr for q in self.queues.values() for qr in q]
        if not all_queued:
            return

        # Sort by urgency descending, take top batch_size
        candidates = sorted(all_queued, key=lambda r: r.urgency, reverse=True)[:batch_size]

        # Only dispatch if we have a full batch OR any request is nearly expired
        nearly_expired = any(
            qr.time_remaining_ms < self.config.ALPHA * self._avg_inf_ms.get(qr.req.model_type, 50.0)
            for qr in candidates
        )
        if len(candidates) >= batch_size or nearly_expired:
            self._dispatch(candidates)

    def _find_critical(self) -> List[QueuedRequest]:
        """Find requests whose deadline is within ALPHA * avg_inference_time."""
        critical = []
        for mtype, queue in self.queues.items():
            avg_ms = self._avg_inf_ms.get(mtype, 50.0)
            for qr in queue:
                if qr.time_remaining_ms < self.config.ALPHA * avg_ms:
                    critical.append(qr)
        return critical

    def _compute_batch_size(self) -> int:
        """
        Adaptive batch sizing.
        Reduce batch size when ICU queue is building up to free GPU cycles faster.
        """
        icu_pressure = len(self.queues["icu"])
        if icu_pressure >= self.config.HIGH_PRESSURE_THR:
            return max(1, self.config.MAX_BATCH // 2)
        return self.config.MAX_BATCH

    def _dispatch(self, candidates: List[QueuedRequest]):
        """
        Run inference for the selected batch.
        Groups same-model-type requests together for efficient batching.
        Updates timing and signals completion events.
        """
        from system.request import ModelType

        # Group by model type
        by_type: Dict[str, List[QueuedRequest]] = defaultdict(list)
        for qr in candidates:
            by_type[qr.req.model_type].append(qr)

        for mtype, batch in by_type.items():
            # Prepare inputs
            if mtype == "nlp":
                inputs = [qr.req.input_data for qr in batch]
            else:
                inputs = [torch.tensor(qr.req.input_data).unsqueeze(0) for qr in batch]

            # Run inference
            engine = self.registry._engines[ModelType(mtype)]
            t0     = time.perf_counter()
            _, inf_ms = engine.infer(inputs)
            result_at_ms = time.perf_counter() * 1000.0

            # Update avg inference time (exponential moving average)
            self._avg_inf_ms[mtype] = 0.8 * self._avg_inf_ms[mtype] + 0.2 * inf_ms

            # Mark each request complete
            for qr in batch:
                qr.req.result_at_ms = result_at_ms
                qr.req.inference_ms = inf_ms
                self._results[qr.req.request_id] = (None, inf_ms)
                # Signal completion
                evt = self._result_events.get(qr.req.request_id)
                if evt:
                    evt.set()

            # Remove from queues
            for qr in batch:
                try:
                    self.queues[mtype].remove(qr)
                except ValueError:
                    pass


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_medserve_benchmark(
    load_level:    str   = "medium",
    n_requests:    int   = 500,
    mix:           dict  = None,
    device:        str   = "cuda",
    weights_dir:   str   = "results/weights",
    output_dir:    str   = "results/logs",
    seed:          int   = 42,
    realtime:      bool  = True,
    config:        SchedulerConfig = None,
) -> MetricsCollector:
    """
    Run MedServe SLA-aware scheduler benchmark.

    Args:
        load_level:  "low" | "medium" | "high"
        n_requests:  total requests (same as other modes for fair comparison)
        seed:        must match other modes
        config:      SchedulerConfig — tune hyperparameters here
    """
    from models.registry import ModelRegistry

    config = config or SchedulerConfig()

    print(f"\n{'='*60}")
    print(f"  BENCHMARK: MedServe  |  Load: {load_level.upper()}")
    print(f"  Requests: {n_requests}   Device: {device.upper()}")
    print(f"  Config: alpha={config.ALPHA} max_batch={config.MAX_BATCH} tick={config.TICK_MS}ms")
    print(f"{'='*60}\n")

    # Load models
    registry = ModelRegistry.default(
        weights_dir=weights_dir, device=device, use_fp16=(device=="cuda")
    )
    registry.warmup_all(n_runs=20)

    # Prime avg_inf_ms from warmup metadata
    avg_inf = {}
    for mtype_str in ["icu", "nlp", "imaging"]:
        from system.request import ModelType
        try:
            meta = registry.get_metadata(ModelType(mtype_str))
            avg_inf[mtype_str] = meta.avg_latency_ms
        except Exception:
            pass

    # Start scheduler
    scheduler = MedServeScheduler(registry=registry, config=config)
    if avg_inf:
        scheduler._avg_inf_ms.update(avg_inf)
    scheduler.start()

    # Generate workload
    gen      = WorkloadGenerator(seed=seed)
    rate     = LOAD_RATES[LoadLevel(load_level)]
    requests = gen.generate(n=n_requests, arrival_rate=rate, mix=mix)

    # Submit requests at correct arrival times
    collector = MetricsCollector(mode="medserve")
    t_start   = time.perf_counter() * 1000.0
    collector.start()

    events = {}
    print(f"  Submitting {n_requests} requests...")
    for req in requests:
        if realtime:
            target = t_start + req.arrival_offset_ms
            now    = time.perf_counter() * 1000.0
            if target > now:
                time.sleep((target - now) / 1000.0)

        req.sent_at_ms = time.perf_counter() * 1000.0
        events[req.request_id] = scheduler.submit(req)

    # Wait for all results
    print("  Waiting for all requests to complete...")
    timeout_s = 30.0
    for req in requests:
        evt = events.get(req.request_id)
        if evt:
            evt.wait(timeout=timeout_s)
        collector.record(req, batch_size=1,
                         inference_ms=req.inference_ms or req.total_latency_ms)

    collector.stop()
    scheduler.stop()

    collector.print_summary()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    collector.save_json(f"{output_dir}/medserve_{load_level}.json")
    collector.save_csv(f"{output_dir}/medserve_{load_level}.csv")

    return collector


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--load",   default="medium", choices=["low","medium","high"])
    p.add_argument("--n",      type=int, default=500)
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha",  type=float, default=1.5)
    p.add_argument("--batch",  type=int, default=16)
    args = p.parse_args()
    cfg = SchedulerConfig(ALPHA=args.alpha, MAX_BATCH=args.batch)
    run_medserve_benchmark(load_level=args.load, n_requests=args.n,
                           device=args.device, config=cfg)

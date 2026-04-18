"""
benchmarks/mode_medserve.py — MedServe SLA-aware scheduler benchmark.
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
# Scheduler hyperparameters
# ---------------------------------------------------------------------------

class SchedulerConfig:
    TICK_MS           = 3.0
    ALPHA             = 1.5      # FIXED: dispatch when time_remaining < ALPHA * avg_inf_ms
                                  # ALPHA > 1.0 means you dispatch BEFORE deadline, with headroom
    MAX_BATCH         = 12
    HIGH_PRESSURE_THR = 2
    AGING_FACTOR      = 1.6
    AGING_INTERVAL_S  = 1.0
    SAFETY_MARGIN_MS  = 10.0     # NEW: extra buffer on top of ALPHA * avg_inf_ms
    TIER_WEIGHTS: Dict = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        if self.TIER_WEIGHTS is None:
            self.TIER_WEIGHTS = {"icu": 10.0, "nlp": 2.0, "imaging": 1.0}

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Queued request wrapper
# ---------------------------------------------------------------------------

class QueuedRequest:
    def __init__(self, req: WorkloadRequest, config: SchedulerConfig):
        self.req          = req
        self.config       = config
        self.queued_at_ms = time.perf_counter() * 1000.0
        self.aging_mult   = 1.0

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

    def __init__(self, registry, config: SchedulerConfig):
        self.registry  = registry
        self.config    = config
        self.queues: Dict[str, List[QueuedRequest]] = {
            "icu": [], "nlp": [], "imaging": []
        }
        self.lock        = threading.Lock()
        self._stop_event = threading.Event()
        self._results: Dict[str, tuple]          = {}
        self._result_events: Dict[str, threading.Event] = {}
        self._last_aging = time.perf_counter()

        # CRITICAL: these must reflect real GPU inference times.
        # Primed from warmup in run_medserve_benchmark before scheduler starts.
        self._avg_inf_ms: Dict[str, float] = {
            "icu":     80.0,
            "nlp":     150.0,
            "imaging": 200.0,
        }

    def submit(self, req: WorkloadRequest) -> threading.Event:
        event = threading.Event()
        with self.lock:
            self._result_events[req.request_id] = event
            self.queues[req.model_type].append(QueuedRequest(req, self.config))
        return event

    def get_result(self, request_id: str, timeout_s: float = 2.0):
        event = self._result_events.get(request_id)
        if event:
            event.wait(timeout=timeout_s)
        return self._results.get(request_id, (None, 9999.0))

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._scheduling_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=5.0)

    # -------------------------------------------------------------------------
    # Core scheduling loop
    # -------------------------------------------------------------------------

    def _scheduling_loop(self):
        while not self._stop_event.is_set():
            loop_start = time.perf_counter()
            with self.lock:
                self._apply_aging()
                self._dispatch_one_cycle()
            elapsed_ms = (time.perf_counter() - loop_start) * 1000.0
            sleep_ms   = max(0.0, self.config.TICK_MS - elapsed_ms)
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)

    def _apply_aging(self):
        now = time.perf_counter()
        if now - self._last_aging >= self.config.AGING_INTERVAL_S:
            for queue in self.queues.values():
                for qr in queue:
                    qr.apply_aging()
            self._last_aging = now

    # def _dispatch_one_cycle(self):
    #     """
    #     Fixed dispatch logic:

    #     STEP 1 — Emergency dispatch.
    #         Threshold = ALPHA * avg_inf_ms + SAFETY_MARGIN_MS + TICK_MS
    #         With ALPHA=1.5, ICU threshold = 1.5*80 + 10 + 3 = 133ms remaining.
    #         This gives 133ms of runway for an 80ms inference — plenty of headroom.
    #         Previous bug: ALPHA=0.85 → threshold=68ms < 80ms inference → always violated.

    #     STEP 2 — Proactive dispatch for ICU even when not yet critical.
    #         If any ICU request is in queue AND scheduler is not busy with higher-urgency work,
    #         dispatch it immediately rather than waiting for it to become critical.
    #         This is the key insight: for 80ms inference + 250ms SLA, a request that arrives
    #         and waits even 1 tick (3ms) needs to be dispatched within 170ms. Don't wait.

    #     STEP 3 — Urgency-sorted batch dispatch for non-critical work.
    #     """

    #     # --- STEP 1: Emergency dispatch (deadline imminent) ---
    #     for mtype in ("icu", "nlp", "imaging"):   # priority order
    #         queue = self.queues[mtype]
    #         if not queue:
    #             continue
    #         avg_ms    = self._avg_inf_ms[mtype]
    #         # FIXED threshold: must be > avg_inf_ms to dispatch before deadline expires
    #         threshold = self.config.ALPHA * avg_ms + self.config.SAFETY_MARGIN_MS + self.config.TICK_MS
    #         critical  = [qr for qr in queue if qr.time_remaining_ms < threshold]
    #         if critical:
    #             self._dispatch(critical)
    #             return   # one dispatch per tick

    #     # --- STEP 2: Proactive ICU dispatch ---
    #     # ICU requests should NEVER sit in queue longer than needed.
    #     # As soon as there's an ICU request and the urgency score makes it the top type,
    #     # dispatch it immediately — don't wait to fill a batch.
    #     if self.queues["icu"]:
    #         batch_size = self._compute_batch_size()
    #         candidates = sorted(self.queues["icu"], key=lambda r: r.urgency, reverse=True)
    #         self._dispatch(candidates[:batch_size])
    #         return

    #     # --- STEP 3: Normal urgency-sorted batch dispatch ---
    #     batch_size = self._compute_batch_size()

    #     # Find type with highest-urgency head request
    #     best_type    = None
    #     best_urgency = -1.0
    #     for mtype, queue in self.queues.items():
    #         if not queue:
    #             continue
    #         top_urgency = max(qr.urgency for qr in queue)
    #         if top_urgency > best_urgency:
    #             best_urgency = top_urgency
    #             best_type    = mtype

    #     if best_type is None:
    #         return

    #     queue      = self.queues[best_type]
    #     candidates = sorted(queue, key=lambda r: r.urgency, reverse=True)[:batch_size]

    #     # Dispatch when batch is full OR any candidate is getting close
    #     avg_ms    = self._avg_inf_ms[best_type]
    #     threshold = self.config.ALPHA * avg_ms + self.config.SAFETY_MARGIN_MS
    #     nearly_expired = any(qr.time_remaining_ms < threshold for qr in candidates)

    #     if len(queue) >= batch_size or nearly_expired:
    #         self._dispatch(candidates)
    def _dispatch_one_cycle(self):
        """
        Fixed: deadline-aware dispatch that protects ICU without starving NLP/imaging.
        
        Key insight: dispatch the type whose head request has the LEAST remaining
        time relative to its inference cost. This naturally prioritizes ICU when
        it's urgent without monopolizing the GPU when it isn't.
        """
    
        # --- STEP 1: Emergency dispatch — any type with imminent deadline ---
        # Check all types, dispatch the most urgent one
        most_urgent_qr  = None
        most_urgent_type = None
    
        for mtype, queue in self.queues.items():
            if not queue:
                continue
            avg_ms    = self._avg_inf_ms[mtype]
            threshold = self.config.ALPHA * avg_ms + self.config.SAFETY_MARGIN_MS + self.config.TICK_MS
            for qr in queue:
                if qr.time_remaining_ms < threshold:
                    if most_urgent_qr is None or qr.urgency > most_urgent_qr.urgency:
                        most_urgent_qr   = qr
                        most_urgent_type = mtype
    
        if most_urgent_qr is not None:
            # Dispatch all critical requests of this type as a batch
            avg_ms    = self._avg_inf_ms[most_urgent_type]
            threshold = self.config.ALPHA * avg_ms + self.config.SAFETY_MARGIN_MS + self.config.TICK_MS
            critical  = [qr for qr in self.queues[most_urgent_type]
                         if qr.time_remaining_ms < threshold]
            self._dispatch(critical)
            return
    
        # --- STEP 2: Proactive dispatch based on deadline urgency ---
        # Pick the type whose head request has consumed the most of its SLA budget.
        # This is different from urgency score — it's about proportional time consumed.
        # A NLP request that has used 280ms of its 300ms SLA is more urgent than
        # an ICU request that has used 5ms of its 100ms SLA, even though ICU has
        # higher tier weight.
        
        best_type       = None
        best_consumed   = -1.0   # fraction of SLA budget consumed
    
        for mtype, queue in self.queues.items():
            if not queue:
                continue
            # Look at the head request (oldest = most time consumed)
            head_qr      = min(queue, key=lambda r: r.req.sent_at_ms)
            sla          = head_qr.req.sla_ms
            elapsed      = (time.perf_counter() * 1000.0) - head_qr.req.sent_at_ms
            # Weight by tier so ICU still gets preference when budget consumption is similar
            tier_weight  = self.config.TIER_WEIGHTS.get(mtype, 1.0)
            weighted     = (elapsed / sla) * tier_weight
    
            if weighted > best_consumed:
                best_consumed = weighted
                best_type     = mtype
    
        if best_type is None:
            return
    
        batch_size = self._compute_batch_size()
        queue      = self.queues[best_type]
        candidates = sorted(queue, key=lambda r: r.urgency, reverse=True)[:batch_size]
    
        # Dispatch if: batch is full, OR head request has used >50% of its SLA budget
        head_qr  = min(queue, key=lambda r: r.req.sent_at_ms)
        elapsed  = (time.perf_counter() * 1000.0) - head_qr.req.sent_at_ms
        budget_half_consumed = elapsed > (head_qr.req.sla_ms * 0.5)
    
        if len(queue) >= batch_size or budget_half_consumed:
            self._dispatch(candidates)

    # def _compute_batch_size(self) -> int:
    #     icu_pressure = len(self.queues["icu"])
    #     if icu_pressure >= self.config.HIGH_PRESSURE_THR:
    #         return max(1, self.config.MAX_BATCH // 2)
    #     if icu_pressure > 0:
    #         return max(2, self.config.MAX_BATCH // 4)
    #     return self.config.MAX_BATCH

    def _compute_batch_size(self) -> int:
        icu_pressure = len(self.queues["icu"])
        if icu_pressure >= self.config.HIGH_PRESSURE_THR:
            return max(1, self.config.MAX_BATCH // 2)
        # Don't reduce batch size unless ICU is actually backed up
        return self.config.MAX_BATCH

    def _dispatch(self, candidates: List[QueuedRequest]):
        from system.request import ModelType

        by_type: Dict[str, List[QueuedRequest]] = defaultdict(list)
        for qr in candidates:
            by_type[qr.req.model_type].append(qr)

        for mtype, batch in by_type.items():
            if mtype == "nlp":
                inputs = [qr.req.input_data for qr in batch]
            else:
                inputs = [torch.tensor(qr.req.input_data).unsqueeze(0) for qr in batch]

            engine = self.registry._engines[ModelType(mtype)]
            _, inf_ms    = engine.infer(inputs)
            result_at_ms = time.perf_counter() * 1000.0

            # Exponential moving average of inference time — keeps threshold calibrated
            self._avg_inf_ms[mtype] = 0.8 * self._avg_inf_ms[mtype] + 0.2 * inf_ms

            for qr in batch:
                qr.req.result_at_ms = result_at_ms
                qr.req.inference_ms = inf_ms
                self._results[qr.req.request_id] = (None, inf_ms)
                evt = self._result_events.get(qr.req.request_id)
                if evt:
                    evt.set()

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
    from models.registry import ModelRegistry

    config = config or SchedulerConfig()

    print(f"\n{'='*60}")
    print(f"  BENCHMARK: MedServe  |  Load: {load_level.upper()}")
    print(f"  Requests: {n_requests}   Device: {device.upper()}")
    print(f"  Config: alpha={config.ALPHA} max_batch={config.MAX_BATCH} tick={config.TICK_MS}ms")
    print(f"{'='*60}\n")

    registry = ModelRegistry.default(
        weights_dir=weights_dir, device=device, use_fp16=(device=="cuda")
    )
    registry.warmup_all(n_runs=20)

    # Prime avg_inf_ms from warmup BEFORE scheduler starts — critical for correct thresholds
    avg_inf = {}
    for mtype_str in ["icu", "nlp", "imaging"]:
        from system.request import ModelType
        try:
            meta = registry.get_metadata(ModelType(mtype_str))
            avg_inf[mtype_str] = meta.avg_latency_ms
            print(f"  Warmup avg latency — {mtype_str}: {meta.avg_latency_ms:.1f}ms")
        except Exception:
            pass

    scheduler = MedServeScheduler(registry=registry, config=config)
    if avg_inf:
        scheduler._avg_inf_ms.update(avg_inf)
    scheduler.start()

    gen      = WorkloadGenerator(seed=seed)
    rate     = LOAD_RATES[LoadLevel(load_level)]
    requests = gen.generate(n=n_requests, arrival_rate=rate, mix=mix)

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

        req.sent_at_ms = time.perf_counter() * 1000.0   # arrival timestamp
        events[req.request_id] = scheduler.submit(req)

    print("  Waiting for all requests to complete...")
    for req in requests:
        evt = events.get(req.request_id)
        if evt:
            evt.wait(timeout=30.0)
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
    p.add_argument("--batch",  type=int, default=12)
    args = p.parse_args()
    cfg = SchedulerConfig(ALPHA=args.alpha, MAX_BATCH=args.batch)
    run_medserve_benchmark(load_level=args.load, n_requests=args.n,
                           device=args.device, config=cfg)

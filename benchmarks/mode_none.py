"""
benchmarks/mode_none.py — Baseline: no scheduler, pure FIFO, no batching.

This is your weakest baseline — the system that "just runs models."
Every request is processed immediately and alone (batch_size=1),
in the order it arrives (FIFO), with no priority awareness.

Why this baseline matters for your paper:
    It shows the worst-case behavior. The gap between this and MedServe
    is your main result. A large gap = strong paper claim.
    A small gap = you need to stress the system harder (increase load rate).

What it measures:
    - Raw inference latency per model with no overhead
    - What happens to ICU SLA when imaging requests queue ahead of it
    - The "floor" throughput before any scheduling overhead

Run on Kaggle/Colab:
    from benchmarks.mode_none import run_no_scheduler
    results = run_no_scheduler(load_level="medium")
"""

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from typing import List, Dict, Optional
from collections import deque
import threading

from benchmarks.workload import WorkloadGenerator, WorkloadRequest, LoadLevel, LOAD_RATES, DEFAULT_MIX
from benchmarks.metrics  import MetricsCollector


# ---------------------------------------------------------------------------
# FIFO inference runner
# ---------------------------------------------------------------------------

class FIFORunner:
    """
    Pure FIFO, no batching, no priority.
    Requests are processed in arrival order, one at a time.
    """

    def __init__(self, registry, device: str = "cuda"):
        self.registry = registry
        self.device   = device
        self.queue    = deque()
        self.lock     = threading.Lock()

    def _run_single(self, req: WorkloadRequest) -> float:
        """Run inference for one request. Returns inference_ms."""
        import torch
        t0 = time.perf_counter()

        if req.model_type == "nlp":
            inputs = [req.input_data]
        else:
            tensor = torch.tensor(req.input_data).unsqueeze(0)   # add batch dim
            inputs = [tensor]

        engine = self.registry._engines[
            __import__('system.request', fromlist=['ModelType']).ModelType(req.model_type)
        ]
        _, latency = engine.infer(inputs)
        return latency

    def process_stream(
        self,
        requests:   List[WorkloadRequest],
        collector:  MetricsCollector,
        realtime:   bool = True,
    ):
        """
        Process all requests in FIFO order.
        With realtime=True, respects inter-arrival timing (realistic).
        With realtime=False, runs as fast as possible (throughput test).
        """
        t_start = time.perf_counter() * 1000.0
        collector.start()

        for req in requests:
            # Sleep until this request's scheduled arrival
            if realtime:
                target = t_start + req.arrival_offset_ms
                now    = time.perf_counter() * 1000.0
                if target > now:
                    time.sleep((target - now) / 1000.0)

            req.sent_at_ms = time.perf_counter() * 1000.0

            # Inference — single request, no batching
            inf_ms = self._run_single(req)

            req.result_at_ms = time.perf_counter() * 1000.0
            req.inference_ms = inf_ms

            collector.record(req, batch_size=1, inference_ms=inf_ms)

        collector.stop()


# ---------------------------------------------------------------------------
# Static batching baseline
# ---------------------------------------------------------------------------

class StaticBatchRunner:
    """
    Static batching: wait until N same-type requests are queued, then dispatch.
    No priority. FIFO within each model type's queue.
    This is your second baseline — better than pure FIFO but no SLA awareness.
    """

    def __init__(self, registry, batch_size: int = 8, device: str = "cuda"):
        self.registry   = registry
        self.batch_size = batch_size
        self.device     = device

    def process_stream(
        self,
        requests:  List[WorkloadRequest],
        collector: MetricsCollector,
        realtime:  bool = True,
    ):
        from system.request import ModelType

        # Buffer requests by model type
        buffers: Dict[str, List[WorkloadRequest]] = {"icu":[], "nlp":[], "imaging":[]}
        t_start = time.perf_counter() * 1000.0
        collector.start()

        def flush(mtype: str):
            buf = buffers[mtype]
            if not buf: return
            # Stack inputs
            if mtype == "nlp":
                inputs = [r.input_data for r in buf]
            else:
                inputs = [torch.tensor(r.input_data).unsqueeze(0) for r in buf]

            engine = self.registry._engines[ModelType(mtype)]
            _, inf_ms = engine.infer(inputs)

            result_at = time.perf_counter() * 1000.0
            for r in buf:
                r.result_at_ms = result_at
                r.inference_ms = inf_ms
                collector.record(r, batch_size=len(buf), inference_ms=inf_ms)
            buffers[mtype].clear()

        for req in requests:
            if realtime:
                target = t_start + req.arrival_offset_ms
                now    = time.perf_counter() * 1000.0
                if target > now:
                    time.sleep((target - now) / 1000.0)

            req.sent_at_ms = time.perf_counter() * 1000.0
            buffers[req.model_type].append(req)

            # Dispatch when buffer is full
            if len(buffers[req.model_type]) >= self.batch_size:
                flush(req.model_type)

        # Flush remaining
        for mtype in buffers:
            flush(mtype)

        collector.stop()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_no_scheduler(
    load_level:    str   = "medium",
    n_requests:    int   = 500,
    mix:           dict  = None,
    device:        str   = "cuda",
    weights_dir:   str   = "results/weights",
    output_dir:    str   = "results/logs",
    seed:          int   = 42,
    realtime:      bool  = True,
) -> Dict[str, MetricsCollector]:
    """
    Run both no-scheduler baselines (FIFO and static batching).

    Args:
        load_level:  "low" | "medium" | "high"
        n_requests:  total requests to generate
        mix:         workload mix dict (default: DEFAULT_MIX)
        device:      "cuda" or "cpu"
        weights_dir: path to trained .pt weight files
        output_dir:  where to save results JSON/CSV
        seed:        random seed (must match other modes for fair comparison)
        realtime:    respect inter-arrival timing (True for realistic test)

    Returns:
        Dict with keys "fifo" and "static" → MetricsCollector
    """
    from models.registry import ModelRegistry
    from benchmarks.workload import LOAD_RATES, LoadLevel

    print(f"\n{'='*60}")
    print(f"  BENCHMARK: No Scheduler  |  Load: {load_level.upper()}")
    print(f"  Requests: {n_requests}   Device: {device.upper()}")
    print(f"{'='*60}\n")

    # Load models
    registry = ModelRegistry.default(
        weights_dir=weights_dir, device=device,
        use_fp16=(device == "cuda"),
    )
    registry.warmup_all(n_runs=20)

    # Generate workload (same seed = reproducible)
    gen       = WorkloadGenerator(seed=seed)
    rate      = LOAD_RATES[LoadLevel(load_level)]
    requests  = gen.generate(n=n_requests, arrival_rate=rate, mix=mix)

    results = {}

    # --- Mode 1a: Pure FIFO ---
    print("  Running FIFO baseline...")
    import copy
    fifo_requests = copy.deepcopy(requests)
    fifo_runner   = FIFORunner(registry=registry, device=device)
    fifo_coll     = MetricsCollector(mode="fifo")
    fifo_runner.process_stream(fifo_requests, fifo_coll, realtime=realtime)
    fifo_coll.print_summary()
    fifo_coll.save_json(f"{output_dir}/fifo_{load_level}.json")
    fifo_coll.save_csv(f"{output_dir}/fifo_{load_level}.csv")
    results["fifo"] = fifo_coll

    # --- Mode 1b: Static batching ---
    print("  Running static batching baseline (batch_size=8)...")
    static_requests = copy.deepcopy(requests)
    static_runner   = StaticBatchRunner(registry=registry, batch_size=8, device=device)
    static_coll     = MetricsCollector(mode="static_batch")
    static_runner.process_stream(static_requests, static_coll, realtime=realtime)
    static_coll.print_summary()
    static_coll.save_json(f"{output_dir}/static_batch_{load_level}.json")
    static_coll.save_csv(f"{output_dir}/static_batch_{load_level}.csv")
    results["static"] = static_coll

    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--load",    default="medium", choices=["low","medium","high"])
    p.add_argument("--n",       type=int, default=500)
    p.add_argument("--device",  default="cuda")
    p.add_argument("--no-realtime", action="store_true")
    args = p.parse_args()
    run_no_scheduler(load_level=args.load, n_requests=args.n,
                     device=args.device, realtime=not args.no_realtime)

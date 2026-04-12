"""
benchmarks/metrics.py — Unified metrics for all benchmark modes.
Every mode uses this class so results are directly comparable.
"""

import csv, json, statistics, time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional
from benchmarks.workload import WorkloadRequest, SLA_MS


@dataclass
class RequestRecord:
    request_id: str; model_type: str; priority: str
    arrival_offset_ms: float; total_latency_ms: float
    inference_ms: float; sla_ms: float; sla_violated: bool
    batch_size: int = 1; scheduler_mode: str = ""


@dataclass
class LatencyStats:
    model_type: str; scheduler_mode: str; n_requests: int
    p50_ms: float; p95_ms: float; p99_ms: float
    avg_ms: float; min_ms: float; max_ms: float
    sla_ms: float; sla_violations: int; sla_violation_pct: float
    throughput_rps: float; avg_batch_size: float
    def to_row(self): return asdict(self)


class MetricsCollector:
    PRIORITY_MAP = {"icu": "HIGH", "nlp": "MID", "imaging": "LOW"}

    def __init__(self, mode: str):
        self.mode = mode
        self._records: List[RequestRecord] = []
        self._t_start = self._t_end = None

    def start(self): self._t_start = time.perf_counter()
    def stop(self):  self._t_end   = time.perf_counter()

    def record(self, req: WorkloadRequest, batch_size: int = 1, inference_ms: float = None):
        if req.total_latency_ms is None: return
        self._records.append(RequestRecord(
            request_id=req.request_id, model_type=req.model_type,
            priority=self.PRIORITY_MAP.get(req.model_type, "UNKNOWN"),
            arrival_offset_ms=req.arrival_offset_ms,
            total_latency_ms=req.total_latency_ms,
            inference_ms=inference_ms or req.inference_ms or req.total_latency_ms,
            sla_ms=req.sla_ms, sla_violated=bool(req.sla_violated),
            batch_size=batch_size, scheduler_mode=self.mode,
        ))

    def summarize(self) -> Dict[str, LatencyStats]:
        if not self._records: return {}
        elapsed  = (self._t_end or time.perf_counter()) - (self._t_start or 0)
        by_type  = defaultdict(list)
        for r in self._records:
            by_type[r.model_type].append(r)
        by_type["overall"] = self._records
        out = {}
        for mtype, recs in by_type.items():
            lats = sorted(r.total_latency_ms for r in recs)
            n    = len(lats)
            viol = sum(1 for r in recs if r.sla_violated)
            def pct(p): return lats[max(0, int(p*n)-1)]
            out[mtype] = LatencyStats(
                model_type=mtype, scheduler_mode=self.mode, n_requests=n,
                p50_ms=round(pct(.5),2), p95_ms=round(pct(.95),2), p99_ms=round(pct(.99),2),
                avg_ms=round(statistics.mean(lats),2), min_ms=round(lats[0],2), max_ms=round(lats[-1],2),
                sla_ms=SLA_MS.get(mtype,500) if mtype!="overall" else 0,
                sla_violations=viol, sla_violation_pct=round(100*viol/n,2),
                throughput_rps=round(n/max(elapsed,0.001),2),
                avg_batch_size=round(statistics.mean(r.batch_size for r in recs),2),
            )
        return out

    def print_summary(self):
        stats = self.summarize()
        print(f"\n{'='*68}\n  Mode: {self.mode.upper()}  |  N={len(self._records)}\n{'='*68}")
        print(f"  {'Model':<10} {'N':>5} {'P50':>8} {'P95':>8} {'P99':>8} {'SLA%viol':>9} {'RPS':>8}")
        print(f"  {'-'*60}")
        for mt in ["icu","nlp","imaging","overall"]:
            s = stats.get(mt); 
            if not s: continue
            flag = " !" if s.sla_violation_pct > 5 else ""
            print(f"  {mt:<10} {s.n_requests:>5} {s.p50_ms:>7.1f}ms {s.p95_ms:>7.1f}ms "
                  f"{s.p99_ms:>7.1f}ms {s.sla_violation_pct:>8.1f}% {s.throughput_rps:>7.1f}{flag}")
        print()

    def save_json(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path,"w") as f:
            json.dump({"mode":self.mode,
                       "summary":{k:v.to_row() for k,v in self.summarize().items()},
                       "records":[asdict(r) for r in self._records]}, f, indent=2)
        print(f"  Saved: {path}")

    def save_csv(self, path: str):
        if not self._records: return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path,"w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=asdict(self._records[0]).keys())
            w.writeheader(); w.writerows(asdict(r) for r in self._records)
        print(f"  Saved: {path}")

    @staticmethod
    def load_json(path: str) -> "MetricsCollector":
        with open(path) as f: data = json.load(f)
        c = MetricsCollector(mode=data["mode"])
        for r in data["records"]: c._records.append(RequestRecord(**r))
        return c

    @staticmethod
    def compare_table(paths: List[str]):
        """Print side-by-side from saved JSON files."""
        print(f"\n{'='*80}\n  COMPARISON TABLE\n{'='*80}")
        print(f"  {'Mode':<14} {'Model':<10} {'P50':>8} {'P99':>8} {'SLA%':>8} {'RPS':>10}")
        print(f"  {'-'*60}")
        for p in paths:
            c = MetricsCollector.load_json(p)
            for mt in ["icu","nlp","imaging"]:
                s = c.summarize().get(mt)
                if s:
                    print(f"  {c.mode:<14} {mt:<10} {s.p50_ms:>7.1f}ms "
                          f"{s.p99_ms:>7.1f}ms {s.sla_violation_pct:>7.1f}% {s.throughput_rps:>9.1f}")
        print()

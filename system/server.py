"""
system/server.py — Local FastAPI inference server for MedServe.

This runs entirely on your laptop (CPU).
The server loads trained weights from results/weights/ (synced from GitHub).
Use this for:
  - Integration testing your scheduler and batching logic
  - Latency measurement under simulated load
  - API contract testing before deploying to GPU

Start the server:
    python system/server.py

Then test it:
    curl http://localhost:8000/health
    python experiments/request_generator.py   # sends test load

Endpoints:
    GET  /health              — liveness check + registry metadata
    POST /infer               — single inference request
    POST /infer/batch         — batch of requests
    GET  /metrics             — runtime latency/throughput stats
    POST /registry/swap       — hot-swap a model (for ablation studies)
"""

import sys
import time
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path (for running as script)
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from system.request import Request, ModelType, Priority, SLA_DEADLINES_MS
from models.registry import ModelRegistry


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MedServe",
    description="Latency-aware inference engine for heterogeneous healthcare AI workloads",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# Global registry — loaded at startup
registry: Optional[ModelRegistry] = None

# Runtime stats
_request_log: List[dict] = []   # in-memory log (use CSV for experiments)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class InferRequest(BaseModel):
    model_type: str         # "icu", "imaging", "nlp"
    input:      Any         # raw input: list of floats, nested list, or string
    request_id: Optional[str] = None

class InferResponse(BaseModel):
    request_id:        str
    model_type:        str
    result:            List[float]   # flattened output probabilities
    inference_time_ms: float
    total_latency_ms:  float
    sla_ms:            float
    sla_violated:      bool
    priority:          str

class BatchInferRequest(BaseModel):
    requests: List[InferRequest]

class SwapRequest(BaseModel):
    model_type:  str    # "icu", "imaging", "nlp"
    model_class: str    # e.g. "ICUInferenceEngine"
    module:      str    # e.g. "icu_model"
    model_path:  Optional[str] = None


# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    global registry
    print("\n[MedServe] Starting up...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[MedServe] Device: {device.upper()}")

    weights_dir = Path("results/weights")
    if not weights_dir.exists():
        print(f"[MedServe] WARNING: {weights_dir} not found.")
        print("[MedServe] Run: python scripts/sync_weights.py")
        print("[MedServe] Loading with random/pretrained weights for testing.")

    registry = ModelRegistry.default(
        weights_dir=str(weights_dir),
        device=device,
        use_fp16=(device == "cuda"),
    )
    registry.warmup_all(n_runs=10)
    print(f"[MedServe] Ready. Registered: {registry.registered_types()}\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_model_type(s: str) -> ModelType:
    try:
        return ModelType(s.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model_type '{s}'. Valid: {[t.value for t in ModelType]}"
        )


def build_input_tensor(model_type: ModelType, raw: Any) -> Any:
    """
    Convert JSON input to the format each engine expects.

    ICU:     [[float]*34]*48  →  tensor (1, 48, 34)
    Imaging: [[float]*224]*...  → tensor (1, 3, 224, 224)  OR filepath string
    NLP:     "some text"      →  string (passed directly)
    """
    if model_type == ModelType.NLP:
        if not isinstance(raw, str):
            raise HTTPException(400, "NLP input must be a string (clinical text)")
        return raw

    try:
        t = torch.tensor(raw, dtype=torch.float32)
    except Exception as e:
        raise HTTPException(400, f"Could not parse input as numeric tensor: {e}")

    # Add batch dimension if missing
    expected_dims = {
        ModelType.ICU:     2,   # (48, 34)
        ModelType.IMAGING: 3,   # (3, 224, 224)
    }
    if t.dim() == expected_dims.get(model_type, 0):
        t = t.unsqueeze(0)

    return t


def log_request(req: Request):
    _request_log.append({
        "request_id":        req.request_id,
        "model_type":        req.model_type.value,
        "priority":          req.priority.name,
        "arrival_time_ms":   req.arrival_time_ms,
        "inference_time_ms": req.inference_time_ms,
        "total_latency_ms":  req.total_latency_ms,
        "sla_ms":            SLA_DEADLINES_MS[req.priority],
        "sla_violated":      req.sla_violated,
    })


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Liveness check. Returns registry metadata and device info."""
    return {
        "status":   "ok",
        "device":   str(next(iter(registry._engines.values())).device)
                    if registry else "unloaded",
        "models":   registry.all_metadata() if registry else {},
        "requests_served": sum(registry._call_counts.values()) if registry else 0,
    }


@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    """
    Single inference request.

    Example (ICU):
        curl -X POST http://localhost:8000/infer \\
             -H 'Content-Type: application/json' \\
             -d '{"model_type": "icu", "input": <48x34 array>}'

    Example (NLP):
        curl -X POST http://localhost:8000/infer \\
             -H 'Content-Type: application/json' \\
             -d '{"model_type": "nlp", "input": "Patient is stable."}'
    """
    model_type = parse_model_type(req.model_type)
    tensor     = build_input_tensor(model_type, req.input)

    request = Request(
        model_type=model_type,
        input_tensor=tensor,
        **({"request_id": req.request_id} if req.request_id else {}),
    )

    try:
        result, latency_ms = registry.infer(request)
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")

    log_request(request)

    return InferResponse(
        request_id        = request.request_id,
        model_type        = request.model_type.value,
        result            = result.flatten().tolist(),
        inference_time_ms = round(request.inference_time_ms, 2),
        total_latency_ms  = round(request.total_latency_ms, 2),
        sla_ms            = SLA_DEADLINES_MS[request.priority],
        sla_violated      = request.sla_violated,
        priority          = request.priority.name,
    )


@app.post("/infer/batch")
async def infer_batch(req: BatchInferRequest):
    """
    Batch inference. All requests are dispatched together.
    Mixed model types are handled correctly (grouped internally).
    """
    requests = []
    for r in req.requests:
        mtype  = parse_model_type(r.model_type)
        tensor = build_input_tensor(mtype, r.input)
        requests.append(Request(model_type=mtype, input_tensor=tensor))

    try:
        registry.infer_batch(requests)
    except Exception as e:
        raise HTTPException(500, f"Batch inference failed: {e}")

    for r in requests:
        log_request(r)

    return {
        "results": [
            {
                "request_id":       r.request_id,
                "model_type":       r.model_type.value,
                "result":           r.result.flatten().tolist(),
                "total_latency_ms": round(r.total_latency_ms, 2),
                "sla_violated":     r.sla_violated,
            }
            for r in requests
        ]
    }


@app.get("/metrics")
async def metrics():
    """
    Runtime latency and throughput statistics.
    This is your live dashboard during load testing.
    """
    if not _request_log:
        return {"message": "No requests served yet."}

    import statistics

    by_type = {}
    for entry in _request_log:
        mt = entry["model_type"]
        by_type.setdefault(mt, []).append(entry)

    summary = {}
    for mt, entries in by_type.items():
        latencies = sorted(e["total_latency_ms"] for e in entries)
        violated  = sum(1 for e in entries if e["sla_violated"])
        n         = len(entries)
        summary[mt] = {
            "requests":        n,
            "sla_violations":  violated,
            "sla_violation_%": round(100 * violated / n, 1),
            "p50_ms":          round(latencies[int(0.50 * n)], 2),
            "p95_ms":          round(latencies[int(0.95 * n)], 2),
            "p99_ms":          round(latencies[int(0.99 * n)], 2),
            "avg_ms":          round(statistics.mean(latencies), 2),
        }

    return {
        "total_requests": len(_request_log),
        "by_model_type":  summary,
    }


@app.get("/metrics/export")
async def export_metrics():
    """Export all logged requests as JSON (for saving to CSV in experiments)."""
    return {"log": _request_log}


@app.post("/registry/swap")
async def swap_model(req: SwapRequest):
    """
    Hot-swap a model at runtime without restarting the server.
    Used for ablation studies — swap ICU LSTM for ICU Transformer, etc.

    Example:
        curl -X POST http://localhost:8000/registry/swap \\
             -d '{"model_type":"icu","model_class":"ICUTransformerEngine",
                  "module":"icu_transformer_model","model_path":"results/weights/icu_v2.pt"}'
    """
    import importlib
    mtype  = parse_model_type(req.model_type)

    try:
        module = importlib.import_module(f"models.{req.module}")
        cls    = getattr(module, req.model_class)
        engine = cls(
            model_path=req.model_path,
            device=str(next(iter(registry._engines.values())).device),
            use_fp16=False,
        )
        registry.swap(mtype, engine)
        engine.warmup()
    except Exception as e:
        raise HTTPException(500, f"Swap failed: {e}")

    return {
        "status":      "swapped",
        "model_type":  mtype.value,
        "new_engine":  engine.metadata.model_name,
    }


@app.delete("/metrics/reset")
async def reset_metrics():
    """Clear the request log. Use between experiments."""
    _request_log.clear()
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "system.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,   # set True during development
        log_level="info",
    )

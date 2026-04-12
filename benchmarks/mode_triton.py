"""
benchmarks/mode_triton.py — Triton Inference Server benchmark.

What Triton gives you:
    - Dynamic batching (batch requests automatically)
    - Concurrent model execution (multiple models on same GPU)
    - Priority queues (integer 0-9, not clinical semantics)
    - gRPC/HTTP REST endpoints

What Triton DOES NOT give you (= your paper's gap):
    - Clinical priority tiers (ICU vs imaging is just "priority 9 vs 5")
    - SLA deadline awareness (no concept of "this request must complete in 100ms")
    - Workload-adaptive batch sizing (fixed max_batch_size per model)
    - Aging / starvation prevention for clinical workloads

This file:
    1. Exports your trained PyTorch models to ONNX (Triton's preferred format)
    2. Builds the Triton model repository layout
    3. Starts Triton server via Docker
    4. Sends the same workload via HTTP client
    5. Collects metrics using the same MetricsCollector

IMPORTANT — What you need:
    - Docker installed (free)
    - nvcr.io/nvidia/tritonserver:23.10-py3 image (~8GB, pull once)
    - OR run on Kaggle/Colab which has Docker available

Run on Kaggle/Colab:
    from benchmarks.mode_triton import run_triton_benchmark
    results = run_triton_benchmark(load_level="medium")
"""

import json
import os
import subprocess
import sys
import time
import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from benchmarks.workload import WorkloadGenerator, WorkloadRequest, LoadLevel, LOAD_RATES, DEFAULT_MIX
from benchmarks.metrics  import MetricsCollector


# ---------------------------------------------------------------------------
# Step 1: Export models to ONNX
# ---------------------------------------------------------------------------

def export_to_onnx(weights_dir: str, onnx_dir: str, device: str = "cpu"):
    """
    Export all three PyTorch models to ONNX format for Triton.
    ONNX is Triton's preferred backend (faster than PyTorch backend).
    """
    from models.registry import ModelRegistry
    from system.request  import ModelType

    print("Exporting models to ONNX...")
    Path(onnx_dir).mkdir(parents=True, exist_ok=True)

    registry = ModelRegistry.default(weights_dir=weights_dir, device=device, use_fp16=False)

    export_configs = {
        "icu": {
            "dummy": torch.randn(1, 48, 34),
            "input_names":  ["vitals"],
            "output_names": ["sepsis_prob"],
            "dynamic_axes": {"vitals": {0: "batch"}, "sepsis_prob": {0: "batch"}},
            "path": f"{onnx_dir}/icu_model.onnx",
        },
        "imaging": {
            "dummy": torch.randn(1, 3, 224, 224),
            "input_names":  ["image"],
            "output_names": ["pathology_probs"],
            "dynamic_axes": {"image": {0: "batch"}, "pathology_probs": {0: "batch"}},
            "path": f"{onnx_dir}/imaging_model.onnx",
        },
    }

    for mtype_str, cfg in export_configs.items():
        mtype  = ModelType(mtype_str)
        engine = registry._engines[mtype]
        model  = engine.model.eval().cpu()

        torch.onnx.export(
            model, cfg["dummy"],
            cfg["path"],
            input_names=cfg["input_names"],
            output_names=cfg["output_names"],
            dynamic_axes=cfg["dynamic_axes"],
            opset_version=17,
            do_constant_folding=True,
        )
        size_mb = Path(cfg["path"]).stat().st_size / 1e6
        print(f"  Exported {mtype_str}: {cfg['path']} ({size_mb:.1f} MB)")

    # NLP: export via transformers ONNX export (different path)
    try:
        from transformers.onnx import export as hf_onnx_export
        print("  NLP ONNX export: use `optimum-cli export onnx` for production")
        print("  For benchmark: NLP uses PyTorch backend in Triton (acceptable)")
    except ImportError:
        pass

    print("ONNX export complete.\n")


# ---------------------------------------------------------------------------
# Step 2: Build Triton model repository
# ---------------------------------------------------------------------------

def build_triton_repo(
    onnx_dir:   str,
    repo_dir:   str,
    max_batch:  int = 16,
    gpu_count:  int = 1,
):
    """
    Create the Triton model repository layout:

    triton_repo/
        icu_model/
            config.pbtxt          ← model configuration
            1/
                model.onnx        ← weights
        imaging_model/
            config.pbtxt
            1/
                model.onnx
        nlp_model/
            config.pbtxt
            1/
                model.pt          ← PyTorch backend for NLP
    """
    print("Building Triton model repository...")

    configs = {
        "icu_model": {
            "backend": "onnxruntime",
            "onnx_src": f"{onnx_dir}/icu_model.onnx",
            "max_batch": max_batch,
            "input":  [("vitals",       "TYPE_FP32", [48, 34])],
            "output": [("sepsis_prob",   "TYPE_FP32", [1])],
            "priority": 9,    # highest in Triton (0-9 scale)
            "latency_budget_us": 80000,   # 80ms — Triton's batching window
        },
        "imaging_model": {
            "backend": "onnxruntime",
            "onnx_src": f"{onnx_dir}/imaging_model.onnx",
            "max_batch": max_batch,
            "input":  [("image",          "TYPE_FP32", [3, 224, 224])],
            "output": [("pathology_probs", "TYPE_FP32", [14])],
            "priority": 3,
            "latency_budget_us": 400000,  # 400ms
        },
        "nlp_model": {
            "backend": "pytorch",         # TorchScript backend for NLP
            "onnx_src": None,             # uses .pt file instead
            "max_batch": 8,
            "input":  [("input_ids",      "TYPE_INT64", [-1]),
                       ("attention_mask", "TYPE_INT64", [-1])],
            "output": [("class_probs",    "TYPE_FP32", [-1])],
            "priority": 6,
            "latency_budget_us": 250000,  # 250ms
        },
    }

    for model_name, cfg in configs.items():
        model_dir = Path(repo_dir) / model_name / "1"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Copy weights
        if cfg["onnx_src"] and Path(cfg["onnx_src"]).exists():
            import shutil
            shutil.copy(cfg["onnx_src"], model_dir / "model.onnx")

        # Write config.pbtxt
        config = _make_pbtxt(model_name, cfg, gpu_count)
        (Path(repo_dir) / model_name / "config.pbtxt").write_text(config)
        print(f"  Created: {repo_dir}/{model_name}/")

    print("Triton repository ready.\n")
    return repo_dir


def _make_pbtxt(name: str, cfg: dict, gpu_count: int) -> str:
    """Generate Triton config.pbtxt for a model."""

    def fmt_dims(dims):
        return "[" + ", ".join(str(d) for d in dims) + "]"

    inputs  = "\n".join(
        f'input [{{\n  name: "{n}"\n  data_type: {dt}\n  dims: {fmt_dims(d)}\n}}]'
        for n, dt, d in cfg["input"]
    )
    outputs = "\n".join(
        f'output [{{\n  name: "{n}"\n  data_type: {dt}\n  dims: {fmt_dims(d)}\n}}]'
        for n, dt, d in cfg["output"]
    )

    return f"""name: "{name}"
backend: "{cfg['backend']}"
max_batch_size: {cfg['max_batch']}

{inputs}

{outputs}

dynamic_batching {{
  preferred_batch_size: [ 1, 2, 4, 8 ]
  max_queue_delay_microseconds: {cfg['latency_budget_us'] // 4}
  priority_queue_policy {{
    default_priority_level: {cfg['priority']}
  }}
}}

instance_group [
  {{
    count: 1
    kind: {'KIND_GPU' if gpu_count > 0 else 'KIND_CPU'}
  }}
]

optimization {{
  execution_accelerators {{
    gpu_execution_accelerator: [{{
      name: "tensorrt"
      parameters {{ key: "precision_mode" value: "FP16" }}
    }}]
  }}
}}
"""


# ---------------------------------------------------------------------------
# Step 3: Start / stop Triton server via Docker
# ---------------------------------------------------------------------------

def start_triton(repo_dir: str, http_port: int = 8000, gpu: bool = True) -> subprocess.Popen:
    """Start Triton Inference Server via Docker. Returns the process."""
    gpu_flag   = "--gpus all" if gpu else ""
    image      = "nvcr.io/nvidia/tritonserver:23.10-py3"
    abs_repo   = str(Path(repo_dir).resolve())

    cmd = (
        f"docker run --rm -d {gpu_flag} "
        f"-p {http_port}:8000 -p 8001:8001 -p 8002:8002 "
        f"-v {abs_repo}:/models "
        f"{image} "
        f"tritonserver --model-repository=/models "
        f"--log-verbose=0"
    )
    print(f"Starting Triton server...\n  {cmd}\n")
    proc = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for server to be ready
    print("  Waiting for Triton to be ready (up to 60s)...")
    for i in range(60):
        time.sleep(1)
        try:
            import urllib.request
            urllib.request.urlopen(f"http://localhost:{http_port}/v2/health/ready", timeout=2)
            print(f"  Triton ready after {i+1}s")
            return proc
        except Exception:
            pass

    print("  WARNING: Triton did not become ready in 60s. Continuing anyway.")
    return proc


def stop_triton(proc: subprocess.Popen):
    """Stop the Triton Docker container."""
    proc.terminate()
    subprocess.run("docker stop $(docker ps -q --filter ancestor=nvcr.io/nvidia/tritonserver:23.10-py3)",
                   shell=True, capture_output=True)
    print("Triton server stopped.")


# ---------------------------------------------------------------------------
# Step 4: Send requests to Triton via HTTP
# ---------------------------------------------------------------------------

class TritonHTTPClient:
    """Minimal HTTP client for Triton inference endpoint."""

    def __init__(self, url: str = "http://localhost:8000"):
        self.url = url
        try:
            import tritonclient.http as triton_http
            self.client = triton_http.InferenceServerClient(url=url.replace("http://", ""))
            self.use_sdk = True
        except ImportError:
            self.use_sdk = False
            print("  tritonclient not installed — using raw HTTP (pip install tritonclient[http])")

    def infer(self, req: WorkloadRequest) -> float:
        """Send one request to Triton. Returns inference_ms."""
        t0 = time.perf_counter()

        if self.use_sdk:
            self._infer_sdk(req)
        else:
            self._infer_http(req)

        return (time.perf_counter() - t0) * 1000.0

    def _infer_sdk(self, req: WorkloadRequest):
        import tritonclient.http as triton_http
        import numpy as np

        model_map = {"icu": "icu_model", "imaging": "imaging_model", "nlp": "nlp_model"}
        mname     = model_map[req.model_type]

        if req.model_type == "icu":
            data   = np.array(req.input_data, dtype=np.float32)[np.newaxis]
            inputs = [triton_http.InferInput("vitals", data.shape, "FP32")]
            inputs[0].set_data_from_numpy(data)
        elif req.model_type == "imaging":
            data   = np.array(req.input_data, dtype=np.float32)[np.newaxis]
            inputs = [triton_http.InferInput("image", data.shape, "FP32")]
            inputs[0].set_data_from_numpy(data)
        else:
            # NLP: tokenize first, send input_ids + attention_mask
            from transformers import DistilBertTokenizer
            tok    = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            enc    = tok(req.input_data, return_tensors="np", padding=True, truncation=True)
            inputs = [
                triton_http.InferInput("input_ids",      enc["input_ids"].shape,      "INT64"),
                triton_http.InferInput("attention_mask", enc["attention_mask"].shape,  "INT64"),
            ]
            inputs[0].set_data_from_numpy(enc["input_ids"].astype(np.int64))
            inputs[1].set_data_from_numpy(enc["attention_mask"].astype(np.int64))

        self.client.infer(model_name=mname, inputs=inputs)

    def _infer_http(self, req: WorkloadRequest):
        """Fallback: raw HTTP JSON inference."""
        import urllib.request, json, numpy as np

        model_map = {"icu": "icu_model", "imaging": "imaging_model", "nlp": "nlp_model"}
        mname     = model_map[req.model_type]

        if req.model_type in ("icu", "imaging"):
            data    = np.array(req.input_data, dtype=np.float32)
            payload = json.dumps({
                "inputs": [{"name": "vitals" if req.model_type=="icu" else "image",
                             "shape": [1] + list(data.shape),
                             "datatype": "FP32",
                             "data": data.flatten().tolist()}]
            }).encode()
            url = f"{self.url}/v2/models/{mname}/infer"
            urllib.request.urlopen(urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"}
            ), timeout=5)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_triton_benchmark(
    load_level:    str  = "medium",
    n_requests:    int  = 500,
    mix:           dict = None,
    weights_dir:   str  = "results/weights",
    onnx_dir:      str  = "results/triton_onnx",
    repo_dir:      str  = "results/triton_repo",
    output_dir:    str  = "results/logs",
    http_port:     int  = 8000,
    device:        str  = "cuda",
    seed:          int  = 42,
    realtime:      bool = True,
    skip_export:   bool = False,   # True if ONNX already exported
    skip_docker:   bool = False,   # True if Triton already running
) -> MetricsCollector:
    """
    Full Triton benchmark pipeline:
    export → build repo → start server → benchmark → stop server → save results

    Args:
        skip_export:  set True to reuse previously exported ONNX files
        skip_docker:  set True if Triton is already running externally
    """
    print(f"\n{'='*60}")
    print(f"  BENCHMARK: Triton  |  Load: {load_level.upper()}")
    print(f"  Requests: {n_requests}   Device: {device.upper()}")
    print(f"{'='*60}\n")

    # Export models
    if not skip_export:
        export_to_onnx(weights_dir=weights_dir, onnx_dir=onnx_dir, device="cpu")
        build_triton_repo(onnx_dir=onnx_dir, repo_dir=repo_dir,
                          gpu_count=1 if device=="cuda" else 0)

    # Start Triton
    triton_proc = None
    if not skip_docker:
        triton_proc = start_triton(repo_dir=repo_dir, http_port=http_port,
                                    gpu=(device=="cuda"))

    # Generate workload (same seed as other modes)
    gen      = WorkloadGenerator(seed=seed)
    rate     = LOAD_RATES[LoadLevel(load_level)]
    requests = gen.generate(n=n_requests, arrival_rate=rate, mix=mix)

    # Run benchmark
    client    = TritonHTTPClient(url=f"http://localhost:{http_port}")
    collector = MetricsCollector(mode="triton")
    t_start   = time.perf_counter() * 1000.0
    collector.start()

    print(f"  Sending {n_requests} requests to Triton...")
    for req in requests:
        if realtime:
            target = t_start + req.arrival_offset_ms
            now    = time.perf_counter() * 1000.0
            if target > now:
                time.sleep((target - now) / 1000.0)

        req.sent_at_ms = time.perf_counter() * 1000.0
        try:
            inf_ms = client.infer(req)
            req.inference_ms = inf_ms
        except Exception as e:
            inf_ms = 9999.0   # mark as failed
            req.inference_ms = inf_ms
        req.result_at_ms = time.perf_counter() * 1000.0
        collector.record(req, batch_size=1, inference_ms=inf_ms)

    collector.stop()
    collector.print_summary()

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    collector.save_json(f"{output_dir}/triton_{load_level}.json")
    collector.save_csv(f"{output_dir}/triton_{load_level}.csv")

    # Stop Triton
    if triton_proc:
        stop_triton(triton_proc)

    return collector


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--load",         default="medium", choices=["low","medium","high"])
    p.add_argument("--n",            type=int, default=500)
    p.add_argument("--device",       default="cuda")
    p.add_argument("--skip-export",  action="store_true")
    p.add_argument("--skip-docker",  action="store_true")
    args = p.parse_args()
    run_triton_benchmark(load_level=args.load, n_requests=args.n,
                         device=args.device, skip_export=args.skip_export,
                         skip_docker=args.skip_docker)

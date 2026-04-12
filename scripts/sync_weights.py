#!/usr/bin/env python3
"""
scripts/sync_weights.py — Pull trained model weights from GitHub to local machine.

Run this after your Kaggle notebook finishes training:
    python scripts/sync_weights.py

What it does:
    1. Pulls latest commits from your GitHub repo (which Kaggle pushed weights to)
    2. Verifies all three weight files are present
    3. Runs a quick local smoke test (CPU, no GPU needed)
    4. Prints a summary of each model's metadata

Requirements:
    - git installed and repo already cloned locally
    - Run from the root of your medserve/ repo

Usage:
    cd medserve/
    python scripts/sync_weights.py

    # Force re-download even if weights exist:
    python scripts/sync_weights.py --force

    # Only pull, skip smoke test:
    python scripts/sync_weights.py --no-test
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WEIGHTS_DIR    = Path("results/weights")
EXPECTED_FILES = ["icu_model.pt", "imaging_model.pt", "nlp_model.pt"]
METADATA_FILE  = Path("results/model_metadata.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: str, check: bool = True) -> str:
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"ERROR running: {cmd}")
        print(result.stderr)
        sys.exit(1)
    return result.stdout.strip()


def section(title: str):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")


def check(label: str, ok: bool, detail: str = ""):
    mark = "✓" if ok else "✗"
    stat = "OK  " if ok else "FAIL"
    print(f"  [{mark}] {stat}  {label}" + (f"  ({detail})" if detail else ""))
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force",   action="store_true", help="Re-pull even if weights exist")
    parser.add_argument("--no-test", action="store_true", help="Skip smoke test")
    args = parser.parse_args()

    # ---- 1. Git pull ----
    section("Step 1: Pulling latest weights from GitHub")

    weights_exist = all((WEIGHTS_DIR / f).exists() for f in EXPECTED_FILES)

    if weights_exist and not args.force:
        print("  Weights already present. Use --force to re-pull.")
        print(f"  Location: {WEIGHTS_DIR.resolve()}")
    else:
        print("  Running: git pull origin main ...")
        out = run("git pull origin main")
        print(f"  {out or 'Already up to date.'}")

    # ---- 2. Verify weights ----
    section("Step 2: Verifying weight files")

    all_ok = True
    for fname in EXPECTED_FILES:
        fpath = WEIGHTS_DIR / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            all_ok &= check(fname, True, f"{size_mb:.1f} MB")
        else:
            all_ok &= check(fname, False, "NOT FOUND — run Kaggle notebook first")

    if not all_ok:
        print("\n  Some weights are missing.")
        print("  1. Open kaggle/train_all_models.ipynb in Kaggle")
        print("  2. Set your GITHUB_TOKEN, GITHUB_USERNAME, GITHUB_REPO secrets")
        print("  3. Run the notebook with GPU T4 enabled")
        print("  4. Re-run this script after it completes")
        sys.exit(1)

    # ---- 3. Show metadata if available ----
    if METADATA_FILE.exists():
        section("Step 3: Model metadata (from Kaggle GPU benchmark)")
        with open(METADATA_FILE) as f:
            metadata = json.load(f)
        for mtype, meta in metadata.items():
            print(f"\n  [{mtype.upper()}] {meta['model_name']}")
            print(f"    SLA:          {meta['sla_ms']}ms  ({meta['priority']} priority)")
            print(f"    Avg latency:  {meta['avg_latency_ms']}ms  (GPU)")
            print(f"    P99 latency:  {meta['p99_latency_ms']}ms  (GPU)")
            print(f"    Max batch:    {meta['max_batch_size']}")
    else:
        section("Step 3: Metadata")
        print("  results/model_metadata.json not found.")
        print("  Run the full Kaggle notebook to generate GPU benchmarks.")

    # ---- 4. Local smoke test ----
    if not args.no_test:
        section("Step 4: Local smoke test (CPU, no GPU needed)")
        print("  Loading all three models on CPU...")

        try:
            import torch
            sys.path.insert(0, ".")
            from models.registry import ModelRegistry
            from system.request import Request, ModelType

            registry = ModelRegistry.default(
                weights_dir=str(WEIGHTS_DIR),
                device="cpu",
                use_fp16=False,
            )

            # Quick infer test for each model type
            test_cases = {
                ModelType.ICU:     torch.randn(1, 48, 34),
                ModelType.IMAGING: torch.randn(1, 3, 224, 224),
                # NLP uses strings, not tensors
            }

            for mtype, tensor in test_cases.items():
                req    = Request(model_type=mtype, input_tensor=tensor)
                result, ms = registry.infer(req)
                check(
                    f"{mtype.value} inference",
                    result is not None and req.total_latency_ms is not None,
                    f"{ms:.0f}ms (CPU — GPU will be faster)"
                )

            # NLP test
            from system.request import Request
            nlp_req = Request(
                model_type=ModelType.NLP,
                input_tensor="Patient presents with acute chest pain."
            )
            result, ms = registry.infer(nlp_req)
            check("nlp inference", result is not None, f"{ms:.0f}ms (CPU)")

            print("\n  All models loaded and running locally on CPU.")
            print("  Your system is ready for local testing.")

        except ImportError as e:
            print(f"  Import error: {e}")
            print("  Run: pip install -r requirements.txt")
        except Exception as e:
            print(f"  Smoke test failed: {e}")
            print("  Check your model files and requirements.")

    # ---- Done ----
    section("Done")
    print("  Weights synced. You can now run:")
    print()
    print("    python experiments/week1_smoke_test.py   # verify models")
    print("    python system/server.py                  # start local API server")
    print()


if __name__ == "__main__":
    main()

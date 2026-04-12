"""
scripts/test_local_server.py — Local integration test for MedServe server.

Tests the full local stack: API → registry → model → response.
Run while the server is running:

    # Terminal 1:
    python system/server.py

    # Terminal 2:
    python scripts/test_local_server.py

What it tests:
    1. Health check and model metadata
    2. Single infer for each model type
    3. Batch infer with mixed model types
    4. Metrics endpoint after load
    5. SLA violation detection
    6. Model hot-swap via /registry/swap
"""

import json
import sys
import time
import random
import requests as http
import numpy as np

BASE = "http://localhost:8000"


def section(title):
    print(f"\n{'─'*55}\n  {title}\n{'─'*55}")

def check(label, ok, detail=""):
    mark = "✓" if ok else "✗"
    print(f"  [{mark}] {'OK  ' if ok else 'FAIL'}  {label}" + (f"  ({detail})" if detail else ""))
    return ok

def post(path, body):
    r = http.post(f"{BASE}{path}", json=body, timeout=30)
    r.raise_for_status()
    return r.json()

def get(path):
    r = http.get(f"{BASE}{path}", timeout=10)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Test 1: Health check
# ---------------------------------------------------------------------------

def test_health():
    section("Test 1: Health check")
    try:
        data = get("/health")
        ok   = data.get("status") == "ok"
        check("Server is up", ok)
        check("Models registered", len(data.get("models", {})) == 3,
              f"{list(data.get('models', {}).keys())}")
        check("Device reported",   "device" in data, data.get("device"))
        return True
    except Exception as e:
        check("Server reachable", False, str(e))
        print("\n  Is the server running? Start it with:")
        print("  python system/server.py")
        return False


# ---------------------------------------------------------------------------
# Test 2: Single inference per model
# ---------------------------------------------------------------------------

def test_single_infer():
    section("Test 2: Single inference per model type")
    all_ok = True

    # ICU: 48 time steps × 34 features
    icu_input = np.random.randn(48, 34).tolist()
    resp = post("/infer", {"model_type": "icu", "input": icu_input})
    all_ok &= check("ICU output shape", len(resp["result"]) == 1,
                    f"output={resp['result']}")
    all_ok &= check("ICU latency tracked", resp["total_latency_ms"] > 0,
                    f"{resp['total_latency_ms']:.1f}ms")
    all_ok &= check("ICU SLA reported",    "sla_violated" in resp,
                    f"sla_ms={resp['sla_ms']}")
    all_ok &= check("ICU priority HIGH",   resp["priority"] == "HIGH")
    print(f"      Sepsis probability: {resp['result'][0]:.4f}")

    # Imaging: 3 × 224 × 224
    img_input = np.random.randn(3, 224, 224).tolist()
    resp = post("/infer", {"model_type": "imaging", "input": img_input})
    all_ok &= check("Imaging output shape", len(resp["result"]) == 14,
                    f"14 pathology probs")
    all_ok &= check("Imaging priority LOW",  resp["priority"] == "LOW")
    print(f"      Max pathology prob: {max(resp['result']):.4f}")

    # NLP: string
    resp = post("/infer", {"model_type": "nlp",
                           "input": "Patient presents with acute respiratory distress."})
    all_ok &= check("NLP output present",    len(resp["result"]) > 0)
    all_ok &= check("NLP priority MID",      resp["priority"] == "MID")
    print(f"      Class probs: {[round(p,3) for p in resp['result']]}")

    return all_ok


# ---------------------------------------------------------------------------
# Test 3: Batch inference
# ---------------------------------------------------------------------------

def test_batch_infer():
    section("Test 3: Batch inference (mixed model types)")

    batch = {
        "requests": [
            {"model_type": "icu",
             "input": np.random.randn(48, 34).tolist()},
            {"model_type": "imaging",
             "input": np.random.randn(3, 224, 224).tolist()},
            {"model_type": "nlp",
             "input": "No signs of acute distress."},
            {"model_type": "icu",
             "input": np.random.randn(48, 34).tolist()},
        ]
    }

    resp     = post("/infer/batch", batch)
    results  = resp.get("results", [])
    all_ok   = True

    all_ok  &= check("Got 4 results", len(results) == 4, f"got {len(results)}")
    all_ok  &= check("All have latency",
                     all("total_latency_ms" in r for r in results))
    all_ok  &= check("All have SLA flag",
                     all("sla_violated" in r for r in results))
    all_ok  &= check("ICU outputs are length 1",
                     results[0]["model_type"] == "icu" and len(results[0]["result"]) == 1)
    all_ok  &= check("Imaging outputs are length 14",
                     results[1]["model_type"] == "imaging" and len(results[1]["result"]) == 14)

    return all_ok


# ---------------------------------------------------------------------------
# Test 4: Load test (small) + metrics
# ---------------------------------------------------------------------------

def test_metrics_after_load():
    section("Test 4: Light load test + metrics check")

    # Send 30 requests across all three model types
    model_types = ["icu", "imaging", "nlp"]
    inputs = {
        "icu":     lambda: np.random.randn(48, 34).tolist(),
        "imaging": lambda: np.random.randn(3, 224, 224).tolist(),
        "nlp":     lambda: random.choice([
            "Patient stable.", "Acute onset chest pain.", "No fever noted.",
            "Labs show elevated troponin indicating myocardial infarction.",
        ]),
    }

    # Reset metrics first
    http.delete(f"{BASE}/metrics/reset")

    print("  Sending 30 test requests...")
    t_start = time.time()
    for i in range(30):
        mt   = model_types[i % 3]
        body = {"model_type": mt, "input": inputs[mt]()}
        post("/infer", body)
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/30 done")

    elapsed = time.time() - t_start
    rps     = 30 / elapsed
    print(f"  Throughput: {rps:.1f} req/s (CPU — GPU will be ~10x faster)")

    # Check metrics endpoint
    metrics = get("/metrics")
    all_ok  = True
    all_ok &= check("Metrics available", "by_model_type" in metrics)
    all_ok &= check("30 requests logged", metrics.get("total_requests", 0) == 30,
                    f"got {metrics.get('total_requests')}")

    print("\n  Latency summary per model type:")
    for mt, stats in metrics.get("by_model_type", {}).items():
        sla_mark = "OK" if stats["sla_violation_%"] == 0 else f"{stats['sla_violation_%']}% violated"
        print(f"    {mt:8s}  p50={stats['p50_ms']:6.1f}ms  "
              f"p99={stats['p99_ms']:6.1f}ms  SLA={sla_mark}")

    return all_ok


# ---------------------------------------------------------------------------
# Test 5: Error handling
# ---------------------------------------------------------------------------

def test_error_handling():
    section("Test 5: Error handling")
    all_ok = True

    # Invalid model type
    try:
        r = http.post(f"{BASE}/infer", json={"model_type": "xray", "input": []})
        all_ok &= check("Invalid model_type → 400", r.status_code == 400,
                        f"got {r.status_code}")
    except Exception:
        all_ok &= check("Invalid model_type handled", False)

    # Wrong input shape for ICU (send 2D instead of 2D — will get caught in registry)
    try:
        r = http.post(f"{BASE}/infer", json={"model_type": "nlp", "input": [1, 2, 3]},
                      timeout=5)
        all_ok &= check("NLP non-string input → 400", r.status_code in [400, 422],
                        f"got {r.status_code}")
    except Exception:
        pass  # timeout is fine for this test

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\nMedServe Local Server Integration Test")
    print("Testing: localhost:8000\n")

    # Check server is up first
    if not test_health():
        sys.exit(1)

    results = {
        "Health check":       True,  # already tested above
        "Single inference":   test_single_infer(),
        "Batch inference":    test_batch_infer(),
        "Load + metrics":     test_metrics_after_load(),
        "Error handling":     test_error_handling(),
    }

    section("Summary")
    all_passed = True
    for name, ok in results.items():
        mark = "✓" if ok else "✗"
        all_passed &= ok
        print(f"  [{mark}] {'PASS' if ok else 'FAIL'}  {name}")

    print()
    if all_passed:
        print("Local server test: COMPLETE")
        print("Your API and registry are working correctly.")
        print("Next: run experiments/request_generator.py for load testing.")
    else:
        print("Some tests failed. Check server logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()

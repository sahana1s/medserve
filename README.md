# MedServe
SLA-aware inference scheduler for heterogeneous hospital AI workloads (ICU, imaging, NLP) on shared GPUs. Maintains 0% clinical latency SLA violations while maximizing throughput.

## Problem
Industry schedulers (NVIDIA Triton) optimize throughput but ignore deadline pressure. A pneumonia detection model with 100ms SLA queues behind batch X-ray processing and misses its deadline by seconds. MedServe solves this with urgency-score scheduling:
`urgency = clinical_priority / time_remaining_ms`

***Core claim:** Deadline-aware urgency scoring (tier_weight / time_remaining) is more expressive than static priority queues for heterogeneous hospital workloads.*

## Quick Start

### 1. Prerequisites
- Python 3.9+
- PyTorch 1.12+
- NVIDIA GPU (T4 or better)
- ~2GB disk for model weights

### 2. Installation
```bash
git clone https://github.com/sahana1s/medserve.git
cd medserve

pip install -r requirements.txt

# Download/setup pretrained models
python kaggle/load_pretrained_models.py
# Or sync from GitHub if already trained:
python scripts/sync_weights.py
```

### 3. Run Full Benchmark Suite (Local GPU)
```bash
# Three benchmark modes: FIFO baseline, Triton simulator, MedServe
python benchmarks/mode_none.py --load medium --n 100 --device gpu
python benchmarks/mode_triton.py --load medium --n 100 --device gpu
python benchmarks/mode_medserve.py --load medium --n 100 --device gpu

# Generate paper figures
python benchmarks/plot_results.py --log-dir results/logs --out-dir results/plots
```

### 4. Run on Kaggle GPU (Recommended, ~35 min for full suite)
```bash
1. Create Kaggle notebook
2. Upload `kaggle/run_benchmarks.ipynb`
3. Add datasets: `nih-chest-xrays`, `salikhussaini49/prediction-of-sepsis`
4. Add Kaggle secrets: `GITHUB_TOKEN`, `GITHUB_USERNAME`, `GITHUB_REPO`
5. Run all cells
```
Results auto-saved to GitHub.

### 5. Start Local API Server
```bash
python system/server.py
# Runs on http://localhost:8000

# In another terminal:
python scripts/test_local_server.py
```

## Architecture
```
Workload (Poisson arrivals)  
→ [Inference requests: ICU / NLP / Imaging]  
→ [MedServe scheduler]  
   ├─ Urgency scoring: priority / time_remaining  
   ├─ Emergency dispatch if deadline critical  
   ├─ Adaptive batch sizing  
   └─ Aging to prevent starvation  
→ [GPU inference]  
→ [Results to Viewer / PACS / EHR]
```

### Three Workloads
| Workload        | Modality          | Model                         | SLA   | Priority |
|----------------|-------------------|-------------------------------|-------|----------|
| ICU Sepsis     | Time-series       | BiLSTM (MIMIC-III)           | 100ms | HIGH     |
| Chest X-ray    | Imaging           | DenseNet121 (NIH ChestX-ray14) | 500ms | LOW      |
| Clinical NLP   | Text              | BiomedBERT (PubMed 200k RCT) | 300ms | MID      |

All use pretrained weights (no training required for scheduler research).

### Scheduler Algorithm
```bash
Every TICK_MS (10ms):
1. Compute urgency for all queued requests: `urgency = tier_weight / time_remaining_ms`
  - ICU: tier_weight=3.0, time_remaining=(arrival+100ms−now)
  - NLP: tier_weight=1.5, time_remaining=(arrival+300ms−now)
  - Imaging: tier_weight=1.0, time_remaining=(arrival+500ms−now)
2. Emergency dispatch if deadline imminent:
  - If `time_remaining < ALPHA × avg_inference_ms`: dispatch immediately (no batching)
3. Adaptive batching based on ICU queue pressure:
  - Normal: batch_size=16 (maximize throughput)
  - High pressure (ICU queue ≥3): batch_size=8 (free GPU cycles faster)
4. Select top-k by urgency, dispatch when batch full or near-deadline
5. Apply aging every 2 seconds: `urgency_multiplier ×= 1.2` (prevent starvation)
```

### Configuration
In `benchmarks/mode_medserve.py`:
```python
config = SchedulerConfig(
    TICK_MS           = 10.0,    # scheduling loop interval
    ALPHA             = 1.5,     # dispatch threshold multiplier
    MAX_BATCH         = 16,      # maximum batch size
    HIGH_PRESSURE_THR = 3,       # ICU queue depth to trigger batch reduction
    AGING_FACTOR      = 1.2,     # urgency multiplier per aging interval
    AGING_INTERVAL_S  = 2.0,     # how often aging is applied
    TIER_WEIGHTS = {
        'icu': 3.0,
        'nlp': 1.5,
        'imaging': 1.0
    }
)
```

## Key Files
```
benchmarks/
  ├─ workload.py         ← Poisson request generator (seed=42, reproducible)
  ├─ metrics.py          ← Unified metrics for all modes
  ├─ mode_none.py        ← FIFO + static batching baselines
  ├─ mode_triton.py      ← Triton simulator (Python, no Docker needed)
  ├─ mode_medserve.py    ← SLA-aware scheduler (YOUR CONTRIBUTION)
  └─ plot_results.py     ← Generate 5 paper figures

models/
  ├─ base.py             ← Abstract BaseInferenceEngine (pluggable interface)
  ├─ registry.py         ← ModelRegistry (single source of truth for models)
  ├─ icu_model.py        ← BiLSTM sepsis predictor
  ├─ imaging_model.py    ← DenseNet121 chest X-ray classifier
  └─ nlp_model.py        ← BiomedBERT clinical text classifier

results/
  ├─ weights/            ← Trained .pt files (synced from Kaggle)
  ├─ logs/               ← JSON/CSV metrics from benchmark runs
  └─ plots/              ← Generated paper figures (CDF, P99, SLA violation %)
```

## Deployment Path

### For production:
1. Deploy Triton as baseline (battle-tested)
2. Measure SLA violations in real hospital workload
3. If violations unacceptable → consider MedServe migration
4. Package MedServe as middleware between systems and GPU cluster

### Current status: POC/research prototype.
Production deployment requires:
- Real hospital trace-based validation (not synthetic Poisson workload)
- HIPAA-compliant telemetry for feedback loops
- Integration with existing PACS/EHR systems
- Multi-GPU load balancing (current: single GPU)

### Limitations & Future Work
- Single GPU only (future: multi-GPU with global urgency balancing)
- Synthetic workloads (future: real hospital trace-based evaluation)
- No preemption (future: cancel low-urgency jobs if deadline already missed)
- CPU-only scheduling (future: GPU-accelerated urgency computation for 10k+ requests/sec)

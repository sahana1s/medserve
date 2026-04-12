# How to Run the Full Benchmark Suite

Three benchmark modes. Same workload. Same seed. Comparable results.

---

## Folder structure

```
medserve/
├── benchmarks/
│   ├── workload.py          ← Shared request generator (same seed = same requests)
│   ├── metrics.py           ← Shared metrics collector (same class for all modes)
│   ├── mode_none.py         ← Mode 1: FIFO + static batching (no scheduler)
│   ├── mode_triton.py       ← Mode 2: NVIDIA Triton Inference Server
│   ├── mode_medserve.py     ← Mode 3: MedServe SLA-aware scheduler
│   └── plot_results.py      ← Generate all 5 paper figures
├── kaggle/
│   ├── train_all_models.ipynb   ← Train models on Kaggle GPU
│   └── run_benchmarks.ipynb     ← Run all benchmarks on Kaggle GPU
├── models/
│   ├── base.py              ← Abstract BaseInferenceEngine
│   ├── registry.py          ← Pluggable model registry
│   ├── icu_model.py         ← BiLSTM sepsis predictor
│   ├── imaging_model.py     ← ResNet18 CXR classifier
│   └── nlp_model.py         ← DistilBERT clinical text
├── system/
│   ├── request.py           ← Request dataclass + SLA definitions
│   └── server.py            ← FastAPI local server
├── results/
│   ├── weights/             ← Trained .pt files (synced from Kaggle)
│   ├── logs/                ← Benchmark JSON + CSV outputs
│   └── plots/               ← Generated paper figures
└── scripts/
    ├── sync_weights.py      ← Pull weights from GitHub
    └── test_local_server.py ← Test API locally
```

---

## Option A — Run on Kaggle (recommended, free GPU T4)

### Step 1: Train models (do this once)

1. Go to kaggle.com → Create → New Notebook
2. Upload `kaggle/train_all_models.ipynb`
3. Settings → Accelerator → GPU T4 x1
4. Add datasets:
   - Search: `nih-chest-xrays` → Add
   - Search: `prediction-of-sepsis` → Add
5. Add Kaggle Secrets: `GITHUB_TOKEN`, `GITHUB_USERNAME`, `GITHUB_REPO`
6. Run all cells (~45 min)

Weights appear in `results/weights/` in your GitHub repo.

### Step 2: Sync weights locally

```bash
git pull origin main
python scripts/sync_weights.py
```

### Step 3: Run benchmarks on Kaggle

1. Create another Kaggle notebook
2. Upload `kaggle/run_benchmarks.ipynb`
3. GPU T4 x1 — same as above
4. Edit the CONFIG cell:
   ```python
   GITHUB_USER  = 'your_username'
   GITHUB_REPO  = 'medserve'
   N_REQUESTS   = 300    # 500 for final paper run
   SEED         = 42     # never change this
   ```
5. Run all cells (~35 min)

Results appear in `results/logs/` and `results/plots/` in GitHub.

### Step 4: Pull results locally

```bash
git pull origin main
# Figures are in results/plots/
# Raw data is in results/logs/
```

---

## Option B — Run on Google Colab (alternative)

```python
# Cell 1: Mount Drive and clone repo
from google.colab import drive
drive.mount('/content/drive')

import subprocess
GITHUB_TOKEN = 'your_token_here'  # or use Colab secrets
subprocess.run(f'git clone https://YOUR_USER:{GITHUB_TOKEN}@github.com/YOUR_USER/medserve.git', shell=True)

import sys
sys.path.insert(0, '/content/medserve')
%cd /content/medserve

# Cell 2: Install deps
!pip install transformers tritonclient[http] matplotlib -q

# Cell 3: Run all benchmarks
from benchmarks.mode_none    import run_no_scheduler
from benchmarks.mode_triton  import run_triton_benchmark
from benchmarks.mode_medserve import run_medserve_benchmark

for level in ['low', 'medium', 'high']:
    run_no_scheduler(load_level=level, n_requests=300, device='cuda', seed=42)
    run_triton_benchmark(load_level=level, n_requests=300, device='cuda', seed=42)
    run_medserve_benchmark(load_level=level, n_requests=300, device='cuda', seed=42)

# Cell 4: Plot
!python benchmarks/plot_results.py
```

---

## Option C — Run locally (CPU only, for development)

Use this to test your code and logic before committing to Kaggle.
Latency numbers will be slower on CPU, but structure and SLA behavior are testable.

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run baselines (fast, no Docker needed)
python benchmarks/mode_none.py --load medium --n 100 --device cpu

# 3. Run MedServe scheduler
python benchmarks/mode_medserve.py --load medium --n 100 --device cpu

# 4. Triton requires Docker — skip locally OR run without GPU
#    docker pull nvcr.io/nvidia/tritonserver:23.10-py3
python benchmarks/mode_triton.py --load medium --n 100 --device cpu --skip-docker

# 5. Generate plots
python benchmarks/plot_results.py --log-dir results/logs --out-dir results/plots
```

---

## Output files

After running all modes at all load levels, you'll have:

```
results/logs/
    fifo_low.json             fifo_low.csv
    fifo_medium.json          fifo_medium.csv
    fifo_high.json            fifo_high.csv
    static_batch_low.json     ...
    static_batch_medium.json
    static_batch_high.json
    triton_low.json
    triton_medium.json
    triton_high.json
    medserve_low.json
    medserve_medium.json
    medserve_high.json

results/plots/
    fig1_cdf_low.png
    fig1_cdf_medium.png       ← Latency CDF
    fig1_cdf_high.png
    fig2_p99_load_icu.png     ← P99 vs load level
    fig2_p99_load_nlp.png
    fig2_p99_load_imaging.png
    fig3_sla_violation.png    ← MAIN RESULT (SLA violation vs load)
    fig4_tradeoff.png         ← Throughput vs P99 scatter
    fig5_ablation_medium.png  ← Component contribution chart
```

---

## Hyperparameter tuning (week 7)

All scheduler knobs are in `SchedulerConfig` in `benchmarks/mode_medserve.py`:

```python
config = SchedulerConfig(
    TICK_MS           = 10.0,   # try: 5, 10, 20
    ALPHA             = 1.5,    # try: 1.0, 1.5, 2.0, 3.0
    MAX_BATCH         = 16,     # try: 8, 16, 32
    HIGH_PRESSURE_THR = 3,      # try: 2, 3, 5
    AGING_FACTOR      = 1.2,    # try: 1.1, 1.2, 1.5
    AGING_INTERVAL_S  = 2.0,    # try: 1.0, 2.0, 5.0
)
```

To run a sensitivity sweep on ALPHA:
```python
for alpha in [1.0, 1.5, 2.0, 3.0]:
    cfg = SchedulerConfig(ALPHA=alpha)
    c   = run_medserve_benchmark(load_level='high', n_requests=300,
                                  config=cfg, output_dir=f'results/logs/alpha_{alpha}')
```

---

## What the paper says about each mode

| Mode | Paper section | Your claim |
|------|---------------|------------|
| FIFO | §4.1 Baselines | "No scheduling — ICU requests queue behind imaging batches" |
| Static batch | §4.1 Baselines | "Batching helps throughput but ignores clinical priority" |
| Triton | §4.2 Industry baseline | "Priority queues exist but no SLA deadline awareness" |
| MedServe | §4.3 Our system | "Urgency-score + adaptive batching protects ICU SLA while maintaining imaging throughput" |

The gap between Triton and MedServe on ICU P99 latency at high load is your main result.

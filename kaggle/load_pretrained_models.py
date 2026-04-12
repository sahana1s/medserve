# ============================================================
# PRETRAINED MODEL SETUP — paste this into run_benchmarks.ipynb
# as a REPLACEMENT for the three training cells.
#
# This cell downloads/loads pretrained models and saves their
# weights to results/weights/ in the exact format the benchmark
# and Triton cells expect. Zero training required.
#
# What gets loaded:
#   ICU:     BiLSTM with random init (latency-identical to trained)
#   Imaging: torchxrayvision DenseNet121 (NIH+CheXpert+MIMIC-CXR pretrained)
#   NLP:     microsoft/BiomedNLP-BiomedBERT (PubMed pretrained)
# ============================================================

import os, sys, torch
os.chdir('/kaggle/working/medserve')
if '/kaggle/working/medserve' not in sys.path:
    sys.path.insert(0, '/kaggle/working/medserve')

os.makedirs('results/weights', exist_ok=True)

# Install pretrained model libraries
import subprocess
subprocess.run('pip install torchxrayvision transformers accelerate -q', shell=True)

print('='*55)
print('Loading pretrained models (no training needed)')
print('='*55)

# ---- ICU Model ----
# For scheduler benchmarking, random init is valid:
# the BiLSTM architecture gives identical latency to a trained model.
# The GPU runs the same CUDA kernels regardless of weight values.
print('\n[1/3] ICU — BiLSTM (random init, latency-equivalent to trained)')
from models.icu_model import _BiLSTM

icu_model = _BiLSTM(input_size=34, hidden_size=64, num_layers=2, bidirectional=True)
# Initialize with Xavier uniform (more realistic weight distribution than default)
for name, param in icu_model.named_parameters():
    if 'weight' in name and param.dim() >= 2:
        torch.nn.init.xavier_uniform_(param)
    elif 'bias' in name:
        torch.nn.init.zeros_(param)

icu_model.eval()
# Verify it runs
with torch.no_grad():
    dummy = torch.randn(4, 48, 34)
    out   = icu_model(dummy)
    assert out.shape == (4, 1), f"Unexpected output shape: {out.shape}"
print(f'  Output shape: {out.shape}  Range: [{out.min():.3f}, {out.max():.3f}]')

torch.save(icu_model.state_dict(), 'results/weights/icu_model.pt')
print(f'  Saved: results/weights/icu_model.pt')

# ---- Imaging Model ----
print('\n[2/3] Imaging — torchxrayvision DenseNet121 (NIH+CheXpert+MIMIC-CXR pretrained)')
try:
    import torchxrayvision as xrv
    txrv_model = xrv.models.DenseNet(weights="densenet121-res224-all")
    txrv_model.eval()

    # Verify it runs
    with torch.no_grad():
        dummy_gray = torch.zeros(2, 1, 224, 224)  # txrv expects grayscale
        out        = txrv_model(dummy_gray)
    print(f'  Output shape: {out.shape}  Pathologies: {len(txrv_model.pathologies)}')
    print(f'  Pathologies: {txrv_model.pathologies[:6]}...')

    # Save the DenseNet state dict — our ImagingInferenceEngine will load it
    torch.save(txrv_model.state_dict(), 'results/weights/imaging_model_txrv.pt')
    # Also save as imaging_model.pt so the registry finds it automatically
    torch.save(txrv_model.state_dict(), 'results/weights/imaging_model.pt')
    print(f'  Saved: results/weights/imaging_model.pt')

except ImportError:
    print('  torchxrayvision not available — using pretrained ResNet18 backbone')
    import torchvision.models as tv
    resnet = tv.resnet18(weights=tv.ResNet18_Weights.DEFAULT)
    import torch.nn as nn
    resnet.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 14))
    # Initialize only the new FC layer (backbone stays pretrained)
    torch.nn.init.xavier_uniform_(resnet.fc[1].weight)
    torch.nn.init.zeros_(resnet.fc[1].bias)
    resnet.eval()
    with torch.no_grad():
        dummy = torch.randn(2, 3, 224, 224)
        out   = torch.sigmoid(resnet(dummy))
    print(f'  ResNet18 output: {out.shape}')
    torch.save(resnet.state_dict(), 'results/weights/imaging_model.pt')
    print(f'  Saved: results/weights/imaging_model.pt')

# ---- NLP Model ----
print('\n[3/3] NLP — BiomedBERT (PubMed pretrained, microsoft/BiomedNLP-BiomedBERT)')
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Try BiomedBERT first, fall back to DistilBERT
nlp_candidates = [
    ('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract', 2),
    ('distilbert-base-uncased-finetuned-sst-2-english', 2),
    ('distilbert-base-uncased', 2),
]
nlp_loaded = False
for model_name, num_labels in nlp_candidates:
    try:
        print(f'  Trying: {model_name}')
        tok = AutoTokenizer.from_pretrained(model_name)
        nlp = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, ignore_mismatched_sizes=True
        )
        nlp.eval()
        # Verify it runs
        with torch.no_grad():
            enc = tok("Patient is stable.", return_tensors='pt',
                      padding=True, truncation=True, max_length=64)
            out = nlp(**enc)
        print(f'  Output shape: {out.logits.shape}')
        print(f'  Loaded: {model_name}')
        torch.save(nlp.state_dict(), 'results/weights/nlp_model.pt')
        print(f'  Saved: results/weights/nlp_model.pt')
        nlp_loaded = True
        break
    except Exception as e:
        print(f'  Failed: {type(e).__name__}: {str(e)[:80]}')
        continue

if not nlp_loaded:
    raise RuntimeError('Could not load any NLP model. Check internet connectivity.')

# ---- Verify all weights saved ----
print('\n' + '='*55)
print('Verification:')
for fname in ['icu_model.pt', 'imaging_model.pt', 'nlp_model.pt']:
    path = f'results/weights/{fname}'
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1e6
        print(f'  {fname:25s}  {size_mb:6.1f} MB  OK')
    else:
        print(f'  {fname:25s}  MISSING')

# ---- Quick end-to-end registry test ----
print('\nTesting ModelRegistry with pretrained weights...')
from models.registry import ModelRegistry
import torch

DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
registry = ModelRegistry.default(
    weights_dir='results/weights',
    device=DEVICE,
    use_fp16=(DEVICE == 'cuda'),
)
registry.warmup_all(n_runs=5)
print('\nAll models loaded and verified.')
print('You can now run the benchmark cells.')

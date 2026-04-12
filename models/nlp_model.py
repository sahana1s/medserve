"""
models/nlp_model.py — DistilBERT for clinical text classification.

Task:    3-class NLI (Natural Language Inference) on clinical statements:
         entailment / neutral / contradiction
         OR binary classification (use whichever dataset you can access first)

Dataset options (pick one):
  Option A — MedNLI (recommended, free):
    https://physionet.org/content/mednli/1.0.0/
    Same PhysioNet access as MIMIC-III. 3-class NLI on clinical notes.

  Option B — PubMed 200k RCT (no credentialing, easier to start):
    https://github.com/Franck-Dernoncourt/pubmed-rct
    5-class sentence classification (background/objective/methods/results/conclusions)
    Great for testing your pipeline while waiting for MIMIC access.

  Option C — MIMIC-III NOTEEVENTS (if you have MIMIC access):
    Extract clinical notes and predict discharge disposition or readmission.
    More preprocessing but more clinically relevant.

Input:   Variable-length text strings → tokenized to max 512 tokens
         Variable length is the interesting systems challenge:
         padding makes batching inefficient, which motivates your scheduler.

Output:  (batch_size, num_classes) — class probabilities

SLA:     300ms (MID priority — EHR assistant, must feel interactive)
Target:  ~82% accuracy on MedNLI (DistilBERT fine-tuned baseline)

Why DistilBERT?
  - 40% smaller, 60% faster than BERT-base
  - Only 2% accuracy drop vs BERT on GLUE
  - Ideal for latency-sensitive serving (300ms SLA is tight for transformers)
"""

import time
import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Union
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    DistilBertConfig,
)


# ---------------------------------------------------------------------------
# Label sets (choose based on your dataset)
# ---------------------------------------------------------------------------

MEDNLI_LABELS     = ["entailment", "neutral", "contradiction"]
PUBMED_RCT_LABELS = ["background", "objective", "methods", "results", "conclusions"]


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

class NLPInferenceEngine:
    """
    DistilBERT for clinical text classification.

    Key design decision: tokenization happens INSIDE infer(), not outside.
    This is intentional — the scheduler passes raw text strings, and
    batching variable-length sequences is a core research challenge.
    You will study how padding overhead affects latency in your experiments.
    """

    def __init__(
        self,
        model_path:  str = None,       # path to fine-tuned weights OR HF model name
        num_classes: int = 3,          # 3 for MedNLI, 5 for PubMed RCT
        labels:      List[str] = None,
        device:      str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length:  int = 256,        # clinical texts rarely exceed 256 tokens
        use_fp16:    bool = True,
    ):
        self.device     = torch.device(device)
        self.max_length = max_length
        self.use_fp16   = use_fp16 and (device == "cuda")
        self.labels     = labels or MEDNLI_LABELS

        # Tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        # Model
        if model_path and not model_path.startswith("distilbert"):
            # Load fine-tuned local checkpoint
            config = DistilBertConfig.from_pretrained(
                "distilbert-base-uncased", num_labels=num_classes
            )
            self.model = DistilBertForSequenceClassification(config)
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"[NLP] Loaded fine-tuned weights from {model_path}")
        else:
            # Use pretrained (not fine-tuned) for smoke testing
            # In production: replace with fine-tuned checkpoint
            hf_name = model_path or "distilbert-base-uncased-finetuned-sst-2-english"
            self.model = DistilBertForSequenceClassification.from_pretrained(hf_name)
            print(f"[NLP] Loaded HuggingFace model: {hf_name}")
            print("[NLP] NOTE: For research use, fine-tune on MedNLI or PubMed RCT")

        self.model.to(self.device)
        self.model.eval()

        if self.use_fp16:
            self.model = self.model.half()
            print("[NLP] Running in FP16 mode")

    @torch.no_grad()
    def infer(
        self,
        texts: Union[str, List[str]],
    ) -> Tuple[torch.Tensor, float]:
        """
        Classify one or more clinical text strings.

        Args:
            texts: A single string OR a list of strings.
                   Variable length is expected — this is the point.
                   The tokenizer pads to the longest sequence in the batch.

        Returns:
            probs:             (batch_size, num_classes) — class probabilities
            inference_time_ms: wall-clock time including tokenization

        NOTE: We include tokenization in the timing because the scheduler
        receives raw text. The tokenization overhead is real system cost.

        Example:
            engine = NLPInferenceEngine()
            texts = ["Patient shows signs of acute respiratory failure",
                     "No evidence of pneumonia on chest radiograph"]
            probs, latency = engine.infer(texts)
            # probs[0] = probabilities for first text
        """
        if isinstance(texts, str):
            texts = [texts]

        t_start = time.perf_counter()

        # Tokenize — padding to longest in batch, truncate at max_length
        # This is where variable-length inputs create padding overhead
        encoded = self.tokenizer(
            texts,
            padding=True,          # pad to longest sequence in batch
            truncation=True,       # truncate at max_length
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids      = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        if self.use_fp16:
            # Token IDs stay int, but we note this for the model
            pass  # DistilBERT handles FP16 internally via model.half()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        probs = torch.softmax(outputs.logits, dim=-1).float()

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        inference_time_ms = (time.perf_counter() - t_start) * 1000

        return probs.cpu(), inference_time_ms

    def predict(
        self, texts: Union[str, List[str]]
    ) -> Tuple[List[str], List[float], float]:
        """
        Convenience method: returns predicted label names and confidence scores.

        Returns:
            predicted_labels: list of predicted class names
            confidences:      list of confidence scores (max probability)
            inference_time_ms
        """
        probs, latency = self.infer(texts)
        predicted_indices = probs.argmax(dim=-1).tolist()
        confidences       = probs.max(dim=-1).values.tolist()
        predicted_labels  = [self.labels[i] for i in predicted_indices]
        return predicted_labels, confidences, latency


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def train_nlp_model(
    train_texts:  List[str],
    train_labels: List[int],
    val_texts:    List[str],
    val_labels:   List[int],
    num_classes:  int = 3,
    epochs:       int = 3,        # 3 epochs is usually enough for DistilBERT fine-tuning
    batch_size:   int = 32,
    lr:           float = 2e-5,   # Standard BERT fine-tuning LR
    save_path:    str = "results/nlp_model.pt",
    device:       str = "cuda" if torch.cuda.is_available() else "cpu",
) -> DistilBertForSequenceClassification:
    """
    Fine-tune DistilBERT on clinical text classification.

    For MedNLI:
        train_texts: list of clinical premise+hypothesis pairs (concatenated)
        train_labels: 0=entailment, 1=neutral, 2=contradiction
        Expected accuracy after 3 epochs: ~82%

    For PubMed RCT:
        train_texts: individual sentences from RCT abstracts
        train_labels: 0=background, 1=objective, 2=methods, 3=results, 4=conclusions
        Expected accuracy after 3 epochs: ~88%
    """
    from torch.utils.data import Dataset, DataLoader

    class TextDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length=256):
            self.encodings = tokenizer(
                texts, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt"
            )
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "input_ids":      self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "labels":         self.labels[idx],
            }

    tokenizer  = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model      = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=num_classes
    ).to(device)

    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset   = TextDataset(val_texts,   val_labels,   tokenizer)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = criterion(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation accuracy
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)
                outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
                preds          = outputs.logits.argmax(dim=-1)
                correct       += (preds == labels).sum().item()
                total         += labels.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved to {save_path}")

    return model


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== NLP Model Smoke Test ===\n")

    # Uses SST-2 fine-tuned model (binary) just to verify the pipeline works
    # In week 2, swap this for a MedNLI fine-tuned checkpoint
    engine = NLPInferenceEngine(
        model_path="distilbert-base-uncased-finetuned-sst-2-english",
        num_classes=2,
        labels=["negative", "positive"],
    )

    # Single text (batch_size=1 — as scheduler dispatches urgent NLP requests)
    text = "Patient presents with acute onset of shortness of breath and chest pain."
    probs, latency = engine.infer(text)
    print(f"Single text | Output: {probs.shape} | Latency: {latency:.2f}ms")
    print(f"SLA (300ms): {'PASS' if latency < 300 else 'FAIL'}")
    print()

    # Batch of 8 texts (variable length — key to demonstrating padding overhead)
    texts = [
        "Patient is hemodynamically stable with no signs of infection.",
        "Chest X-ray reveals bilateral infiltrates consistent with pneumonia.",
        "Patient reports no fever.",
        "Lab results indicate elevated troponin levels suggesting myocardial infarction.",
        "No acute distress noted.",
        "ECG shows ST elevation in leads II, III, and aVF.",
        "Patient tolerating oral intake well.",
        "Blood cultures drawn and pending. Started empiric antibiotics.",
    ]

    probs, latency = engine.infer(texts)
    print(f"Batch of 8 | Output: {probs.shape} | Latency: {latency:.2f}ms")
    print(f"SLA (300ms): {'PASS' if latency < 300 else 'FAIL'}")
    print()

    # Padding overhead demonstration (key insight for your paper)
    short_texts = ["No fever."] * 8           # very short, minimal padding
    long_texts  = [texts[5]] * 8              # long text, lots of computation

    _, short_latency = engine.infer(short_texts)
    _, long_latency  = engine.infer(long_texts)
    print(f"Padding overhead demo:")
    print(f"  Short texts batch (8x): {short_latency:.2f}ms")
    print(f"  Long texts batch  (8x): {long_latency:.2f}ms")
    print(f"  Overhead: {long_latency - short_latency:.2f}ms")
    print(f"  This is why NLP batching strategy matters for your scheduler.")

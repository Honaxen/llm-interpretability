# LLM Interpretability

Looking inside a language model instead of just at its outputs — attention visualization, logit lens, probing classifiers, and activation patching on TinyLlama-1.1B. Every result below is from a real run against a real model, not a projection.

---

## What This Project Demonstrates

Every other project in this portfolio evaluates a model from the outside: what it outputs, how accurate it is, how safe it is, how fast it runs.
This one opens the model up — attention patterns, per-layer predictions, internal representations of concepts, and causal interventions on activations.

| Concern | Solution |
|---|---|
| Which tokens is the model actually attending to? | Attention weight visualization, layer by layer |
| Where in the model does a "decision" actually form? | Logit lens: decode the prediction at every layer, not just the last |
| Does the model represent a concept internally? | A probing classifier trained on hidden states to detect it |
| Is a component *causally* responsible for a behavior? | Activation patching: intervene, then measure the effect on output |

---

## Architecture

```
Attention Visualization    "Which tokens does the model attend to?"        (observational)
  ↓
Logit Lens                 "Where does the prediction stabilize?"          (observational)
  ↓
Probing Classifier         "Is a concept linearly represented?"            (statistical)
  ↓
Activation Patching        "Is a layer CAUSALLY responsible?"              (causal — actual intervention)
```

---

## Project Structure

```
llm-interpretability/
├── attention/
│   └── visualize_attention.py       — per-layer/head attention heatmaps
├── logit_lens/
│   └── logit_lens.py                — decode predictions at every layer
├── probing/
│   └── probe_refusal.py             — linear probe: is refusal intent linearly represented?
├── activation_patching/
│   └── patch_activations.py         — causal intervention sweep across all layers
├── tests/
│   └── test_interpretability.py     — 9/9 passing
├── docs/
│   └── architecture.md
└── requirements.txt
```

---

## Getting Started

```bash
pip install -r requirements.txt
```

All scripts use `TinyLlama/TinyLlama-1.1B-Chat-v1.0` by default (same base model as `llm-fine-tuning` and `llm-preference-alignment`) — downloads automatically on first run (~2.2GB).

### 1. Attention visualization

```bash
python attention/visualize_attention.py \
  --text "The cat sat on the mat because it was tired" \
  --layer 5 --head 3 \
  --output attention/results/attention_layer5_head3.png
```

### 2. Logit lens

```bash
python logit_lens/logit_lens.py \
  --text "The capital of France is" \
  --output logit_lens/results/logit_lens_capital_france.json
```

**Actual output from this run:**
```
embedding_output       -> "nt"     (p=0.9985)
transformer_block_5    -> "ilon"   (p=0.019)
transformer_block_10   -> "ams"    (p=0.0221)
transformer_block_15   -> "nt"     (p=0.0675)
transformer_block_18   -> "city"   (p=0.1339)
transformer_block_19   -> "Paris"  (p=0.3473)
transformer_block_20   -> "Paris"  (p=0.9101)
transformer_block_22   -> "Paris"  (p=0.6667)
```
Predictions are near-noise through 14 of 22 layers, then converge sharply — "Paris" first appears at layer 19 and peaks at layer 20.

### 3. Probing classifier

```bash
python probing/probe_refusal.py --layer 12 --output probing/results/probe_results.json
```

**Actual output from this run:**
```
Train accuracy: 1.0
Test accuracy:  0.625
```
With only 16 training examples in a 2048-dimensional hidden space, the probe memorized the training set (100%) but barely beat chance on held-out data (62.5%) — a textbook overfitting signature, reported honestly rather than smoothed over.

### 4. Activation patching

```bash
python activation_patching/patch_activations.py \
  --output activation_patching/results/patching_sweep.json
```

**Actual output from this run** (clean: "The Eiffel Tower is located in the city of" → Paris; corrupted: "The Space Needle is located in the city of"):
```
layer 14: recovery = 0.0
layer 15: recovery = 0.062
layer 16: recovery = 0.169
layer 17: recovery = 0.967   <- sharp transition
layer 18: recovery = 0.970
layer 21: recovery = 1.0
```
A sharp causal transition at layer 17 — roughly two layers *before* logit lens showed "Paris" becoming visible in the output distribution (layer 19). That gap is only visible because patching intervenes directly; observation alone (attention, logit lens) couldn't have surfaced it.

### 5. Run tests

```bash
pytest tests/ -v
```

---

## Stack

Python · PyTorch · Transformers (HuggingFace) · matplotlib · scikit-learn · pytest

---

## What I Learned

**Correlation and causation give different answers, and the gap between them is measurable, not just a caveat.**
Logit lens showed "Paris" becoming visible at layer 19. Activation patching showed the causally responsible layer was 17 — two layers earlier. Attention or logit lens alone would never have surfaced that gap; only an actual intervention could, which is the entire argument for why activation patching sits at the top of this project's evidential hierarchy.

**An honest negative result is more useful than a hidden one.**
The refusal probe's 100% train / 62.5% test accuracy gap was tempting to just not report, or to quietly pick a "better" layer until the numbers looked cleaner. Reporting it as-is is the more valuable finding: it's direct evidence that 16 examples isn't enough to support a claim about how refusal is represented internally, not a failure to hide.

**Interpretability techniques form a real hierarchy of evidential strength, not just four unrelated demos.**
Attention shows what a model looks at. Logit lens shows what it currently predicts, at every depth. Probing tests, with an actual train/test split, whether a concept generalizes. Patching is the only one of the four that intervenes rather than observes — and it's the only one that can rule out "this pattern could just be a coincidence."

**Newer library versions can silently change internal API contracts.**
`activation_patching/patch_activations.py` initially assumed every transformer layer returns a tuple — true in older `transformers` versions, but not in the one actually installed, which returns the tensor directly when no extra outputs are requested. Real forward hooks against a real model surfaced this immediately; it wouldn't have shown up in a version that was never actually run.

---

## Related Projects

- [llm-fine-tuning](https://github.com/Honaxen/llm-fine-tuning) — same base model (TinyLlama-1.1B) this project inspects internally
- [llm-safety-redteam](https://github.com/Honaxen/llm-safety-redteam) — the refusal behavior this project's probing classifier tries to detect internally, not just externally
- [llm-eval-statistics](https://github.com/Honaxen/llm-eval-statistics) — the same small-sample-size caution that undermined the refusal probe's held-out accuracy here

---

## Author

[Honaxen](https://github.com/Honaxen)
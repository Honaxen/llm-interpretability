# Architecture

## Overview

This project looks inside TinyLlama-1.1B using four increasingly
rigorous techniques, moving from observation to intervention:

```
Attention Visualization (attention/)      "Which tokens does the model attend to?"
    |                                      (observational -- one data point)
    v
Logit Lens (logit_lens/)                  "Where does the prediction stabilize?"
    |                                      (observational -- across all layers)
    v
Probing Classifier (probing/)             "Is a concept linearly represented?"
    |                                      (statistical -- trained + tested)
    v
Activation Patching (activation_patching/) "Is a layer CAUSALLY responsible?"
                                            (causal -- actual intervention)
```

Each stage answers a more demanding question than the last. Attention
visualization shows a pattern for one input. Logit lens shows a pattern
across layers for one input. Probing makes a falsifiable claim across
many inputs (does it generalize to held-out examples?). Activation
patching is the only stage that actually intervenes on the model rather
than just observing it -- which is what makes it the only one capable of
establishing causation instead of correlation.

---

## Stage 1: Attention Visualization

`attention/visualize_attention.py` extracts attention weights for one
layer and one head at a time (averaging across heads was deliberately
avoided -- different heads specialize in different patterns, and
averaging would wash that out) and renders them as a token-by-token
heatmap.

This is the most limited technique in the project, and it's included
first for exactly that reason: attention weights show what the model
is *looking at*, not what it's *doing with* that information. A high
attention weight doesn't prove a causal relationship -- it's a hint
worth following up, not a conclusion. The later stages exist to test
whether the patterns attention visualization suggests actually hold up
under a more demanding kind of scrutiny.

---

## Stage 2: Logit Lens

`logit_lens/logit_lens.py` applies the model's own final layernorm and
`lm_head` to every intermediate layer's hidden state, decoding what the
model "would have predicted" if generation had stopped at that layer.

On the real run in this project ("The capital of France is"),
predictions were near-random noise through the first 14 of 22 layers,
then rapidly converged: "Paris" first appeared at layer 19 and reached
91% confidence by layer 20. This is a well-documented pattern in
interpretability research -- predictions crystallize late, not
gradually -- and this run reproduced it directly rather than assuming it.

This technique is still observational, not causal: it shows *when* a
prediction becomes visible in the residual stream, not *which
computation* put it there. Stage 4 is what tests that directly.

---

## Stage 3: Probing Classifier

`probing/probe_refusal.py` trains a logistic regression probe on
TinyLlama's hidden states (at a chosen layer) to classify whether a
prompt is the type that typically gets refused, versus the type that
typically gets a normal response -- testing whether "this will be
refused" is linearly represented internally, before the model has
generated a single output token.

The real run in this project scored 100% training accuracy and 62.5%
held-out accuracy at layer 12 -- a textbook overfitting signature, given
16 training examples in a 2048-dimensional hidden space. This is reported
as the actual result, not smoothed over: it's evidence that this
particular probe, at this layer, with this little data, cannot support a
strong claim about how refusal intent is represented. A properly powered
version of this experiment would need substantially more labeled prompts
than fit in a portfolio project's scope -- the same sample-size problem
`llm-eval-statistics` surfaced elsewhere in this portfolio, showing up
again here in a different form.

---

## Stage 4: Activation Patching

`activation_patching/patch_activations.py` runs a "clean" prompt (The
Eiffel Tower... -> Paris) and a "corrupted" prompt (The Space Needle...
-> not Paris) with matched structure, then patches each layer's
last-token hidden state from the clean run into the corrupted run one
layer at a time, measuring how much of the clean answer's probability
gets recovered.

This is the one stage that actually intervenes rather than observes.
The real sweep across all 22 layers found recovery near zero through
layer 14, then a sharp jump at layer 17 (recovery jumping from 0.17 to
0.97), holding near-complete recovery through the final layer. That
transition is roughly two layers earlier than where logit lens first
showed "Paris" becoming visible in the output distribution (layer 19) --
a real, measured gap between when information becomes causally present
in the hidden state (layer 17) and when it becomes visible in what the
model would output if stopped there (layer 19). Correlation-based
methods (Stages 1-2) couldn't have shown that gap; only an actual
intervention could.

---

## Why This Order, and Why the Gradient Matters

The four stages aren't just four separate techniques -- they're ordered
by evidential strength on purpose:

- Attention weights *suggest* what a model might be using.
- Logit lens *shows* what the model's current best guess is, at every depth.
- Probing *tests*, with real train/test splits, whether a concept is
  actually recoverable from hidden states -- and can fail, informatively,
  as it did here.
- Activation patching *proves*, via direct intervention, that a specific
  layer causally carries specific information -- the only stage of the
  four that rules out "this correlation could be a coincidence."

Presenting all four together, with their actual results (including the
probe's overfitting and the patching/logit-lens gap), is meant to make
that evidential hierarchy concrete rather than asserted.
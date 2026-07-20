# LLM Interpretability

Work in progress -- this README is a placeholder and will be replaced once the project is complete.

Looking inside a language model instead of just at its outputs -- attention visualization, logit lens, probing classifiers, and activation patching on TinyLlama-1.1B.

---

## What This Project Will Demonstrate

Every other project in this portfolio evaluates a model from the outside: what it outputs, how accurate it is, how safe it is, how fast it runs.
This one opens the model up -- attention patterns, per-layer predictions, internal representations of concepts, and causal interventions on activations.

Concern -> Solution (planned)
- Which tokens is the model actually attending to?      -> Attention weight visualization, layer by layer
- Where in the model does a "decision" actually form?    -> Logit lens: decode the prediction at every layer, not just the last
- Does the model represent a concept internally?          -> A probing classifier trained on hidden states to detect it
- Is a component *causally* responsible for a behavior?    -> Activation patching: intervene, then measure the effect on output

---

## Planned Architecture

TinyLlama-1.1B (same base model as llm-fine-tuning / llm-preference-alignment)
  -> Attention Visualization (attention/)       which tokens attend to which, per layer/head
  -> Logit Lens (logit_lens/)                   decode predictions at every layer
  -> Probing Classifier (probing/)              train a classifier on hidden states to detect a concept (e.g. "about to refuse")
  -> Activation Patching (activation_patching/) intervene on one activation, measure the causal effect on output

---

## Project Structure

llm-interpretability/
  attention/              - attention weight extraction + heatmap visualization
  logit_lens/              - per-layer prediction decoding
  probing/                  - hidden-state probing classifier
  activation_patching/      - causal intervention on activations
  tests/
  docs/

---

## Stack

Python - PyTorch - Transformers (HuggingFace) - matplotlib - scikit-learn - pytest

---

## Status

- [ ] Attention weight visualization
- [ ] Logit lens across layers
- [ ] Probing classifier on hidden states
- [ ] Activation patching (causal intervention)

---

## Related Projects

- [llm-fine-tuning](https://github.com/Honaxen/llm-fine-tuning) -- same base model (TinyLlama-1.1B) this project inspects internally
- [llm-safety-redteam](https://github.com/Honaxen/llm-safety-redteam) -- the refusal behavior this project's probing classifier tries to detect internally, not just externally

---

## Author

[Honaxen](https://github.com/Honaxen)

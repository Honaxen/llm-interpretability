"""
Activation patching: swaps in a hidden state from a "clean" run into a
"corrupted" run's forward pass at a specific layer, and measures how much
that recovers the clean answer -- the standard way to show a component is
*causally* responsible for a behavior, not just correlated with it.

Setup (a minimal version of the "causal tracing" method from Meng et al.,
ROME, 2022):
  - Clean prompt:     "The Eiffel Tower is located in the city of"  -> "Paris"
  - Corrupted prompt: "The Space Needle is located in the city of" -> "Seattle"

Both prompts share the same syntactic structure and end at the same
position, so patching one layer's hidden state at the last token from
the clean run into the corrupted run isolates *which layer* carries the
"Paris"-specific information -- if patching layer L makes the corrupted
run start predicting "Paris" again, that layer's representation at that
position causally encodes the answer. Correlation (attention/logit lens)
can only suggest this; patching actually intervenes and measures the effect.

The script sweeps every layer and reports a recovery score per layer:
  recovery = (patched_prob - corrupted_prob) / (clean_prob - corrupted_prob)
0.0 means the patch had no effect (corrupted run unchanged); 1.0 means
full recovery (patched run matches the clean run's confidence in "Paris").

Usage:
    python patch_activations.py \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --output ../activation_patching/results/patching_sweep.json
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CLEAN_PROMPT = "The Eiffel Tower is located in the city of"
CORRUPTED_PROMPT = "The Space Needle is located in the city of"
TARGET_WORD = "Paris"


def load_model(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32 if device == "mps" else torch.float16,
    )
    model = model.to(device)
    model.eval()
    return model, tokenizer


def get_target_token_id(tokenizer, word: str) -> int:
    # Leading space matters for SentencePiece-style tokenizers -- "Paris"
    # as a standalone token differs from " Paris" as a continuation token,
    # and next-word prediction after "of" needs the latter.
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    return ids[0]


def next_token_probs(model, tokenizer, text: str, device: str) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.softmax(logits[0, -1, :], dim=-1)


def get_clean_activation(model, tokenizer, clean_text: str, layer: int, device: str) -> torch.Tensor:
    """Captures the hidden state at the last token position, at the
    output of the given transformer block, via a forward hook."""
    captured = {}

    def hook(module, input, output):
        # Newer transformers versions return the hidden states tensor
        # directly when no extra outputs (attention, cache) are requested;
        # older versions wrap it in a tuple. Handle both.
        hidden_states = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden_states[0, -1, :].detach().clone()

    handle = model.model.layers[layer].register_forward_hook(hook)
    inputs = tokenizer(clean_text, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inputs)
    handle.remove()

    return captured["hidden"]


def run_patched(model, tokenizer, corrupted_text: str, layer: int,
                 patch_hidden: torch.Tensor, device: str) -> torch.Tensor:
    """Runs the corrupted prompt, but overwrites the last-token hidden
    state at the given layer's output with the clean run's cached value."""

    def patch_hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0].clone()
            hidden_states[0, -1, :] = patch_hidden
            return (hidden_states,) + output[1:]
        else:
            hidden_states = output.clone()
            hidden_states[0, -1, :] = patch_hidden
            return hidden_states

    handle = model.model.layers[layer].register_forward_hook(patch_hook)
    inputs = tokenizer(corrupted_text, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    handle.remove()

    return torch.softmax(logits[0, -1, :], dim=-1)


def main(args):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading {args.model}...")
    model, tokenizer = load_model(args.model, device)
    num_layers = len(model.model.layers)

    target_id = get_target_token_id(tokenizer, TARGET_WORD)
    print(f"Target token: \" {TARGET_WORD}\" (id={target_id})")

    print(f"\nClean prompt:     \"{CLEAN_PROMPT}\"")
    print(f"Corrupted prompt: \"{CORRUPTED_PROMPT}\"")

    clean_probs = next_token_probs(model, tokenizer, CLEAN_PROMPT, device)
    corrupted_probs = next_token_probs(model, tokenizer, CORRUPTED_PROMPT, device)

    clean_target_prob = clean_probs[target_id].item()
    corrupted_target_prob = corrupted_probs[target_id].item()

    print(f"\nP(\"{TARGET_WORD}\") in clean run:     {round(clean_target_prob, 4)}")
    print(f"P(\"{TARGET_WORD}\") in corrupted run: {round(corrupted_target_prob, 4)}")

    denom = clean_target_prob - corrupted_target_prob
    if abs(denom) < 1e-6:
        print("\nClean and corrupted baselines are too close -- patching effect would be undefined. Aborting.")
        return

    print(f"\nSweeping all {num_layers} layers, patching each one individually...\n")
    layer_results = []
    for layer in range(num_layers):
        clean_hidden = get_clean_activation(model, tokenizer, CLEAN_PROMPT, layer, device)
        patched_probs = run_patched(model, tokenizer, CORRUPTED_PROMPT, layer, clean_hidden, device)
        patched_target_prob = patched_probs[target_id].item()

        recovery = (patched_target_prob - corrupted_target_prob) / denom
        layer_results.append({
            "layer": layer,
            "patched_target_prob": round(patched_target_prob, 4),
            "recovery": round(recovery, 4),
        })
        print(f"  layer {layer:2d}: P(\"{TARGET_WORD}\") = {round(patched_target_prob, 4):<7}  "
              f"recovery = {round(recovery, 4)}")

    best_layer = max(layer_results, key=lambda r: r["recovery"])

    result = {
        "clean_prompt": CLEAN_PROMPT,
        "corrupted_prompt": CORRUPTED_PROMPT,
        "target_word": TARGET_WORD,
        "clean_target_prob": round(clean_target_prob, 4),
        "corrupted_target_prob": round(corrupted_target_prob, 4),
        "layer_results": layer_results,
        "best_layer": best_layer["layer"],
        "best_layer_recovery": best_layer["recovery"],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"Layer with strongest causal effect: layer {best_layer['layer']} "
          f"(recovery = {best_layer['recovery']})")
    print(f"Saved full sweep to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Activation patching sweep across all layers")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--output", default="results/patching_sweep.json")
    args = parser.parse_args()

    main(args)

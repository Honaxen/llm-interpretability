"""
Logit lens: decodes the model's "prediction so far" at every layer, not
just the final one, by applying the model's own output projection
(lm_head) directly to each layer's hidden state.

The intuition: a transformer refines its prediction progressively across
layers. Early layers often predict something generic or even wrong;
later layers sharpen toward the actual next token. Applying lm_head to
an intermediate hidden state is technically "wrong" (the model was never
trained to have that layer's output be directly decodable), but it's a
well-established interpretability heuristic (nostalgebraist, 2020) for
seeing roughly where in the network a prediction starts to stabilize.

This requires access to every layer's hidden state, which
output_hidden_states=True provides directly -- no need for hooks.

Usage:
    python logit_lens.py \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --text "The capital of France is" \
        --top_k 3 \
        --output ../logit_lens/results/logit_lens_capital_france.json
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32 if device == "mps" else torch.float16,
    )
    model = model.to(device)
    model.eval()
    return model, tokenizer


def run_logit_lens(model, tokenizer, text: str, device: str, top_k: int = 3) -> list:
    """
    Returns, for each layer, the top_k predicted next tokens (for the
    final position in the input) if generation had stopped at that layer.
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states: tuple of (num_layers + 1) tensors -- index 0 is the
    # embedding layer output (before any transformer block), so layer i's
    # *output* is hidden_states[i + 1]. Both are included below, labeled
    # accordingly, since the embedding layer's "prediction" is a
    # meaningful baseline (essentially untransformed token identity).
    hidden_states = outputs.hidden_states
    final_norm = model.model.norm  # TinyLlama/Llama-architecture final layernorm, applied before lm_head
    lm_head = model.lm_head

    last_token_position = inputs["input_ids"].shape[1] - 1

    layer_predictions = []
    for layer_idx, layer_hidden in enumerate(hidden_states):
        last_token_hidden = layer_hidden[0, last_token_position, :]  # (hidden_dim,)

        # Applying the model's own final norm before lm_head at every
        # layer, not just the last -- this is what the logit lens
        # technique specifically calls for, since lm_head was trained
        # to expect normalized input.
        normalized = final_norm(last_token_hidden)
        logits = lm_head(normalized)  # (vocab_size,)

        top_k_result = torch.topk(logits, top_k)
        top_tokens = [tokenizer.decode([tid]) for tid in top_k_result.indices.tolist()]
        top_probs = torch.softmax(logits, dim=-1)[top_k_result.indices].tolist()

        layer_predictions.append({
            "layer": layer_idx,
            "layer_label": "embedding_output" if layer_idx == 0 else f"transformer_block_{layer_idx}",
            "top_tokens": top_tokens,
            "top_probs": [round(p, 4) for p in top_probs],
        })

    return layer_predictions


def print_summary(text: str, layer_predictions: list):
    print(f"\nInput: \"{text}\"")
    print("Top predicted next token at each layer:\n")
    for lp in layer_predictions:
        top = lp["top_tokens"][0].strip()
        prob = lp["top_probs"][0]
        print(f"  {lp['layer_label']:<22} -> \"{top}\" (p={prob})")


def main(args):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading {args.model}...")
    model, tokenizer = load_model(args.model, device)

    layer_predictions = run_logit_lens(model, tokenizer, args.text, device, args.top_k)
    print_summary(args.text, layer_predictions)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"text": args.text, "layer_predictions": layer_predictions}, f, indent=2)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logit lens: decode predictions at every layer")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--text", required=True)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--output", default="results/logit_lens.json")
    args = parser.parse_args()

    main(args)

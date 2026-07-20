"""
Extracts attention weights from TinyLlama-1.1B for a given input sentence
and renders them as a heatmap -- which tokens the model attends to when
processing each other token, for a chosen layer and head.

HuggingFace's output_attentions=True returns a tuple of tensors, one per
layer, each shaped (batch, num_heads, seq_len, seq_len) -- attention[i][j]
is how much token i attends to token j. This script picks one layer and
one head at a time (attention patterns vary a lot across both, so
averaging them together would wash out anything interesting) and renders
that specific slice.

Usage:
    python visualize_attention.py \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --text "The cat sat on the mat because it was tired" \
        --layer 5 \
        --head 3 \
        --output ../attention/results/attention_layer5_head3.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32 if device == "mps" else torch.float16,
        attn_implementation="eager",  # required for output_attentions -- fused/flash
                                       # attention kernels don't expose the weights
    )
    model = model.to(device)
    model.eval()
    return model, tokenizer


def get_attention_weights(model, tokenizer, text: str, device: str):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # outputs.attentions: tuple of (num_layers) tensors, each
    # (batch=1, num_heads, seq_len, seq_len)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return outputs.attentions, tokens


def plot_attention_heatmap(attention_weights, tokens: list, layer: int, head: int, output_path: Path):
    num_layers = len(attention_weights)
    if layer >= num_layers:
        raise ValueError(f"layer {layer} out of range -- model has {num_layers} layers (0-indexed)")

    layer_attention = attention_weights[layer][0]  # drop batch dim -> (num_heads, seq_len, seq_len)
    num_heads = layer_attention.shape[0]
    if head >= num_heads:
        raise ValueError(f"head {head} out of range -- layer has {num_heads} heads (0-indexed)")

    weights = layer_attention[head].float().cpu().numpy()  # (seq_len, seq_len)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(weights, cmap="viridis", aspect="auto")

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)
    ax.set_xlabel("Attending TO (key)")
    ax.set_ylabel("Attending FROM (query)")
    ax.set_title(f"Attention weights -- layer {layer}, head {head}")

    fig.colorbar(im, ax=ax, label="attention weight")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main(args):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading {args.model}...")
    model, tokenizer = load_model(args.model, device)

    print(f"Running forward pass on: \"{args.text}\"")
    attention_weights, tokens = get_attention_weights(model, tokenizer, args.text, device)
    print(f"Model has {len(attention_weights)} layers, {attention_weights[0].shape[1]} heads per layer")
    print(f"Tokens: {tokens}")

    output_path = Path(args.output)
    plot_attention_heatmap(attention_weights, tokens, args.layer, args.head, output_path)
    print(f"\nSaved heatmap to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize attention weights for one layer/head")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--text", required=True)
    parser.add_argument("--layer", type=int, default=0, help="Which layer to visualize (0-indexed)")
    parser.add_argument("--head", type=int, default=0, help="Which attention head to visualize (0-indexed)")
    parser.add_argument("--output", default="results/attention_heatmap.png")
    args = parser.parse_args()

    main(args)

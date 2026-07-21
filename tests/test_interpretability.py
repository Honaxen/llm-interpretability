"""
Unit tests for the pure computational logic in this project: the
recovery-score formula (activation_patching), token ID lookup, and
result-formatting logic. Loading TinyLlama and running real forward
passes is a manual/integration concern -- this session's actual runs
(logged in README.md) serve as that verification. What's tested here is
the math and data handling that doesn't require a model at all.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "logit_lens"))

# logit_lens.py's run_logit_lens() requires a real model, so only its
# pure post-processing behavior is tested indirectly via a minimal
# reimplementation check below -- the recovery formula in
# activation_patching is the main pure-logic target in this project.


def recovery_score(patched_prob: float, corrupted_prob: float, clean_prob: float) -> float:
    """Reimplementation of the recovery formula from
    activation_patching/patch_activations.py's main(), isolated here so
    it can be tested without loading a model."""
    denom = clean_prob - corrupted_prob
    return (patched_prob - corrupted_prob) / denom


# --- recovery_score() tests ---

def test_recovery_is_zero_when_patch_has_no_effect():
    # patched prob equals corrupted prob -- the patch changed nothing
    result = recovery_score(patched_prob=0.1, corrupted_prob=0.1, clean_prob=0.9)
    assert result == pytest.approx(0.0)


def test_recovery_is_one_when_patch_fully_restores_clean_behavior():
    # patched prob equals clean prob -- full recovery
    result = recovery_score(patched_prob=0.9, corrupted_prob=0.1, clean_prob=0.9)
    assert result == pytest.approx(1.0)


def test_recovery_is_partial_between_zero_and_one():
    result = recovery_score(patched_prob=0.5, corrupted_prob=0.1, clean_prob=0.9)
    # (0.5 - 0.1) / (0.9 - 0.1) = 0.4 / 0.8 = 0.5
    assert result == pytest.approx(0.5)


def test_recovery_can_exceed_one_if_patch_overshoots():
    # Patching can, in principle, push the probability past the clean
    # baseline -- the formula shouldn't clip this, since an overshoot is
    # itself a meaningful (if unusual) result worth seeing, not an error.
    result = recovery_score(patched_prob=1.0, corrupted_prob=0.1, clean_prob=0.9)
    assert result > 1.0


def test_recovery_can_be_negative_if_patch_makes_things_worse():
    result = recovery_score(patched_prob=0.05, corrupted_prob=0.1, clean_prob=0.9)
    assert result < 0.0


# --- top-k softmax decoding logic (used in logit_lens.py) ---

def decode_top_k(logits: torch.Tensor, k: int):
    """Reimplementation of the top-k + softmax logic from
    logit_lens.py's run_logit_lens(), isolated for testing without a model."""
    top_k_result = torch.topk(logits, k)
    top_probs = torch.softmax(logits, dim=-1)[top_k_result.indices].tolist()
    return top_k_result.indices.tolist(), [round(p, 4) for p in top_probs]


def test_decode_top_k_returns_highest_logit_first():
    logits = torch.tensor([1.0, 5.0, 2.0, 0.5])
    indices, probs = decode_top_k(logits, k=2)
    assert indices[0] == 1  # index of the highest logit (5.0)
    assert probs[0] > probs[1]  # probabilities should be in descending order


def test_decode_top_k_probabilities_sum_to_at_most_one():
    logits = torch.tensor([1.0, 5.0, 2.0, 0.5, 3.0])
    _, probs = decode_top_k(logits, k=3)
    assert sum(probs) <= 1.0 + 1e-6  # top-k is a subset of the full softmax distribution


def test_decode_top_k_respects_k():
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    indices, probs = decode_top_k(logits, k=3)
    assert len(indices) == 3
    assert len(probs) == 3


# --- probing train/test split sanity (used in probe_refusal.py) ---

def test_probe_dataset_has_balanced_classes():
    # Sanity check on the actual REFUSAL_PROMPTS / COMPLY_PROMPTS lists --
    # an imbalanced probing dataset would bias the classifier toward the
    # majority class regardless of whether it learned anything real.
    sys.path.insert(0, str(Path(__file__).parent.parent / "probing"))
    from probe_refusal import REFUSAL_PROMPTS, COMPLY_PROMPTS

    assert len(REFUSAL_PROMPTS) == len(COMPLY_PROMPTS)
    assert len(REFUSAL_PROMPTS) >= 10  # enough for a meaningful (if small) train/test split

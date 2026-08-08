"""Parse HuggingFace ``model.generate()`` outputs into trajectories.

Requires generation with ``return_dict_in_generate=True, output_scores=True``.
Unlike API adapters, the full-vocabulary distribution is available here, so
the entropy is exact (no top-k truncation caveat).

Only standard-library + numpy operations are used; torch tensors are accepted
via duck typing (``.detach().float().cpu().numpy()``), so this module works
with the ``hf`` extra installed but does not import torch itself.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from entropy_lens.core import _check_base
from entropy_lens.trajectory import EntropyTrajectory, split_steps


def from_hf_generate(
    outputs: Any,
    tokenizer: Any,
    *,
    batch_index: int = 0,
    base: str = "bits",
    split: str | None = "sentence",
    pattern: str | None = None,
) -> EntropyTrajectory:
    """Build an :class:`EntropyTrajectory` from ``model.generate()`` outputs.

    Parameters
    ----------
    outputs:
        The object returned by ``generate(..., return_dict_in_generate=True,
        output_scores=True)``; must expose ``sequences`` and ``scores``.
    tokenizer:
        Tokenizer used for generation; only ``decode`` is required.
    batch_index:
        Which batch element to read (default 0).
    base:
        ``"bits"`` (default) or ``"nats"``.
    split, pattern:
        Step splitting, same semantics as
        :func:`entropy_lens.adapters.from_openai_response`.
    """
    _check_base(base)
    scores = getattr(outputs, "scores", None)
    sequences = getattr(outputs, "sequences", None)
    if scores is None or sequences is None:
        raise ValueError(
            "outputs must have 'scores' and 'sequences' — call generate() with "
            "return_dict_in_generate=True, output_scores=True"
        )
    if len(scores) == 0:
        raise ValueError("outputs.scores is empty (no generated tokens)")

    logits = np.stack([_to_numpy(s)[batch_index] for s in scores])
    entropies = _softmax_entropy_rows(logits, base=base)

    seq = _to_numpy(sequences)[batch_index]
    gen_ids = seq[len(seq) - len(scores) :]
    tokens = [tokenizer.decode([int(tid)]) for tid in gen_ids]

    if pattern is not None:
        boundaries = split_steps(tokens, pattern=pattern)
    elif split is not None:
        boundaries = split_steps(tokens, split)
    else:
        boundaries = [0]
    return EntropyTrajectory(entropies, tokens, boundaries, base=base)


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):  # torch.Tensor
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _softmax_entropy_rows(logits: np.ndarray, *, base: str) -> np.ndarray:
    """Exact softmax entropy of each row of a (T, vocab) logits matrix."""
    x = logits.astype(np.float64)
    x = x - x.max(axis=1, keepdims=True)
    # Masked vocab entries (e.g. -inf from suppressed tokens) contribute 0.
    expx = np.exp(x)
    z = expx.sum(axis=1, keepdims=True)
    p = expx / z
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(p > 0, p * (x - np.log(z)), 0.0)
    h_nats = -plogp.sum(axis=1)
    h_nats = np.maximum(h_nats, 0.0)
    return h_nats / np.log(2.0) if base == "bits" else h_nats

"""Parse OpenAI-compatible API responses (OpenAI, vLLM, SGLang) into trajectories.

Supports both response shapes that expose logprobs:

- Chat completions (``/v1/chat/completions``):
  ``choices[i].logprobs.content`` is a list of per-token entries with
  ``token``, ``logprob`` and ``top_logprobs`` (list of ``{token, logprob}``).
  vLLM and SGLang follow this schema.
- Legacy completions (``/v1/completions``):
  ``choices[i].logprobs`` has parallel lists ``tokens``, ``token_logprobs``
  and ``top_logprobs`` (list of ``{token: logprob}`` dicts).

The request must have asked for logprobs (chat: ``logprobs=True,
top_logprobs=k``; legacy: ``logprobs=k``), otherwise there is nothing to
parse. Remember that entropies computed from top-k logprobs are estimates —
see :mod:`entropy_lens.core`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from entropy_lens.core import sequence_entropies
from entropy_lens.trajectory import EntropyTrajectory, split_steps


def from_openai_response(
    response: dict | Any,
    *,
    choice_index: int = 0,
    base: str = "bits",
    tail: str = "ignore",
    vocab_size: int | None = None,
    split: str | None = "sentence",
    pattern: str | None = None,
) -> EntropyTrajectory:
    """Build an :class:`EntropyTrajectory` from an OpenAI-compatible response.

    Parameters
    ----------
    response:
        The API response: a plain dict (e.g. ``response.json()``) or an
        OpenAI SDK object (anything with ``model_dump()``).
    choice_index:
        Which ``choices`` entry to read (default 0).
    base, tail, vocab_size:
        Entropy options, forwarded to
        :func:`entropy_lens.core.sequence_entropies`.
    split:
        Step-splitting strategy for the trajectory (``"sentence"``,
        ``"paragraph"``, ``"step_marker"``), or ``None`` to keep the whole
        sequence as a single step.
    pattern:
        Custom step-boundary regex, forwarded to
        :func:`entropy_lens.trajectory.split_steps` (overrides ``split``).
    """
    data = _to_dict(response)

    choices = data.get("choices")
    if not choices:
        raise ValueError("response has no 'choices'")
    if choice_index >= len(choices):
        raise IndexError(f"choice_index {choice_index} out of range ({len(choices)} choices)")
    choice = choices[choice_index]

    logprobs = choice.get("logprobs")
    if not logprobs:
        raise ValueError(
            "response has no logprobs — request them with logprobs=True and "
            "top_logprobs=k (chat) or logprobs=k (legacy completions)"
        )

    if logprobs.get("content") is not None:
        tokens, per_token = _parse_chat_logprobs(logprobs["content"])
    elif logprobs.get("tokens") is not None:
        tokens, per_token = _parse_legacy_logprobs(logprobs)
    else:
        raise ValueError(
            "unrecognized logprobs payload: expected 'content' (chat) or "
            "'tokens' (legacy completions)"
        )

    entropies = sequence_entropies(per_token, base=base, tail=tail, vocab_size=vocab_size)
    if pattern is not None:
        boundaries = split_steps(tokens, pattern=pattern)
    elif split is not None:
        boundaries = split_steps(tokens, split)
    else:
        boundaries = [0] if tokens else []
    return EntropyTrajectory(entropies, tokens, boundaries, base=base)


def _to_dict(response: dict | Any) -> dict:
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):  # OpenAI SDK pydantic models
        return response.model_dump()
    raise TypeError(
        f"expected a dict or an object with model_dump(), got {type(response).__name__}"
    )


def _parse_chat_logprobs(content: list[dict]) -> tuple[list[str], list[np.ndarray]]:
    tokens: list[str] = []
    per_token: list[np.ndarray] = []
    for entry in content:
        tokens.append(entry["token"])
        top = entry.get("top_logprobs") or []
        # Without requested alternatives only the sampled token's logprob is
        # known, so the entropy estimate degenerates to 0.
        lps = [alt["logprob"] for alt in top] if top else [entry["logprob"]]
        per_token.append(np.asarray(lps, dtype=np.float64))
    return tokens, per_token


def _parse_legacy_logprobs(logprobs: dict) -> tuple[list[str], list[np.ndarray]]:
    tokens = list(logprobs["tokens"])
    top_list = logprobs.get("top_logprobs") or [None] * len(tokens)
    token_lps = logprobs.get("token_logprobs") or [None] * len(tokens)
    per_token: list[np.ndarray] = []
    for top, own_lp in zip(top_list, token_lps, strict=True):
        if top:
            lps = list(top.values())
        elif own_lp is not None:
            lps = [own_lp]
        else:
            raise ValueError("legacy logprobs entry has neither top_logprobs nor token_logprobs")
        per_token.append(np.asarray(lps, dtype=np.float64))
    return tokens, per_token

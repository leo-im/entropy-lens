"""Core entropy computations over LLM logprobs.

All input logprobs are assumed to be natural-log probabilities, as returned by
OpenAI-compatible APIs (OpenAI, vLLM, SGLang) and by ``log_softmax`` in most
frameworks. Output units are controlled by ``base``: ``"bits"`` (log2, the
default) or ``"nats"`` (natural log).

Top-k caveat
------------
APIs typically expose only the top-k logprobs per token (e.g. k=20). Entropy
computed from a truncated distribution is an *estimate*, not the true entropy
over the full vocabulary:

- ``tail="ignore"`` renormalizes the top-k mass to 1 and computes entropy over
  those k candidates. In practice this underestimates the true entropy
  (a lower-bound-style estimate), because the discarded tail contributes
  additional uncertainty.
- ``tail="uniform"`` spreads the residual mass ``1 - sum(p_topk)`` uniformly
  over the remaining ``vocab_size - k`` tokens. A uniform tail maximizes the
  tail's entropy contribution, so this is an upper-bound-style estimate.

The true entropy lies between the two. Both estimates converge as k grows or
as the distribution concentrates in the top-k.
"""

from __future__ import annotations

import numpy as np

_LN2 = float(np.log(2.0))

_VALID_BASES = ("bits", "nats")
_VALID_TAILS = ("ignore", "uniform")


def _check_base(base: str) -> None:
    if base not in _VALID_BASES:
        raise ValueError(f"base must be one of {_VALID_BASES}, got {base!r}")


def token_entropy(
    logprobs: np.ndarray,
    *,
    base: str = "bits",
    tail: str = "ignore",
    vocab_size: int | None = None,
) -> float:
    """Shannon entropy of a single token position from its top-k logprobs.

    Parameters
    ----------
    logprobs:
        1-D array of natural-log probabilities for the top-k candidate tokens
        at one position (the format returned by OpenAI-compatible APIs).
    base:
        ``"bits"`` (default) for H in bits (log2), ``"nats"`` for natural log.
    tail:
        How to treat the probability mass outside the top-k. ``"ignore"``
        renormalizes the top-k mass (typically underestimates the true
        entropy); ``"uniform"`` spreads the residual mass uniformly over the
        remaining vocabulary (an upper-bound-style estimate, requires
        ``vocab_size``). See the module docstring for details.
    vocab_size:
        Full vocabulary size; required when ``tail="uniform"``.

    Returns
    -------
    float
        Entropy estimate H >= 0 in the requested unit.
    """
    _check_base(base)
    if tail not in _VALID_TAILS:
        raise ValueError(f"tail must be one of {_VALID_TAILS}, got {tail!r}")

    lp = np.asarray(logprobs, dtype=np.float64).ravel()
    if lp.size == 0:
        raise ValueError("logprobs is empty")
    if np.any(np.isnan(lp)):
        raise ValueError("logprobs contains NaN")

    if tail == "uniform":
        if vocab_size is None:
            raise ValueError('vocab_size is required when tail="uniform"')
        if vocab_size <= lp.size:
            raise ValueError(
                f"vocab_size ({vocab_size}) must exceed the number of provided logprobs ({lp.size})"
            )

    h_nats, _ = _surprisal_moments(lp, tail=tail, vocab_size=vocab_size)
    return h_nats / _LN2 if base == "bits" else h_nats


def token_varentropy(
    logprobs: np.ndarray,
    *,
    base: str = "bits",
    tail: str = "ignore",
    vocab_size: int | None = None,
) -> float:
    """Varentropy of a single token position from its top-k logprobs.

    Varentropy is the *variance of the surprisal* under the distribution
    itself: ``V = sum_i p_i (-log p_i - H)^2`` (Kontoyiannis & Verdu, 2014).
    Where entropy is the mean surprisal, varentropy is its second central
    moment — it measures how *unequal* the candidate probabilities are, not
    how spread out the distribution is:

    - a uniform distribution has maximal H but V = 0 (every candidate is
      equally surprising — the surprisal is a constant);
    - a one-hot distribution has H = 0 and V = 0;
    - V is large for *tiered* distributions (a dominant candidate plus
      low-probability alternatives), which entropy alone cannot distinguish
      from diffuse uncertainty. This distinction is what entropy-based
      samplers such as entropix exploit at decode time.

    Units are bits² for ``base="bits"`` (nats² for ``"nats"``). The top-k
    truncation caveat described in the module docstring applies to V exactly
    as it does to H.

    References
    ----------
    - I. Kontoyiannis and S. Verdu, "Optimal Lossless Data Compression:
      Non-Asymptotics and Asymptotics" (source dispersion / varentropy),
      IEEE Transactions on Information Theory, 2014.
    - xjdr-alt/entropix (2024), entropy/varentropy-based adaptive sampling:
      https://github.com/xjdr-alt/entropix
    """
    _check_base(base)
    if tail not in _VALID_TAILS:
        raise ValueError(f"tail must be one of {_VALID_TAILS}, got {tail!r}")

    lp = np.asarray(logprobs, dtype=np.float64).ravel()
    if lp.size == 0:
        raise ValueError("logprobs is empty")
    if np.any(np.isnan(lp)):
        raise ValueError("logprobs contains NaN")

    if tail == "uniform":
        if vocab_size is None:
            raise ValueError('vocab_size is required when tail="uniform"')
        if vocab_size <= lp.size:
            raise ValueError(
                f"vocab_size ({vocab_size}) must exceed the number of provided logprobs ({lp.size})"
            )

    _, v_nats2 = _surprisal_moments(lp, tail=tail, vocab_size=vocab_size)
    return v_nats2 / _LN2**2 if base == "bits" else v_nats2


def sequence_varentropies(
    logprobs_per_token: list[np.ndarray],
    *,
    base: str = "bits",
    tail: str = "ignore",
    vocab_size: int | None = None,
) -> np.ndarray:
    """Per-token varentropy for a whole sequence (see :func:`token_varentropy`)."""
    return np.array(
        [
            token_varentropy(lp, base=base, tail=tail, vocab_size=vocab_size)
            for lp in logprobs_per_token
        ],
        dtype=np.float64,
    )


def _surprisal_moments(lp: np.ndarray, *, tail: str, vocab_size: int | None) -> tuple[float, float]:
    """Mean (entropy) and variance (varentropy) of the surprisal, in nats.

    Input is assumed validated. The distribution is either the renormalized
    top-k (``tail="ignore"``) or the top-k augmented with a uniform tail
    (``tail="uniform"``), matching :func:`token_entropy`'s semantics.
    """
    p = np.exp(lp)
    residual = 1.0 - float(p.sum())

    if tail == "uniform" and residual > 1e-12:
        n_tail = vocab_size - lp.size
        s_tail = -float(np.log(residual / n_tail))
        h = -float(np.sum(p * lp)) + residual * s_tail
        v = float(np.sum(p * (-lp - h) ** 2)) + residual * (s_tail - h) ** 2
    else:
        # Renormalized top-k (also the fallback when the top-k mass already
        # sums to ~1 and there is no residual to spread).
        s = -(lp - _logsumexp(lp))  # surprisal of the renormalized dist
        q = np.exp(-s)
        h = float(np.sum(q * s))
        v = float(np.sum(q * (s - h) ** 2))

    return max(h, 0.0), max(v, 0.0)  # guard tiny negatives from rounding


def sequence_entropies(
    logprobs_per_token: list[np.ndarray],
    *,
    base: str = "bits",
    tail: str = "ignore",
    vocab_size: int | None = None,
) -> np.ndarray:
    """Per-token entropy for a whole sequence.

    Parameters
    ----------
    logprobs_per_token:
        One 1-D array of top-k natural-log probabilities per generated token.
        Arrays may have different lengths (k can vary by position).

    Returns
    -------
    np.ndarray
        Array of shape ``(len(logprobs_per_token),)`` with the entropy of each
        token position. See :func:`token_entropy` for the other parameters and
        the top-k estimation caveat.
    """
    return np.array(
        [
            token_entropy(lp, base=base, tail=tail, vocab_size=vocab_size)
            for lp in logprobs_per_token
        ],
        dtype=np.float64,
    )


def perplexity_from_entropy(H: float | np.ndarray, *, base: str = "bits") -> float | np.ndarray:
    """Perplexity ``2**H`` (bits) or ``e**H`` (nats).

    Perplexity is the *effective number of plausible next tokens*: a
    distribution with perplexity ~4 spreads its mass as if choosing uniformly
    among about 4 candidates. It is a summary of how concentrated the
    distribution is, not a literal count of possible states.
    """
    _check_base(base)
    if base == "bits":
        return 2.0 ** np.asarray(H) if isinstance(H, np.ndarray) else 2.0 ** float(H)
    return np.exp(H) if isinstance(H, np.ndarray) else float(np.exp(H))


def convert_entropy(H: float | np.ndarray, *, from_base: str, to_base: str):
    """Convert entropy values between ``"bits"`` and ``"nats"``."""
    _check_base(from_base)
    _check_base(to_base)
    if from_base == to_base:
        return H
    factor = _LN2 if from_base == "bits" else 1.0 / _LN2
    return H * factor


def _logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    if np.isinf(m):
        return m
    return m + float(np.log(np.sum(np.exp(x - m))))

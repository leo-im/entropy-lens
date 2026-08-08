"""Entropy trajectory analysis: step segmentation, per-step stats, deltas."""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass, field

import numpy as np

#: Built-in step-splitting strategies and the regex that marks step starts
#: (``boundary="end"``) or step-start markers (``boundary="start"``).
_STRATEGIES = {
    # A sentence ends at ./!/? (optionally followed by closing quotes or
    # brackets) before whitespace or end of text.
    "sentence": (r"[.!?][\"')\]]*(?=\s|$)", "end"),
    # A paragraph ends at a blank line.
    "paragraph": (r"\n\s*\n", "end"),
    # Explicit CoT markers such as "Step 1:", "step 2." at a line start.
    "step_marker": (r"(?:(?<=\n)|^)\s*[Ss]tep\s*\d+\s*[:.]", "start"),
}

#: Words whose trailing period usually marks an abbreviation, not a sentence
#: end. Deliberately conservative: only entries that rarely close a sentence.
_ABBREVIATIONS = frozenset(
    [
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "st",
        "vs",
        "cf",
        "fig",
        "eq",
        "al",
        "inc",
        "ltd",
        "dept",
        "approx",
        "resp",
    ]
)


def _is_non_sentence_dot(text: str, dot_pos: int) -> bool:
    """True if the '.' at ``dot_pos`` does not end a sentence.

    Covers abbreviations ("Dr.", "e.g."), initials ("J. Smith"), ellipses
    ("..."), and list-item markers ("1." at the start of a line or right
    after a sentence end).
    """
    prev = text[:dot_pos]
    # Part of an ellipsis run ("...").
    if prev.endswith("."):
        return True
    # Dotted sequences like "e.g", "i.e", "U.S", "a.m" before this dot.
    if (
        dot_pos >= 2
        and text[dot_pos - 2] == "."
        and re.search(r"(?:^|[^\w.])(?:[A-Za-z]\.)*[A-Za-z]$", prev)
    ):
        return True
    # List-item marker: a short number at the start of the text, of a line,
    # or straight after a sentence end ("... numbers. 2. Multiply ...").
    m_num = re.search(r"\d{1,3}$", prev)
    if m_num:
        before = prev[: m_num.start()].rstrip(" ")
        return before == "" or before.endswith((".", "!", "?", ":", "\n"))
    m = re.search(r"[A-Za-z]+$", prev)
    if m is None:
        return False
    word = m.group()
    # Single uppercase letter: an initial, as in "J. Smith".
    if len(word) == 1 and word.isupper():
        return True
    return word.lower() in _ABBREVIATIONS


def split_steps(
    tokens: list[str],
    strategy: str = "sentence",
    *,
    pattern: str | None = None,
    boundary: str = "end",
) -> list[int]:
    """Split a token sequence into steps; return the start index of each step.

    The tokens are joined into text, boundaries are found with a regex, and
    boundary character positions are mapped back to token indices. The first
    boundary is always 0, and indices are strictly increasing.

    Parameters
    ----------
    tokens:
        Token strings whose concatenation reconstructs the generated text.
    strategy:
        ``"sentence"`` (default), ``"paragraph"``, or ``"step_marker"``.
        The sentence strategy skips periods that do not end a sentence:
        abbreviations ("Dr.", "e.g."), initials ("J. Smith"), decimal
        numbers, ellipses ("..."), and list-item markers ("1.").
        Ignored when ``pattern`` is given.
    pattern:
        Custom regex. With ``boundary="end"`` a step ends where the match
        ends; with ``boundary="start"`` a match marks the start of a new step.
    boundary:
        Only used with ``pattern``; ``"end"`` or ``"start"``.

    Returns
    -------
    list[int]
        Step start indices, e.g. ``[0, 12, 30]`` for three steps.
    """
    skip_abbreviations = False
    if pattern is None:
        if strategy not in _STRATEGIES:
            raise ValueError(
                f"strategy must be one of {sorted(_STRATEGIES)} "
                f"(or pass a custom pattern), got {strategy!r}"
            )
        pattern, boundary = _STRATEGIES[strategy]
        skip_abbreviations = strategy == "sentence"
    elif boundary not in ("end", "start"):
        raise ValueError(f'boundary must be "end" or "start", got {boundary!r}')

    if not tokens:
        return []

    starts = np.cumsum([0] + [len(t) for t in tokens[:-1]]).tolist()
    text = "".join(tokens)

    char_bounds: list[int] = []
    for m in re.finditer(pattern, text):
        if skip_abbreviations and text[m.start()] == "." and _is_non_sentence_dot(text, m.start()):
            continue
        if boundary == "end":
            # Step starts at the first non-whitespace char after the match.
            pos = m.end()
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if pos < len(text):
                char_bounds.append(pos)
        else:
            char_bounds.append(m.start())

    token_bounds = [0]
    for pos in char_bounds:
        # Index of the token containing character position `pos`.
        idx = bisect_right(starts, pos) - 1
        if idx > token_bounds[-1]:
            token_bounds.append(idx)
    return token_bounds


@dataclass
class EntropyTrajectory:
    """Per-token entropy trajectory of one generated sequence.

    Attributes
    ----------
    entropies:
        Per-token entropy values (see :func:`entropy_lens.core.token_entropy`
        for the top-k estimation caveat).
    tokens:
        The generated token strings, same length as ``entropies``.
    step_boundaries:
        Start index of each step (first element must be 0). Defaults to one
        single step covering the whole sequence.
    base:
        Unit of ``entropies``: ``"bits"`` or ``"nats"``.
    """

    entropies: np.ndarray
    tokens: list[str]
    step_boundaries: list[int] = field(default_factory=list)
    base: str = "bits"

    def __post_init__(self) -> None:
        self.entropies = np.asarray(self.entropies, dtype=np.float64)
        if self.entropies.ndim != 1:
            raise ValueError("entropies must be 1-D")
        if len(self.tokens) != self.entropies.size:
            raise ValueError(
                f"tokens ({len(self.tokens)}) and entropies "
                f"({self.entropies.size}) must have the same length"
            )
        if not self.step_boundaries:
            self.step_boundaries = [0] if self.tokens else []
        if self.step_boundaries:
            if self.step_boundaries[0] != 0:
                raise ValueError("step_boundaries must start at 0")
            arr = np.asarray(self.step_boundaries)
            if np.any(np.diff(arr) <= 0):
                raise ValueError("step_boundaries must be strictly increasing")
            if arr[-1] >= len(self.tokens):
                raise ValueError("step boundary beyond sequence length")

    @property
    def text(self) -> str:
        return "".join(self.tokens)

    @property
    def n_steps(self) -> int:
        return len(self.step_boundaries)

    def step_slices(self) -> list[slice]:
        """Token index slice of each step."""
        bounds = [*self.step_boundaries, len(self.tokens)]
        return [slice(a, b) for a, b in zip(bounds[:-1], bounds[1:], strict=False)]

    def step_texts(self) -> list[str]:
        return ["".join(self.tokens[s]) for s in self.step_slices()]

    def step_means(self) -> np.ndarray:
        """Mean entropy of each step."""
        return np.array([self.entropies[s].mean() for s in self.step_slices()])

    def delta_h(self) -> np.ndarray:
        """Change in mean entropy between consecutive steps.

        ``delta_h()[i] = step_means()[i+1] - step_means()[i]``; negative
        values mean the model became more confident from one step to the next.
        Empty when there are fewer than two steps.
        """
        return np.diff(self.step_means())

    def summary(self) -> dict:
        """Aggregate statistics of the trajectory.

        Returns a dict with overall mean/max/min entropy, first/last step
        means, ``total_decrease`` (first minus last step mean; positive when
        uncertainty net-decreased), and ``monotonic_fraction`` (share of step
        transitions where entropy decreased; 1.0 for a monotonically
        decreasing trajectory, NaN with fewer than two steps).
        """
        means = self.step_means()
        deltas = np.diff(means)
        return {
            "n_tokens": len(self.tokens),
            "n_steps": self.n_steps,
            "base": self.base,
            "mean_entropy": float(self.entropies.mean()),
            "max_entropy": float(self.entropies.max()),
            "min_entropy": float(self.entropies.min()),
            "first_step_mean": float(means[0]),
            "last_step_mean": float(means[-1]),
            "total_decrease": float(means[0] - means[-1]),
            "monotonic_fraction": (float((deltas < 0).mean()) if deltas.size else float("nan")),
        }

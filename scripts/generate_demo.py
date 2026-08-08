#!/usr/bin/env python
"""Generate the README demo image without a live server.

Builds a synthetic-but-realistic CoT answer (structured per-token entropies:
each sentence step opens with higher uncertainty and settles, later steps are
overall more confident) and renders it with entropy_lens.viz.

Run: python scripts/generate_demo.py [docs/assets/demo_trajectory.png]

With a live vLLM server available, prefer scripts/verify_trajectory.py to
produce the same plot from real logprobs.
"""

import sys
from pathlib import Path

import numpy as np

from entropy_lens.trajectory import EntropyTrajectory, split_steps
from entropy_lens.viz import plot_trajectory

ANSWER = (
    "First, multiply 3 boxes by 12 pencils to get 36 pencils. "
    "Then, subtract the 7 pencils Maria gave away. "
    "That leaves 36 minus 7, which is 29. "
    "The answer is 29."
)


def crude_tokenize(text: str) -> list[str]:
    """Whitespace tokenizer with the leading space attached (BPE-style)."""
    words = text.split(" ")
    return [words[0]] + [" " + w for w in words[1:]]


def synth_entropies(tokens: list[str], boundaries: list[int], seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(tokens)
    h = np.empty(n)
    bounds = [*boundaries, n]
    for step, (a, b) in enumerate(zip(bounds[:-1], bounds[1:], strict=False)):
        length = b - a
        step_level = 1.8 * (0.65**step) + 0.15  # later steps more confident
        within = np.linspace(1.6, 0.6, length)  # step opens uncertain, settles
        noise = rng.gamma(2.0, 0.18, length)
        h[a:b] = step_level * within + noise
    return h


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/assets/demo_trajectory.png")
    tokens = crude_tokenize(ANSWER)
    boundaries = split_steps(tokens, "sentence")
    traj = EntropyTrajectory(synth_entropies(tokens, boundaries), tokens, boundaries)

    fig = plot_trajectory(
        traj, title="CoT entropy trajectory (synthetic demo data)", show_tokens=True
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"steps: {traj.n_steps}, tokens: {len(tokens)}")
    for i, (text, mean) in enumerate(zip(traj.step_texts(), traj.step_means(), strict=True), 1):
        print(f"  step {i}: mean H = {mean:.3f} bits | {text.strip()}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()

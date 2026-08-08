"""Plotting for entropy trajectories (requires the ``viz`` extra: matplotlib).

Two building blocks plus one convenience wrapper:

- :func:`plot_token_entropies` — per-token entropy line with step boundaries.
- :func:`plot_step_means` — per-step mean entropy (the CoT-level trajectory).
- :func:`plot_trajectory` — both, stacked in one figure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from entropy_lens.trajectory import EntropyTrajectory

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_LINE = "#4C72B0"
_STEP = "#DD8452"
_BOUNDARY = "#999999"


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for entropy_lens.viz — "
            'install it with: pip install "entropy-lens[viz]"'
        ) from e
    return plt


def plot_token_entropies(
    traj: EntropyTrajectory,
    *,
    ax: Axes | None = None,
    show_tokens: bool = False,
    max_token_labels: int = 60,
) -> Axes:
    """Line plot of per-token entropy with step boundaries as dashed lines.

    Parameters
    ----------
    traj:
        The trajectory to plot.
    ax:
        Existing axes to draw into; a new figure is created when omitted.
    show_tokens:
        Label the x-axis with token strings (skipped when the sequence is
        longer than ``max_token_labels``).
    """
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3.5))

    x = np.arange(len(traj.tokens))
    ax.plot(x, traj.entropies, color=_LINE, lw=1.5, marker="o", ms=3, label="token H")

    for b in traj.step_boundaries[1:]:
        ax.axvline(b - 0.5, color=_BOUNDARY, ls="--", lw=0.8)

    # Per-step mean as horizontal segments over each step's token range.
    for s, mean in zip(traj.step_slices(), traj.step_means(), strict=True):
        ax.hlines(mean, s.start - 0.4, s.stop - 0.6, color=_STEP, lw=2, alpha=0.8)
    ax.hlines([], [], [], color=_STEP, lw=2, label="step mean")  # legend entry

    if show_tokens and len(traj.tokens) <= max_token_labels:
        ax.set_xticks(x)
        ax.set_xticklabels(
            [t.replace("\n", "\\n") for t in traj.tokens],
            rotation=60,
            ha="right",
            fontsize=7,
        )
    else:
        ax.set_xlabel("token index")

    ax.set_ylabel(f"entropy ({traj.base})")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=8)
    return ax


def plot_step_means(traj: EntropyTrajectory, *, ax: Axes | None = None) -> Axes:
    """Per-step mean entropy — the step-level (CoT) trajectory."""
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))

    means = traj.step_means()
    steps = np.arange(1, traj.n_steps + 1)
    ax.plot(steps, means, color=_STEP, lw=2, marker="o", ms=6)

    ax.set_xticks(steps)
    ax.set_xlabel("step")
    ax.set_ylabel(f"mean entropy ({traj.base})")
    ax.set_ylim(bottom=0)
    return ax


def plot_trajectory(
    traj: EntropyTrajectory,
    *,
    title: str | None = None,
    show_tokens: bool = False,
) -> Figure:
    """Stacked figure: per-token entropies on top, step means below."""
    plt = _require_matplotlib()
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 6), height_ratios=[2, 1], constrained_layout=True
    )
    plot_token_entropies(traj, ax=ax_top, show_tokens=show_tokens)
    plot_step_means(traj, ax=ax_bot)
    if title:
        fig.suptitle(title)
    return fig

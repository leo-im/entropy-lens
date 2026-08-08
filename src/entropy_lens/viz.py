"""Plotting for entropy trajectories (requires the ``viz`` extra: matplotlib).

Building blocks plus one convenience wrapper:

- :func:`plot_token_entropies` — per-token entropy line with step boundaries.
- :func:`plot_step_means` — per-step mean entropy (the CoT-level trajectory).
- :func:`plot_hv` — the trajectory as a path in the entropy–varentropy plane.
- :func:`plot_trajectory` — token + step views, stacked in one figure.
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


def plot_hv(
    traj: EntropyTrajectory,
    *,
    level: str = "step",
    ax: Axes | None = None,
) -> Axes:
    """The trajectory as a connected path in the entropy–varentropy plane.

    Entropy (mean surprisal) separates confident from uncertain positions;
    varentropy (variance of surprisal) separates *diffuse* uncertainty from
    *tiered* competition between a few candidates. Plotting (H, V) per step
    (or per token with ``level="token"``) shows which kind of uncertainty
    each part of the generation carried. Requires ``traj.varentropies``.

    No thresholds or decision regions are drawn: entropy-lens measures;
    what counts as "high" is model- and task-dependent and belongs to the
    consumer.
    """
    plt = _require_matplotlib()
    if traj.varentropies is None:
        raise ValueError("plot_hv requires a trajectory with varentropies")
    if level not in ("step", "token"):
        raise ValueError(f'level must be "step" or "token", got {level!r}')
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))

    if level == "step":
        h, v = traj.step_means(), traj.step_varentropy_means()
        labels = [str(i + 1) for i in range(len(h))]
    else:
        h, v = traj.entropies, traj.varentropies
        labels = None

    ax.plot(h, v, color=_BOUNDARY, lw=1.0, ls=":", zorder=1)
    ax.scatter(h, v, s=55 if labels else 18, color=_LINE, zorder=2)
    if labels:
        for hi, vi, lab in zip(h, v, labels, strict=True):
            ax.annotate(lab, (hi, vi), textcoords="offset points", xytext=(6, 5), fontsize=8)

    unit = "bits" if traj.base == "bits" else "nats"
    ax.set_xlabel(f"entropy H ({unit})")
    ax.set_ylabel(f"varentropy V ({unit}²)")
    ax.set_xlim(left=0)
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

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from entropy_lens.trajectory import EntropyTrajectory  # noqa: E402
from entropy_lens.viz import (  # noqa: E402
    plot_step_means,
    plot_token_entropies,
    plot_trajectory,
)


@pytest.fixture
def traj() -> EntropyTrajectory:
    rng = np.random.default_rng(0)
    n = 30
    return EntropyTrajectory(
        entropies=rng.uniform(0.1, 3.0, n),
        tokens=[f"tok{i} " for i in range(n)],
        step_boundaries=[0, 10, 20],
    )


def test_plot_token_entropies(traj):
    ax = plot_token_entropies(traj)
    assert len(ax.lines[0].get_xdata()) == 30
    # 2 interior step boundaries -> 2 dashed vlines.
    assert sum(1 for line in ax.lines if line.get_linestyle() == "--") == 2


def test_plot_token_entropies_with_labels(traj):
    ax = plot_token_entropies(traj, show_tokens=True)
    assert len(ax.get_xticklabels()) == 30


def test_plot_token_entropies_existing_ax(traj):
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    assert plot_token_entropies(traj, ax=ax) is ax


def test_plot_step_means(traj):
    ax = plot_step_means(traj)
    x, y = ax.lines[0].get_xdata(), ax.lines[0].get_ydata()
    np.testing.assert_array_equal(x, [1, 2, 3])
    np.testing.assert_allclose(y, traj.step_means())


def test_plot_trajectory_figure(traj, tmp_path):
    fig = plot_trajectory(traj, title="demo")
    assert len(fig.axes) == 2
    out = tmp_path / "traj.png"
    fig.savefig(out)
    assert out.stat().st_size > 0

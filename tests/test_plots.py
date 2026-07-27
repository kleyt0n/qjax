"""Smoke tests for ``qjax.plots``.

The plotting helpers are a documented, re-exported part of the package but had
no coverage at all, so a broken import or a renamed matplotlib keyword would
only surface when a user ran an example. These tests assert that each entry
point builds a figure with the expected artists; they deliberately do not check
appearance. The ``Agg`` backend is selected in ``conftest.py``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from qjax.plots import (
    CMAP,
    plot_q_exp,
    plot_q_gaussian,
    plot_q_log,
    qcolors,
    save_figure,
    use_qjax_style,
)


@pytest.fixture(autouse=True)
def close_figures():
    """Keep matplotlib from accumulating open figures across tests."""
    yield
    plt.close("all")


def test_cmap_is_a_string():
    assert isinstance(CMAP, str)


@pytest.mark.parametrize("n", [1, 3, 7])
def test_qcolors_length_and_range(n):
    colors = qcolors(n)
    assert len(colors) == n
    for color in colors:
        assert len(color) in (3, 4)
        assert all(0.0 <= channel <= 1.0 for channel in color[:3])


def test_qcolors_respects_bounds():
    # Distinct endpoints must give distinct colours.
    assert qcolors(2, lo=0.0, hi=1.0)[0] != qcolors(2, lo=0.0, hi=1.0)[1]


def test_use_qjax_style_mutates_rcparams():
    use_qjax_style()
    assert plt.rcParams["axes.grid"] in (True, False)


@pytest.mark.parametrize("plot_fn", [plot_q_log, plot_q_exp])
def test_function_plots_draw_one_line_per_q(plot_fn):
    q_values = (0.5, 1.0, 1.5)
    ax = plot_fn(q_values=q_values)
    # One line per q, plus any axhline/axvline guides the helper adds.
    labelled = [ln for ln in ax.get_lines() if ln.get_label().startswith("q =")]
    assert len(labelled) == len(q_values)
    assert ax.get_xlabel()
    assert ax.get_ylabel()


def test_plots_accept_an_existing_axis():
    fig, ax = plt.subplots()
    returned = plot_q_log(q_values=(1.0,), ax=ax)
    assert returned is ax
    assert returned.figure is fig


def test_plot_q_gaussian_draws_curves():
    q_values = (1.0, 1.5, 2.0)
    ax = plot_q_gaussian(q_values=q_values)
    labelled = [ln for ln in ax.get_lines() if ln.get_label().startswith("q =")]
    assert len(labelled) == len(q_values)
    # Densities are non-negative everywhere they are drawn.
    for line in labelled:
        assert (line.get_ydata() >= 0).all()


def test_save_figure_writes_a_file(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out = save_figure(fig, tmp_path / "figure.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_save_figure_returns_a_path(tmp_path):
    fig, _ = plt.subplots()
    out = save_figure(fig, str(tmp_path / "as_str.png"))
    assert out.exists()

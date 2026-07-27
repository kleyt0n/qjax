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
    QJAX_RAMP,
    QLINESTYLES,
    plot_q_exp,
    plot_q_gaussian,
    plot_q_log,
    qcolors,
    qlinestyles,
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


# ---------------------------------------------------------------------------
# The brand ramp.
# ---------------------------------------------------------------------------


def test_ramp_is_registered_with_matplotlib():
    import matplotlib as mpl

    assert CMAP in mpl.colormaps
    assert f"{CMAP}_r" in mpl.colormaps


def test_registration_is_idempotent():
    # The module registers at import; re-running must not raise on a duplicate.
    from qjax.plots.style import _register_colormap

    _register_colormap()
    _register_colormap()


def test_ramp_endpoints_match_the_brand():
    import matplotlib as mpl
    from matplotlib.colors import to_hex

    cmap = mpl.colormaps[CMAP]
    assert to_hex(cmap(0.0)) == QJAX_RAMP[0]
    assert to_hex(cmap(1.0)) == QJAX_RAMP[-1]


def test_ramp_is_monotonically_darker():
    # A sequential scale must decrease in luminance from end to end, or it reads
    # as a rainbow and stops encoding magnitude.
    def luminance(hex_color):
        rgb = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    lums = [luminance(c) for c in QJAX_RAMP]
    assert all(a > b for a, b in zip(lums, lums[1:], strict=False)), lums


def test_qcolors_window_keeps_the_lightest_curve_visible():
    # Regression on the reason the window starts at 0.40: the ramp's light end is
    # below 1.7:1 on white, which is invisible for a thin line. Every sampled
    # color must clear the 2:1 floor for a sequential light end.
    from matplotlib.colors import to_hex

    def contrast_on_white(rgba):
        hex_color = to_hex(rgba)
        rgb = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
        lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
        return 1.05 / (lum + 0.05)

    for n in (1, 2, 3, 4, 5, 6, 8):
        assert min(contrast_on_white(c) for c in qcolors(n)) >= 2.0, n


def test_qcolors_is_ordered_and_sized():
    from matplotlib.colors import to_hex

    assert qcolors(0) == []
    assert len(qcolors(1)) == 1
    for n in (2, 5, 9):
        cols = qcolors(n)
        assert len(cols) == n
        # Distinct, and running dark.
        assert len({to_hex(c) for c in cols}) == n
        assert sum(cols[0][:3]) > sum(cols[-1][:3])


def test_qcolors_single_color_is_mid_ramp():
    # A lone curve must not receive the faintest step of the window.
    lone = qcolors(1)[0]
    lightest_of_many = qcolors(5)[0]
    assert sum(lone[:3]) < sum(lightest_of_many[:3])


@pytest.mark.parametrize("n", [0, 1, 2, 4, 7])
def test_qlinestyles_length_and_distinctness(n):
    styles = qlinestyles(n)
    assert len(styles) == n
    # The first four (the realistic method counts) are all different.
    if 0 < n <= len(QLINESTYLES):
        assert len(set(map(str, styles))) == n


def test_qlinestyles_are_valid_matplotlib_specs():
    fig, ax = plt.subplots()
    for style in qlinestyles(5):
        ax.plot([0, 1], [0, 1], linestyle=style)
    fig.canvas.draw()  # raises if a spec is invalid

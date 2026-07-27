"""The public API surface.

Nothing previously imported ``qjax`` at the top level or asserted what it
exports, so a re-export could be dropped or renamed without a test noticing.
"""

from __future__ import annotations

import importlib.metadata

import pytest

import qjax

EXPECTED_EXPORTS = {
    "q_log",
    "q_exp",
    "q_add",
    "q_diff",
    "q_prod",
    "q_div",
    "tsallis_entropy",
    "tsallis_cross_entropy",
    "tsallis_divergence",
    "normalization",
    "q_gaussian_pdf",
    "q_gaussian_logpdf",
    "sample",
    "tsallis_entmax",
}


def test_all_matches_expected_surface():
    assert set(qjax.__all__) == EXPECTED_EXPORTS | {"__version__"}


@pytest.mark.parametrize("name", sorted(EXPECTED_EXPORTS))
def test_every_export_is_importable_and_callable(name):
    assert callable(getattr(qjax, name))


def test_version_is_exposed_and_matches_metadata():
    assert isinstance(qjax.__version__, str)
    assert qjax.__version__.count(".") >= 2
    # Single source of truth: the installed distribution metadata is derived
    # from qjax.__version__, so the two must never drift.
    assert importlib.metadata.version("qjax") == qjax.__version__


def test_importing_qjax_does_not_require_matplotlib(monkeypatch):
    # The core maths must stay usable without the optional plotting extra.
    import sys

    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    for module in [m for m in sys.modules if m.startswith("qjax")]:
        monkeypatch.delitem(sys.modules, module, raising=False)

    import qjax as reloaded

    assert callable(reloaded.q_log)


def test_package_ships_type_information():
    from pathlib import Path

    assert (Path(qjax.__file__).parent / "py.typed").exists()

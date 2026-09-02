"""Shared pytest configuration.

Most test modules enable float64 at import so that identities can be asserted to
near machine precision. That is *not* what a user gets by default -- JAX runs in
float32 unless told otherwise -- so the ``float32`` marker below runs a subset of
the suite in the default precision, where cancellation near ``q = 1`` actually
bites.
"""

from __future__ import annotations

import jax
import matplotlib
import pytest

# Plot tests must not try to open a window.
matplotlib.use("Agg")


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "float32: run in JAX's default single precision rather than x64"
    )


@pytest.fixture
def float32_mode():
    """Temporarily disable x64 so a test sees the default user precision."""
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _load_example(name):
    """Import an example script by path, without executing its ``main``."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_example_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def example():
    """Load an example module by name, memoized for the session.

    The four statistical-physics examples carry claims that are worth gating in
    CI -- most of all the ``q``-deformed free-energy gradient, whose whole point
    is that autodiff reproduces the analytic REINFORCE estimator. The library
    kernels they use are tested directly in ``tests/test_physics_*.py``; this
    fixture is for the parts that live in the scripts.
    """
    cache: dict = {}

    def load(name):
        if name not in cache:
            cache[name] = _load_example(name)
        return cache[name]

    return load

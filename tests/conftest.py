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

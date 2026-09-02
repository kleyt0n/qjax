r"""The Sherrington-Kirkpatrick spin glass: Hamiltonian and exact thermodynamics.

The SK model is the mean-field spin glass,

$$H(s) = -\tfrac12 \sum_{i \neq j} J_{ij} s_i s_j,
\qquad J_{ij} = J_{ji} \sim \mathcal N(0, 1/N),$$

with a rugged, frustrated landscape whose ground-state energy per spin tends to
Parisi's $-0.7633$ as $N \to \infty$. It is the standard hard case for
variational methods: a distribution over $2^N$ states that concentrates on a
few of them scores well on the naive objective while getting the physics wrong,
which is why the free energy here is always reported *against* an exact value.

For $N \le 22$ the full state space is enumerable, so the free energy,
internal energy and two-point correlations are available exactly. The
enumeration is streamed in chunks with a running ``logsumexp``, so the memory
cost is set by the chunk size rather than by $2^N$.

References:
    Sherrington, D. & Kirkpatrick, S. (1975). *Phys. Rev. Lett.* **35**, 1792.
    Parisi, G. (1980). *J. Phys. A* **13**, L115.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from qjax.shared.types import Array, Scalar

#: Largest system whose full state space `sk_exact_observables` will enumerate.
MAX_ENUMERATED_SPINS: int = 22


def sk_couplings(key: jax.Array, num_spins: int) -> jax.Array:
    """Draw a symmetric SK coupling matrix with zero diagonal.

    Off-diagonal entries are Gaussian with variance ``1 / num_spins``, the
    scaling that makes the energy per spin intensive.

    Args:
        key: PRNG key.
        num_spins: Number of spins ``N``.

    Returns:
        A symmetric ``(N, N)`` matrix with zeros on the diagonal.
    """
    raw = jax.random.normal(key, (num_spins, num_spins)) / jnp.sqrt(num_spins)
    upper = jnp.triu(raw, 1)
    return upper + upper.T


def sk_energy(spins: Array, couplings: Array) -> jax.Array:
    r"""SK energy $-\tfrac12 s^{\mathsf T} J s$.

    Args:
        spins: Configuration(s) of shape ``(..., N)`` with entries in
            ``{-1, +1}``.
        couplings: Symmetric ``(N, N)`` coupling matrix with zero diagonal.

    Returns:
        Total energy per configuration, shape ``(...)``.
    """
    spins = jnp.asarray(spins, dtype=jnp.result_type(float))
    couplings = jnp.asarray(couplings, dtype=jnp.result_type(float))
    return -0.5 * jnp.einsum("...i,ij,...j->...", spins, couplings, spins)


def _chunk_configurations(offset: jax.Array, chunk: int, num_spins: int) -> jax.Array:
    """Materialize ``chunk`` consecutive configurations starting at ``offset``."""
    states = offset + jnp.arange(chunk, dtype=jnp.int32)
    bits = (states[:, None] >> jnp.arange(num_spins, dtype=jnp.int32)[None, :]) & 1
    return 1.0 - 2.0 * bits.astype(jnp.result_type(float))


def _largest_power_of_two(value: int, ceiling: int) -> int:
    """Largest power of two that is ``<= value`` and ``<= ceiling``."""
    capped = min(max(value, 1), ceiling)
    return 1 << (capped.bit_length() - 1)


def sk_exact_observables(
    couplings: Array, temperature: Scalar, chunk: int = 4096
) -> dict[str, jax.Array]:
    r"""Exact SK thermodynamics by streamed exhaustive enumeration.

    Accumulates $Z$, $\langle E \rangle$ and $\langle s_i s_j \rangle$
    over all $2^N$ configurations in chunks, rescaling the running sums
    whenever a new maximum log-weight appears. This is a numerically exact
    ``logsumexp`` at ``O(chunk)`` memory.

    Args:
        couplings: Symmetric ``(N, N)`` coupling matrix, ``N <=``
            `MAX_ENUMERATED_SPINS`.
        temperature: Temperature ``T``.
        chunk: Configurations per chunk; rounded down to a power of two and
            capped at ``2**N``.

    Returns:
        A dict with ``log_z``, ``free_energy_per_spin``, ``energy_per_spin``,
        ``ground_state_energy_per_spin`` and ``correlations`` (an ``(N, N)``
        array of $\langle s_i s_j \rangle$).

    Raises:
        ValueError: If ``N`` exceeds `MAX_ENUMERATED_SPINS`.
    """
    couplings = jnp.asarray(couplings, dtype=jnp.result_type(float))
    num_spins = couplings.shape[-1]
    if num_spins > MAX_ENUMERATED_SPINS:
        raise ValueError(
            f"enumerating {num_spins} spins needs 2**{num_spins} states; "
            f"the limit is {MAX_ENUMERATED_SPINS}."
        )
    total = 2**num_spins
    chunk = _largest_power_of_two(chunk, total)
    beta = 1.0 / jnp.asarray(temperature, dtype=jnp.result_type(float))

    Carry = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]

    def step(carry: Carry, offset: jax.Array) -> tuple[Carry, None]:
        peak, mass, energy_mass, correlation_mass, minimum = carry
        spins = _chunk_configurations(offset, chunk, num_spins)
        energies = sk_energy(spins, couplings)
        log_weights = -beta * energies

        new_peak = jnp.maximum(peak, jnp.max(log_weights))
        rescale = jnp.exp(peak - new_peak)
        weights = jnp.exp(log_weights - new_peak)
        return (
            new_peak,
            mass * rescale + jnp.sum(weights),
            energy_mass * rescale + jnp.sum(weights * energies),
            correlation_mass * rescale + jnp.einsum("bi,bj,b->ij", spins, spins, weights),
            jnp.minimum(minimum, jnp.min(energies)),
        ), None

    zero = jnp.zeros((), dtype=jnp.result_type(float))
    initial: Carry = (
        jnp.full((), -jnp.inf, dtype=jnp.result_type(float)),
        zero,
        zero,
        jnp.zeros((num_spins, num_spins), dtype=jnp.result_type(float)),
        jnp.full((), jnp.inf, dtype=jnp.result_type(float)),
    )
    offsets = jnp.arange(0, total, chunk, dtype=jnp.int32)
    (peak, mass, energy_mass, correlation_mass, minimum), _ = jax.lax.scan(step, initial, offsets)

    log_z = peak + jnp.log(mass)
    return {
        "log_z": log_z,
        "free_energy_per_spin": -log_z / (beta * num_spins),
        "energy_per_spin": energy_mass / mass / num_spins,
        "ground_state_energy_per_spin": minimum / num_spins,
        "correlations": correlation_mass / mass,
    }


def sk_exact_correlations(couplings: Array, temperature: Scalar, chunk: int = 4096) -> jax.Array:
    r"""Exact two-point correlations $\langle s_i s_j \rangle$ by enumeration.

    A thin wrapper around `sk_exact_observables`. The correlation matrix is
    the sharpest diagnostic of variational mode collapse: a distribution that
    has collapsed onto one configuration reports ``+/-1`` everywhere, however
    good its free energy looks.

    Args:
        couplings: Symmetric ``(N, N)`` coupling matrix.
        temperature: Temperature ``T``.
        chunk: Configurations per enumeration chunk.

    Returns:
        An ``(N, N)`` array of two-point correlations.
    """
    return sk_exact_observables(couplings, temperature, chunk)["correlations"]


__all__ = [
    "MAX_ENUMERATED_SPINS",
    "sk_couplings",
    "sk_energy",
    "sk_exact_observables",
    "sk_exact_correlations",
]

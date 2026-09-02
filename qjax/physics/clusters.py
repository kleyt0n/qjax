r"""Lennard-Jones atomic clusters: potential, local quenching, and geometry.

The pair potential

$$V(r) = 4\epsilon\Big[(\sigma/r)^{12} - (\sigma/r)^{6}\Big]$$

is the standard benchmark landscape for global optimization: an $n$-atom
cluster has a number of local minima growing roughly exponentially in $n$
(around $10^8$ already at $n = 20$), so the number of basins, not the
dimension, is what makes it hard. Two properties make it a *verifiable*
benchmark rather than a demo:

- For $n \le 4$ the global minimum is a closed form. A regular simplex with
  every edge at $r = 2^{1/6}\sigma$ puts every pair exactly at the potential
  minimum, contributing $-\epsilon$ each: $-1$, $-3$, $-6$.
- For larger $n$ the global minima are tabulated to six decimals in the
  Cambridge Cluster Database (see `qjax.physics.reference`), including the
  double-funnel case $n = 38$.

References:
    Wales, D. J. & Doye, J. P. K. (1997). *J. Phys. Chem. A* **101**, 5111.
    Doye, J. P. K., Miller, M. A. & Wales, D. J. (1999).
        *J. Chem. Phys.* **110**, 6896.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from qjax.shared.types import Array, Scalar


def lj_energy(
    positions: Array,
    epsilon: Scalar = 1.0,
    sigma: Scalar = 1.0,
    softening: float = 1e-12,
) -> jax.Array:
    r"""Total Lennard-Jones energy of a cluster, summed over unordered pairs.

    The $r^{-12}$ term diverges as two atoms coincide, so the squared
    separations are floored at ``softening`` *and* the diagonal is replaced
    before the power rather than after: a bare ``1 / 0`` would back-propagate
    ``NaN`` into every coordinate even though the diagonal is masked out of the
    sum.

    Args:
        positions: Atomic coordinates of shape ``(..., n, 3)``.
        epsilon: Well depth.
        sigma: Length scale; the pair minimum sits at ``2**(1/6) * sigma``.
        softening: Floor on the squared separation.

    Returns:
        Total energy per cluster, shape ``(...)``. Equals ``-1``, ``-3``, ``-6``
        for a regular simplex of 2, 3, 4 atoms at ``r = 2**(1/6) sigma``.
    """
    positions = jnp.asarray(positions, dtype=jnp.result_type(float))
    epsilon = jnp.asarray(epsilon, dtype=jnp.result_type(float))
    sigma = jnp.asarray(sigma, dtype=jnp.result_type(float))

    offsets = positions[..., :, None, :] - positions[..., None, :, :]
    squared = jnp.sum(offsets**2, axis=-1)
    off_diagonal = ~jnp.eye(positions.shape[-2], dtype=bool)

    safe = jnp.where(off_diagonal, jnp.maximum(squared, softening), 1.0)
    inverse_sixth = (sigma**2 / safe) ** 3
    pair = 4.0 * epsilon * (inverse_sixth**2 - inverse_sixth)
    return 0.5 * jnp.sum(jnp.where(off_diagonal, pair, 0.0), axis=(-2, -1))


def lj_energy_confined(
    positions: Array,
    container_radius: Scalar,
    stiffness: Scalar = 10.0,
    epsilon: Scalar = 1.0,
    sigma: Scalar = 1.0,
) -> jax.Array:
    r"""Lennard-Jones energy plus a soft spherical wall.

    A cluster in free space evaporates: an atom kicked far enough away feels no
    restoring force, and its energy contribution goes to zero rather than to
    something unphysical, so a global search will happily "solve" the problem by
    losing atoms. The wall $k \sum_i [\,|x_i| - R\,]_+^2$ is zero inside the
    container, so a minimum found strictly inside is a minimum of the *bare*
    potential -- which the examples assert rather than assume.

    Args:
        positions: Atomic coordinates of shape ``(..., n, 3)``.
        container_radius: Radius ``R`` beyond which the wall acts.
        stiffness: Wall stiffness ``k``.
        epsilon: Well depth.
        sigma: Length scale.

    Returns:
        Confined energy per cluster, shape ``(...)``. Differentiable everywhere,
        including at the origin.
    """
    positions = jnp.asarray(positions, dtype=jnp.result_type(float))
    # ``sqrt`` has an infinite derivative at zero, and the ``maximum`` below then
    # multiplies it by zero -- so an atom sitting exactly at the origin (the
    # centre of a re-centred icosahedron, say) would give a finite energy and a
    # NaN gradient. Sanitize the radicand before the root, as ``lj_energy`` does.
    squared_radius = jnp.sum(positions**2, axis=-1)
    inside = squared_radius > 0.0
    radius = jnp.where(inside, jnp.sqrt(jnp.where(inside, squared_radius, 1.0)), 0.0)
    excess = jnp.maximum(radius - jnp.asarray(container_radius, dtype=radius.dtype), 0.0)
    wall = jnp.asarray(stiffness, dtype=radius.dtype) * jnp.sum(excess**2, axis=-1)
    return lj_energy(positions, epsilon, sigma) + wall


def lj_random_cluster(key: jax.Array, num_atoms: int, radius: Scalar) -> jax.Array:
    """Draw atomic positions uniformly inside a ball.

    Args:
        key: PRNG key.
        num_atoms: Number of atoms ``n``.
        radius: Ball radius.

    Returns:
        Positions of shape ``(n, 3)``.
    """
    direction_key, magnitude_key = jax.random.split(key)
    direction = jax.random.normal(direction_key, (num_atoms, 3))
    direction /= jnp.linalg.norm(direction, axis=-1, keepdims=True)
    # u**(1/3) makes the radial law uniform in volume.
    magnitude = jax.random.uniform(magnitude_key, (num_atoms, 1)) ** (1.0 / 3.0)
    return direction * magnitude * jnp.asarray(radius, dtype=jnp.result_type(float))


def equidistant_cluster(num_atoms: int, distance: Scalar) -> jax.Array:
    """A regular simplex of 1 to 4 atoms with every pair at ``distance``.

    These are the closed-form Lennard-Jones global minima: at
    ``distance = 2**(1/6) sigma`` every pair sits exactly at the potential
    minimum, so the energy is ``-epsilon`` times the number of pairs.

    Args:
        num_atoms: Number of atoms, 1 to 4.
        distance: Edge length.

    Returns:
        Positions of shape ``(num_atoms, 3)``.

    Raises:
        ValueError: If ``num_atoms`` is outside 1 to 4 (a regular simplex with
            more vertices does not embed in three dimensions).
    """
    if not 1 <= num_atoms <= 4:
        raise ValueError(f"a regular simplex in 3-D has at most 4 vertices; got {num_atoms}.")
    third = jnp.sqrt(jnp.asarray(3.0, dtype=jnp.result_type(float)))
    vertices = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, float(third / 2.0), 0.0],
            [0.5, float(third / 6.0), float(jnp.sqrt(jnp.asarray(2.0 / 3.0)))],
        ],
        dtype=jnp.result_type(float),
    )
    return vertices[:num_atoms] * jnp.asarray(distance, dtype=jnp.result_type(float))


def lj_quench(
    positions: Array,
    steps: int = 40,
    learning_rate: Scalar = 2e-3,
    epsilon: Scalar = 1.0,
    sigma: Scalar = 1.0,
) -> tuple[jax.Array, jax.Array]:
    r"""Locally minimize the bare Lennard-Jones energy with Adam.

    This is the "quench" of Wales-Doye basin-hopping: it maps a proposed
    configuration onto the bottom of the basin it fell into, so a Monte Carlo
    walk explores the graph of local minima rather than the raw landscape.
    Adam rather than plain gradient descent because the $r^{-12}$ core makes
    the gradient scale vary by orders of magnitude across the cluster; the price
    is that the energy trace is not guaranteed monotone.

    Args:
        positions: Starting coordinates of shape ``(n, 3)``.
        steps: Number of Adam steps.
        learning_rate: Adam step size.
        epsilon: Well depth.
        sigma: Length scale.

    Returns:
        ``(final_positions, energies)`` where ``energies`` has shape
        ``(steps + 1,)`` and holds the energy before each step and after the
        last.
    """
    positions = jnp.asarray(positions, dtype=jnp.result_type(float))
    gradient = jax.grad(lambda x: lj_energy(x, epsilon, sigma))
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    Carry = tuple[jax.Array, jax.Array, jax.Array]

    def step(carry: Carry, count: jax.Array) -> tuple[Carry, jax.Array]:
        state, first, second = carry
        energy = lj_energy(state, epsilon, sigma)
        grads = gradient(state)
        first = beta1 * first + (1.0 - beta1) * grads
        second = beta2 * second + (1.0 - beta2) * grads**2
        scale = count + 1.0
        first_hat = first / (1.0 - beta1**scale)
        second_hat = second / (1.0 - beta2**scale)
        state = state - learning_rate * first_hat / (jnp.sqrt(second_hat) + eps)
        return (state, first, second), energy

    initial: Carry = (positions, jnp.zeros_like(positions), jnp.zeros_like(positions))
    (final, _, _), trace = jax.lax.scan(
        step, initial, jnp.arange(steps, dtype=jnp.result_type(float))
    )
    energies = jnp.concatenate([trace, lj_energy(final, epsilon, sigma)[None]])
    return final, energies


def coordination_numbers(positions: Array, cutoff: Scalar = 1.35) -> jax.Array:
    """Count neighbours within ``cutoff`` of each atom.

    Distinguishes surface atoms from core atoms, which is what makes an fcc
    truncated octahedron visually distinguishable from an icosahedron.

    Args:
        positions: Atomic coordinates of shape ``(..., n, 3)``.
        cutoff: Neighbour cutoff, in the same units as ``positions``. The
            default sits between the first and second neighbour shells of a
            close-packed cluster at ``sigma = 1``.

    Returns:
        Integer neighbour counts of shape ``(..., n)``.
    """
    positions = jnp.asarray(positions, dtype=jnp.result_type(float))
    offsets = positions[..., :, None, :] - positions[..., None, :, :]
    squared = jnp.sum(offsets**2, axis=-1)
    off_diagonal = ~jnp.eye(positions.shape[-2], dtype=bool)
    within = (squared < jnp.asarray(cutoff, dtype=squared.dtype) ** 2) & off_diagonal
    return jnp.sum(within, axis=-1)


__all__ = [
    "lj_energy",
    "lj_energy_confined",
    "lj_random_cluster",
    "equidistant_cluster",
    "lj_quench",
    "coordination_numbers",
]

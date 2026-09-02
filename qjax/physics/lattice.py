r"""The 2-D Ising model: Hamiltonian, Metropolis sampler, and exact references.

This module supplies the physical system that the Tsallis machine-learning
examples are tested *against*. It is deliberately small and exact-first: every
sampled quantity has at least one independent closed-form or exhaustive
counterpart in the same file, so a sampler bug shows up as a disagreement rather
than as a plausible-looking curve.

Two Monte Carlo updates are provided: a local checkerboard Metropolis sweep,
and the Wolff single-cluster update, which has no critical slowing down and is
what makes a finite-size-scaling study at $T_c$ trustworthy.

The Hamiltonian is the nearest-neighbour Ising model on an $L \times L$ square
lattice with periodic boundaries,

$$H(s) = -J \sum_{\langle i j \rangle} s_i s_j, \qquad s_i \in \{-1, +1\},$$

whose critical point is known exactly: $T_c = 2 J / \ln(1 + \sqrt 2)$
(Onsager, 1944), with $\nu = 1$, $\beta = 1/8$ and $u(T_c) = -\sqrt 2 J$.

Three mutually independent routes to the exact free energy are provided, which
is what makes the validation credible:

1. `ising_exact_observables` -- exhaustive enumeration of all $2^{L^2}$
   states, for $L \le 4$.
2. `ising_transfer_matrix_log_z` -- the $2^L \times 2^L$ transfer matrix,
   exact for the *finite* periodic lattice, for $L \le 10$.
3. `onsager_free_energy_per_site` -- Onsager's thermodynamic-limit solution.

References:
    Onsager, L. (1944). *Phys. Rev.* **65**, 117.
    Metropolis, N. et al. (1953). *J. Chem. Phys.* **21**, 1087.
    Wolff, U. (1989). *Phys. Rev. Lett.* **62**, 361.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from qjax.shared.types import Array, Scalar

#: Largest number of spins whose full state space is enumerated by
#: `ising_exact_observables`. It admits every lattice up to ``L = 4``
#: (``2**16 = 65536`` states); the headroom to 20 is for non-square use.
MAX_ENUMERATED_SPINS: int = 20

#: Largest lattice handled by `ising_transfer_matrix_log_z` (a
#: ``1024 x 1024`` dense eigendecomposition at ``L = 10``).
MAX_TRANSFER_SIZE: int = 10


def neighbour_sum(spins: Array) -> jax.Array:
    """Sum of the four nearest neighbours of every site, with periodic wrap.

    Args:
        spins: Spin configuration(s) of shape ``(..., L, L)``.

    Returns:
        An array of the same shape whose entry ``[..., i, j]`` is the sum of the
        spins at the four sites adjacent to ``(i, j)``.
    """
    spins = jnp.asarray(spins, dtype=jnp.result_type(float))
    return (
        jnp.roll(spins, 1, axis=-1)
        + jnp.roll(spins, -1, axis=-1)
        + jnp.roll(spins, 1, axis=-2)
        + jnp.roll(spins, -1, axis=-2)
    )


def ising_energy(spins: Array, coupling: Scalar = 1.0) -> jax.Array:
    r"""Total Ising energy $-J \sum_{\langle ij \rangle} s_i s_j$.

    Each bond is counted once: the site-wise sum $\sum_i s_i n_i$ visits
    every bond twice, hence the factor $1/2$.

    Args:
        spins: Spin configuration(s) of shape ``(..., L, L)``.
        coupling: Exchange coupling ``J``. Positive is ferromagnetic.

    Returns:
        Total energy per configuration, shape ``(...)``. For the all-aligned
        state this is ``-2 J L**2``.
    """
    spins = jnp.asarray(spins, dtype=jnp.result_type(float))
    pairs = jnp.sum(spins * neighbour_sum(spins), axis=(-2, -1))
    return -0.5 * jnp.asarray(coupling, dtype=jnp.result_type(float)) * pairs


def ising_magnetization(spins: Array) -> jax.Array:
    """Signed magnetization per site, ``mean(s)``.

    Args:
        spins: Spin configuration(s) of shape ``(..., L, L)``.

    Returns:
        Magnetization per site, shape ``(...)``, in ``[-1, 1]``.
    """
    spins = jnp.asarray(spins, dtype=jnp.result_type(float))
    return jnp.mean(spins, axis=(-2, -1))


def checkerboard_sweep(
    key: jax.Array, spins: jax.Array, beta: Scalar, coupling: Scalar = 1.0
) -> jax.Array:
    """One Metropolis sweep, updating the two lattice sublattices in turn.

    On a bipartite lattice the neighbours of every site lie entirely in the
    other sublattice, so all sites of one colour can be proposed in parallel
    without breaking detailed balance. Two half-updates make one full sweep.

    Args:
        key: PRNG key.
        spins: A single configuration of shape ``(L, L)``.
        beta: Inverse temperature ``1 / T`` (a scalar; ``vmap`` for a batch).
        coupling: Exchange coupling ``J``.

    Returns:
        The configuration after one sweep, shape ``(L, L)``.
    """
    beta = jnp.asarray(beta, dtype=jnp.result_type(float))
    coupling = jnp.asarray(coupling, dtype=jnp.result_type(float))
    size = spins.shape[-1]
    index = jnp.arange(size)
    parity = (index[:, None] + index[None, :]) % 2

    for colour in (0, 1):
        key, subkey = jax.random.split(key)
        # Flipping s_i costs Delta E = 2 J s_i n_i.
        delta = 2.0 * coupling * spins * neighbour_sum(spins)
        # min(1, exp(-beta dE)); the clamp keeps exp() from overflowing to inf
        # for strongly uphill moves, which is a no-op for the comparison.
        accept_prob = jnp.exp(jnp.minimum(-beta * delta, 0.0))
        accepted = jax.random.uniform(subkey, spins.shape) < accept_prob
        spins = jnp.where(accepted & (parity == colour), -spins, spins)
    return spins


def metropolis_chain(
    key: jax.Array, spins: jax.Array, beta: Scalar, sweeps: int, coupling: Scalar = 1.0
) -> jax.Array:
    """Run ``sweeps`` Metropolis sweeps and return the final configuration.

    Args:
        key: PRNG key.
        spins: Initial configuration of shape ``(L, L)``.
        beta: Inverse temperature ``1 / T`` (a scalar; ``vmap`` for a batch).
        sweeps: Number of full sweeps (a Python int; it sets the scan length).
        coupling: Exchange coupling ``J``.

    Returns:
        The configuration after ``sweeps`` sweeps, shape ``(L, L)``.
    """
    Carry = tuple[jax.Array, jax.Array]

    def step(carry: Carry, _: None) -> tuple[Carry, None]:
        chain_key, state = carry
        chain_key, subkey = jax.random.split(chain_key)
        return (chain_key, checkerboard_sweep(subkey, state, beta, coupling)), None

    (_, final), _ = jax.lax.scan(step, (key, spins), None, length=sweeps)
    return final


def wolff_update(
    key: jax.Array, spins: jax.Array, beta: Scalar, coupling: Scalar = 1.0
) -> jax.Array:
    r"""One Wolff single-cluster update.

    A seed site is chosen uniformly, a cluster is grown outward through bonds
    between *equal* spins with probability $p = 1 - e^{-2\beta J}$ each, and the
    whole cluster is flipped. The move is accepted unconditionally: $p$ is
    exactly the value that makes the cluster-construction and Boltzmann factors
    cancel.

    Why it is here rather than only `checkerboard_sweep`: a local update's
    autocorrelation time grows as $L^{z}$ with $z \approx 2.17$ at $T_c$,
    while a cluster update has no such critical slowing down. Note that this
    buys *decorrelation*, not a shortcut to equilibrium from a cold start: one
    update flips one cluster, so a chain started from a random configuration
    still needs enough updates for the cluster to have swept the lattice --
    measured at $T_c$, ``L = 32`` is nowhere near equilibrium after 40 updates
    and settled by about 120. Local sweeps relax a random start more evenly;
    cluster updates decorrelate an equilibrated one far better.

    The cluster is grown as a boolean mask in a `jax.lax.while_loop`: every
    iteration draws one fresh uniform per *bond*, so two frontier sites adjacent
    to the same candidate test their bonds independently, as the algorithm
    requires. The loop is data-dependent, so this is jittable but not
    reverse-differentiable -- which a Monte Carlo update never needs to be.

    Args:
        key: PRNG key.
        spins: A single configuration of shape ``(L, L)``.
        beta: Inverse temperature ``1 / T`` (a scalar; ``vmap`` for a batch).
        coupling: Exchange coupling ``J``, positive (ferromagnetic).

    Returns:
        The configuration after one cluster flip, shape ``(L, L)``.
    """
    beta = jnp.asarray(beta, dtype=jnp.result_type(float))
    coupling = jnp.asarray(coupling, dtype=jnp.result_type(float))
    size = spins.shape[-1]
    add_probability = -jnp.expm1(-2.0 * beta * coupling)

    seed_key, grow_key = jax.random.split(key)
    seed = jax.random.randint(seed_key, (2,), 0, size)
    start = jnp.zeros((size, size), dtype=bool).at[seed[0], seed[1]].set(True)
    aligned = spins == spins[seed[0], seed[1]]

    Carry = tuple[jax.Array, jax.Array, jax.Array]

    def growing(carry: Carry) -> jax.Array:
        _, _, frontier = carry
        return jnp.any(frontier)

    def grow(carry: Carry) -> Carry:
        grow_key, cluster, frontier = carry
        grow_key, right_key, down_key = jax.random.split(grow_key, 3)
        # One uniform per bond per iteration: ``right[i, j]`` is the bond from
        # (i, j) to (i, j+1) and ``down[i, j]`` the bond to (i+1, j).
        right = jax.random.uniform(right_key, (size, size)) < add_probability
        down = jax.random.uniform(down_key, (size, size)) < add_probability
        reached = (
            jnp.roll(frontier & right, 1, axis=-1)  # bond crossed rightward
            | (jnp.roll(frontier, -1, axis=-1) & right)  # ... and leftward
            | jnp.roll(frontier & down, 1, axis=-2)  # downward
            | (jnp.roll(frontier, -1, axis=-2) & down)  # upward
        )
        accepted = reached & aligned & ~cluster
        return grow_key, cluster | accepted, accepted

    _, cluster, _ = jax.lax.while_loop(growing, grow, (grow_key, start, start))
    return jnp.where(cluster, -spins, spins)


def wolff_chain(
    key: jax.Array, spins: jax.Array, beta: Scalar, updates: int, coupling: Scalar = 1.0
) -> jax.Array:
    """Run ``updates`` Wolff cluster updates and return the final configuration.

    Args:
        key: PRNG key.
        spins: Initial configuration of shape ``(L, L)``.
        beta: Inverse temperature ``1 / T`` (a scalar; ``vmap`` for a batch).
        updates: Number of cluster updates (a Python int; it sets the scan
            length).
        coupling: Exchange coupling ``J``.

    Returns:
        The configuration after ``updates`` cluster flips, shape ``(L, L)``.
    """
    Carry = tuple[jax.Array, jax.Array]

    def step(carry: Carry, _: None) -> tuple[Carry, None]:
        chain_key, state = carry
        chain_key, subkey = jax.random.split(chain_key)
        return (chain_key, wolff_update(subkey, state, beta, coupling)), None

    (_, final), _ = jax.lax.scan(step, (key, spins), None, length=updates)
    return final


def sample_ising(
    key: jax.Array,
    size: int,
    temperatures: Array,
    num_samples: int,
    sweeps: int,
    coupling: Scalar = 1.0,
    algorithm: str = "metropolis",
) -> jax.Array:
    r"""Draw equilibrium configurations at each of several temperatures.

    Every sample gets its *own* independent chain, started from a random
    configuration. Critical slowing down therefore affects only how long each
    chain must run to equilibrate, never the independence of the samples -- so
    no decorrelation sweeps between samples are needed.

    How long "long enough" is depends on the update, and the two available here
    fail in opposite directions: Metropolis sweeps relax a random start evenly
    but decorrelate slowly at $T_c$ ($\tau \sim L^{2.17}$), while Wolff
    cluster updates decorrelate without critical slowing down but need enough
    updates to have touched the whole lattice first. Both are checked against
    exhaustive enumeration in the test suite.

    Args:
        key: PRNG key.
        size: Linear lattice size ``L``.
        temperatures: Temperatures to sample at, shape ``(T,)``.
        num_samples: Independent configurations per temperature.
        sweeps: Equilibration steps per chain (a Python int) -- Metropolis
            sweeps, or Wolff cluster updates when ``algorithm="wolff"``.
        coupling: Exchange coupling ``J``.
        algorithm: ``"metropolis"`` for `checkerboard_sweep`, ``"wolff"`` for
            `wolff_update`.

    Returns:
        Configurations of shape ``(T, num_samples, L, L)`` with entries in
        ``{-1.0, +1.0}``.

    Raises:
        ValueError: If ``algorithm`` is neither ``"metropolis"`` nor
            ``"wolff"``.
    """
    if algorithm not in ("metropolis", "wolff"):
        raise ValueError(f"algorithm must be 'metropolis' or 'wolff'; got {algorithm!r}.")
    chain = metropolis_chain if algorithm == "metropolis" else wolff_chain
    temperatures = jnp.asarray(temperatures, dtype=jnp.result_type(float))
    num_temperatures = temperatures.shape[0]
    num_chains = num_temperatures * num_samples

    keys = jax.random.split(key, num_chains + 1)
    start = jax.random.bernoulli(keys[0], 0.5, (num_chains, size, size))
    initial = jnp.where(start, 1.0, -1.0)
    betas = jnp.repeat(1.0 / temperatures, num_samples)

    run = jax.vmap(lambda k, s, b: chain(k, s, b, sweeps, coupling))
    final = run(keys[1:], initial, betas)
    return final.reshape(num_temperatures, num_samples, size, size)


def ising_all_configurations(size: int) -> jax.Array:
    """Enumerate every spin configuration of an ``L x L`` lattice.

    Args:
        size: Linear lattice size ``L``; ``L**2`` must not exceed
            `MAX_ENUMERATED_SPINS`.

    Returns:
        All configurations, shape ``(2**(L*L), L, L)``, entries in
        ``{-1.0, +1.0}``.

    Raises:
        ValueError: If ``L**2`` exceeds `MAX_ENUMERATED_SPINS`.
    """
    num_spins = size * size
    if num_spins > MAX_ENUMERATED_SPINS:
        raise ValueError(
            f"enumerating {num_spins} spins needs 2**{num_spins} states; "
            f"the limit is {MAX_ENUMERATED_SPINS}."
        )
    states = jnp.arange(2**num_spins, dtype=jnp.int32)
    bits = (states[:, None] >> jnp.arange(num_spins, dtype=jnp.int32)[None, :]) & 1
    spins = 1.0 - 2.0 * bits.astype(jnp.result_type(float))
    return spins.reshape(-1, size, size)


def ising_boltzmann_probabilities(
    size: int, temperature: Scalar, coupling: Scalar = 1.0
) -> jax.Array:
    """Exact Boltzmann weights over the full state space, in enumeration order.

    Args:
        size: Linear lattice size ``L``.
        temperature: Temperature ``T``.
        coupling: Exchange coupling ``J``.

    Returns:
        Normalized probabilities of shape ``(2**(L*L),)``, ordered to match
        `ising_all_configurations`.
    """
    energies = ising_energy(ising_all_configurations(size), coupling)
    log_weights = -energies / jnp.asarray(temperature, dtype=jnp.result_type(float))
    return jnp.exp(log_weights - logsumexp(log_weights))


def ising_exact_observables(
    size: int, temperature: Scalar, coupling: Scalar = 1.0
) -> dict[str, jax.Array]:
    """Exact thermodynamics by exhaustive enumeration of the state space.

    Args:
        size: Linear lattice size ``L``; ``L**2 <=`` `MAX_ENUMERATED_SPINS`.
        temperature: Temperature ``T``.
        coupling: Exchange coupling ``J``.

    Returns:
        A dict with ``log_z``, ``free_energy_per_site``, ``energy_per_site``,
        ``abs_magnetization``, ``magnetization_squared`` and
        ``heat_capacity`` (per site).
    """
    temperature = jnp.asarray(temperature, dtype=jnp.result_type(float))
    beta = 1.0 / temperature
    num_spins = size * size

    configurations = ising_all_configurations(size)
    energies = ising_energy(configurations, coupling)
    magnetizations = ising_magnetization(configurations)

    log_weights = -beta * energies
    log_z = logsumexp(log_weights)
    weights = jnp.exp(log_weights - log_z)

    mean_energy = jnp.sum(weights * energies)
    mean_energy_squared = jnp.sum(weights * energies**2)
    variance = mean_energy_squared - mean_energy**2

    return {
        "log_z": log_z,
        "free_energy_per_site": -log_z / (beta * num_spins),
        "energy_per_site": mean_energy / num_spins,
        "abs_magnetization": jnp.sum(weights * jnp.abs(magnetizations)),
        "magnetization_squared": jnp.sum(weights * magnetizations**2),
        "heat_capacity": beta**2 * variance / num_spins,
    }


def ising_transfer_matrix_log_z(
    size: int, temperature: Scalar, coupling: Scalar = 1.0
) -> jax.Array:
    r"""Exact $\log Z$ of the finite periodic $L \times L$ lattice.

    Builds the $2^L \times 2^L$ column-to-column transfer matrix and
    evaluates $Z = \operatorname{Tr} T^L$ from its eigenvalues. The matrix
    entries reach $e^{2 \beta J L}$, so the largest element is factored out
    before exponentiating and the trace is accumulated relative to
    $\lambda_{\max}$ -- without that, ``float64`` overflows already at
    $L = 10$ and low temperature.

    Args:
        size: Linear lattice size ``L``; at most `MAX_TRANSFER_SIZE`.
        temperature: Temperature ``T``.
        coupling: Exchange coupling ``J``.

    Returns:
        A 0-d array holding ``log Z``.

    Raises:
        ValueError: If ``size`` exceeds `MAX_TRANSFER_SIZE`.
    """
    if size > MAX_TRANSFER_SIZE:
        raise ValueError(
            f"the transfer matrix is {2**size} x {2**size} at L = {size}; "
            f"the limit is L = {MAX_TRANSFER_SIZE}."
        )
    beta = 1.0 / jnp.asarray(temperature, dtype=jnp.result_type(float))
    coupling = jnp.asarray(coupling, dtype=jnp.result_type(float))

    states = jnp.arange(2**size, dtype=jnp.int32)
    bits = (states[:, None] >> jnp.arange(size, dtype=jnp.int32)[None, :]) & 1
    columns = 1.0 - 2.0 * bits.astype(jnp.result_type(float))

    # Bonds inside a column (periodic along it) and between adjacent columns.
    intra = jnp.sum(columns * jnp.roll(columns, 1, axis=-1), axis=-1)
    inter = columns @ columns.T
    log_transfer = beta * coupling * (inter + 0.5 * (intra[:, None] + intra[None, :]))

    shift = jnp.max(log_transfer)
    eigenvalues = jnp.linalg.eigvalsh(jnp.exp(log_transfer - shift))
    largest = jnp.max(jnp.abs(eigenvalues))
    ratios = eigenvalues / largest

    # Tr T^L = e^{L shift} lambda_max^L sum_i (lambda_i / lambda_max)^L. Take the
    # power through |ratio| so an odd L keeps the sign of a negative eigenvalue.
    powered = jnp.abs(ratios) ** size
    if size % 2:
        powered = jnp.sign(ratios) * powered
    return size * (shift + jnp.log(largest)) + jnp.log(jnp.sum(powered))


def onsager_free_energy_per_site(
    temperature: Array, coupling: Scalar = 1.0, num_quad: int = 4096
) -> jax.Array:
    r"""Onsager's exact free energy per site in the thermodynamic limit.

    $$-\beta f = \ln(2 \cosh 2\beta J)
      + \frac{1}{\pi} \int_0^{\pi/2}
        \ln\!\Big[\tfrac12\big(1 + \sqrt{1 - \kappa^2 \sin^2\phi}\,\big)\Big]
        \, d\phi,
      \qquad \kappa = \frac{2 \sinh 2\beta J}{\cosh^2 2\beta J}.$$

    The integrand is bounded everywhere, including at $T_c$ where
    $\kappa = 1$ (the singularity is in the *second* derivative), so a plain
    midpoint rule converges. ``cosh`` and ``sech`` are evaluated in log space so
    the expression stays finite down to very low temperature.

    Args:
        temperature: Temperature(s) ``T``, any shape.
        coupling: Exchange coupling ``J``.
        num_quad: Midpoint quadrature nodes on ``(0, pi/2)``.

    Returns:
        Free energy per site, same shape as ``temperature``. Tends to
        ``-T ln 2`` as ``T -> inf`` and to ``-2 J`` as ``T -> 0``.
    """
    temperature = jnp.asarray(temperature, dtype=jnp.result_type(float))
    beta = 1.0 / temperature
    argument = 2.0 * beta * jnp.asarray(coupling, dtype=jnp.result_type(float))

    magnitude = jnp.abs(argument)
    decay = jnp.exp(-2.0 * magnitude)
    log_two_cosh = magnitude + jnp.log1p(decay)
    sech = 2.0 * jnp.exp(-magnitude) / (1.0 + decay)
    kappa = 2.0 * jnp.tanh(argument) * sech

    phi = (jnp.arange(num_quad, dtype=jnp.result_type(float)) + 0.5) * (0.5 * jnp.pi / num_quad)
    radicand = 1.0 - (kappa[..., None] * jnp.sin(phi)) ** 2
    integrand = jnp.log(0.5 * (1.0 + jnp.sqrt(jnp.maximum(radicand, 0.0))))
    # mean * (pi/2) is the integral; dividing by pi leaves mean / 2.
    return -(log_two_cosh + 0.5 * jnp.mean(integrand, axis=-1)) / beta


def onsager_magnetization(temperature: Array, coupling: Scalar = 1.0) -> jax.Array:
    r"""Onsager's exact spontaneous magnetization $m = (1 - \sinh^{-4} 2\beta J)^{1/8}$.

    Zero for $T \ge T_c$ and rising to ``1`` as $T \to 0$, with the exact
    critical exponent $\beta = 1/8$.

    Args:
        temperature: Temperature(s) ``T``, any shape.
        coupling: Exchange coupling ``J``.

    Returns:
        Spontaneous magnetization in ``[0, 1]``, same shape as ``temperature``.
    """
    temperature = jnp.asarray(temperature, dtype=jnp.result_type(float))
    beta = 1.0 / temperature
    sinh = jnp.sinh(2.0 * beta * jnp.asarray(coupling, dtype=jnp.result_type(float)))
    inner = 1.0 - sinh ** (-4.0)
    # Double-where: the fractional power of a negative base would return NaN and
    # back-propagate NaN even from the unselected branch.
    ordered = inner > 0.0
    return jnp.where(ordered, jnp.where(ordered, inner, 1.0) ** 0.125, 0.0)


def onsager_energy_per_site(
    temperature: Array, coupling: Scalar = 1.0, num_quad: int = 4096
) -> jax.Array:
    r"""Onsager's exact internal energy per site, as $\partial (\beta f)/\partial \beta$.

    Taken by automatic differentiation of `onsager_free_energy_per_site`
    rather than from the closed form, which involves a complete elliptic
    integral $K(\kappa)$ that *diverges* logarithmically at $T_c$ and so is
    hard to quadrature there. $\beta f$ is smooth at $T_c$, so its derivative
    is well conditioned.

    Args:
        temperature: Temperature(s) ``T``, any shape.
        coupling: Exchange coupling ``J``.
        num_quad: Quadrature nodes passed through to the free energy.

    Returns:
        Internal energy per site, same shape as ``temperature``. Equals
        ``-sqrt(2) J`` at ``T_c``.
    """
    temperature = jnp.asarray(temperature, dtype=jnp.result_type(float))

    def beta_free_energy(beta: jax.Array) -> jax.Array:
        return beta * onsager_free_energy_per_site(1.0 / beta, coupling, num_quad)

    derivative = jax.grad(beta_free_energy)
    for _ in range(temperature.ndim):
        derivative = jax.vmap(derivative)
    return derivative(1.0 / temperature)


__all__ = [
    "MAX_ENUMERATED_SPINS",
    "MAX_TRANSFER_SIZE",
    "neighbour_sum",
    "ising_energy",
    "ising_magnetization",
    "checkerboard_sweep",
    "metropolis_chain",
    "wolff_update",
    "wolff_chain",
    "sample_ising",
    "ising_all_configurations",
    "ising_boltzmann_probabilities",
    "ising_exact_observables",
    "ising_transfer_matrix_log_z",
    "onsager_free_energy_per_site",
    "onsager_magnetization",
    "onsager_energy_per_site",
]

import math

import jax
import jax.numpy as jnp
import pytest

from qjax.physics import lattice as L
from qjax.physics import observables as O
from qjax.physics.reference import ISING_ENERGY_AT_TC, ISING_TC

jax.config.update("jax_enable_x64", True)


def test_ising_energy_counts_bonds():
    # An L x L periodic lattice has 2 L^2 bonds, so the aligned state sits at -2 J L^2.
    for size in (2, 3, 4, 6):
        aligned = jnp.ones((size, size))
        assert float(L.ising_energy(aligned)) == pytest.approx(-2.0 * size**2)
    # Flipping one spin breaks four bonds: Delta E = 2 J s_i n_i = 8 J.
    aligned = jnp.ones((4, 4))
    flipped = aligned.at[1, 2].set(-1.0)
    assert float(L.ising_energy(flipped) - L.ising_energy(aligned)) == pytest.approx(8.0)


def test_ising_energy_and_magnetization_broadcast():
    configurations = jnp.stack([jnp.ones((4, 4)), -jnp.ones((4, 4))])
    assert L.ising_energy(configurations).shape == (2,)
    assert jnp.allclose(L.ising_magnetization(configurations), jnp.array([1.0, -1.0]))
    # Z2 symmetry of the Hamiltonian.
    assert jnp.allclose(L.ising_energy(configurations[0]), L.ising_energy(-configurations[0]))


def test_neighbour_sum_is_periodic():
    single = jnp.zeros((4, 4)).at[0, 0].set(1.0)
    neighbours = L.neighbour_sum(single)
    # The site at (0, 0) has neighbours (0, 1), (0, 3), (1, 0) and (3, 0).
    assert float(neighbours[0, 1]) == 1.0
    assert float(neighbours[0, 3]) == 1.0
    assert float(neighbours[3, 0]) == 1.0
    assert float(jnp.sum(neighbours)) == 4.0


@pytest.mark.parametrize("temperature", [1.0, ISING_TC, 4.0])
def test_transfer_matrix_matches_brute_force(temperature):
    # Two independent exact codes: full enumeration of 2**16 states against the
    # 16 x 16 transfer matrix. Neither can be wrong without the other noticing.
    brute = L.ising_exact_observables(4, temperature)["log_z"]
    transfer = L.ising_transfer_matrix_log_z(4, temperature)
    assert float(abs(brute - transfer)) < 1e-10


@pytest.mark.parametrize("size", [2, 3])
def test_transfer_matrix_handles_odd_and_even_sizes(size):
    brute = L.ising_exact_observables(size, 1.7)["log_z"]
    transfer = L.ising_transfer_matrix_log_z(size, 1.7)
    assert float(abs(brute - transfer)) < 1e-10


def test_transfer_matrix_survives_low_temperature():
    # Matrix entries reach e^{2 beta J L}; without factoring out the largest
    # element this overflows float64 well before T = 0.2 at L = 10.
    log_z = L.ising_transfer_matrix_log_z(10, 0.2)
    assert jnp.isfinite(log_z)
    # Z -> 2 exp(2 J L^2 / T) as T -> 0 (the two aligned ground states).
    assert float(log_z) == pytest.approx(math.log(2.0) + 2.0 * 100 / 0.2, rel=1e-6)


def test_onsager_limits_and_internal_energy():
    # T -> inf: f -> -T ln 2 (all 2^N states equally likely).
    assert float(-L.onsager_free_energy_per_site(1e6) / 1e6) == pytest.approx(
        math.log(2.0), rel=1e-8
    )
    # T -> 0: f -> -2 J (the aligned ground state).
    assert float(L.onsager_free_energy_per_site(0.05)) == pytest.approx(-2.0, abs=1e-9)
    # u(T_c) = -sqrt(2) J, by autodiff of beta f.
    assert float(L.onsager_energy_per_site(ISING_TC)) == pytest.approx(ISING_ENERGY_AT_TC, abs=1e-4)


def test_onsager_energy_is_array_shaped():
    temperatures = jnp.array([1.5, ISING_TC, 3.0])
    energies = L.onsager_energy_per_site(temperatures)
    assert energies.shape == (3,)
    # Internal energy rises monotonically with temperature.
    assert bool(jnp.all(jnp.diff(energies) > 0.0))


def test_onsager_matches_finite_size_at_high_temperature():
    # Well above T_c the correlation length is short, so a 10 x 10 lattice is
    # already at the thermodynamic limit -- an independent check of both codes.
    exact_finite = -L.ising_transfer_matrix_log_z(10, 6.0) * 6.0 / 100.0
    assert float(abs(exact_finite - L.onsager_free_energy_per_site(6.0))) < 1e-4


def test_onsager_magnetization():
    assert float(L.onsager_magnetization(0.1)) == pytest.approx(1.0, abs=1e-12)
    assert float(L.onsager_magnetization(ISING_TC)) == 0.0
    assert float(L.onsager_magnetization(3.0)) == 0.0
    # Continuous at T_c and monotone below it. The 1/8 power decays very
    # slowly, so m is still ~0.2 one part in 10^6 below T_c -- the continuity
    # that matters is that it stays bounded and drops well below its T -> 0 value.
    below = L.onsager_magnetization(jnp.linspace(0.5, ISING_TC - 1e-6, 50))
    assert bool(jnp.all(jnp.diff(below) < 0.0))
    assert float(below[-1]) < 0.25
    # The exact exponent beta = 1/8 shows up as m ~ (T_c - T)^{1/8}.
    gap = jnp.array([1e-4, 1e-3, 1e-2])
    magnetization = L.onsager_magnetization(ISING_TC - gap)
    slope = jnp.diff(jnp.log(magnetization)) / jnp.diff(jnp.log(gap))
    assert bool(jnp.all(jnp.abs(slope - 0.125) < 0.01))


def test_onsager_magnetization_has_no_nan_gradient_above_tc():
    # The fractional power of a negative base would poison the gradient even in
    # the unselected branch; the double-where in the implementation prevents it.
    gradient = jax.grad(lambda t: L.onsager_magnetization(t))(3.0)
    assert jnp.isfinite(gradient)


def test_exact_observables_high_and_low_temperature():
    hot = L.ising_exact_observables(2, 1e6)
    # T -> inf: log Z -> N ln 2, u -> 0, <|m|> is the random-configuration value.
    assert float(hot["log_z"]) == pytest.approx(4.0 * math.log(2.0), rel=1e-6)
    assert float(hot["energy_per_site"]) == pytest.approx(0.0, abs=1e-5)
    cold = L.ising_exact_observables(4, 0.05)
    assert float(cold["energy_per_site"]) == pytest.approx(-2.0, abs=1e-9)
    assert float(cold["abs_magnetization"]) == pytest.approx(1.0, abs=1e-9)
    assert float(cold["magnetization_squared"]) == pytest.approx(1.0, abs=1e-9)
    # Heat capacity is a variance and cannot be negative.
    for temperature in (0.5, 1.5, ISING_TC, 4.0):
        assert float(L.ising_exact_observables(4, temperature)["heat_capacity"]) >= 0.0


def test_boltzmann_probabilities_are_normalized_and_ordered():
    probabilities = L.ising_boltzmann_probabilities(2, 1.2)
    assert float(jnp.sum(probabilities)) == pytest.approx(1.0)
    energies = L.ising_energy(L.ising_all_configurations(2))
    # The two ground states carry the most weight.
    assert int(jnp.argmax(probabilities)) in set(
        int(i) for i in jnp.where(energies == jnp.min(energies))[0]
    )


def test_all_configurations_are_distinct_and_complete():
    configurations = L.ising_all_configurations(3)
    assert configurations.shape == (2**9, 3, 3)
    assert set(jnp.unique(configurations).tolist()) == {-1.0, 1.0}
    flattened = configurations.reshape(2**9, 9)
    assert len({tuple(row.tolist()) for row in flattened}) == 2**9


def test_enumeration_and_transfer_matrix_reject_oversized_lattices():
    with pytest.raises(ValueError, match="the limit is"):
        L.ising_all_configurations(5)
    with pytest.raises(ValueError, match="the limit is"):
        L.ising_transfer_matrix_log_z(11, 2.0)


@pytest.mark.parametrize("temperature", [1.5, 2.5, 4.0])
def test_metropolis_reproduces_exact_observables(temperature):
    # The detailed-balance gate. A sampler that violates it still produces
    # plausible-looking configurations, so nothing else in the suite would catch
    # it -- but it cannot match an exhaustive enumeration to within 3 sigma.
    samples = 4000
    configurations = L.sample_ising(
        jax.random.PRNGKey(0), 4, jnp.array([temperature]), samples, sweeps=300
    )[0]
    exact = L.ising_exact_observables(4, temperature)

    absolute = jnp.abs(L.ising_magnetization(configurations))
    energy = L.ising_energy(configurations) / 16.0
    for measured, reference in ((absolute, "abs_magnetization"), (energy, "energy_per_site")):
        error = jnp.std(measured) / jnp.sqrt(samples)
        assert float(abs(jnp.mean(measured) - exact[reference])) < 3.0 * float(error)


def test_sample_ising_shape_and_values():
    temperatures = jnp.array([2.0, 3.0])
    configurations = L.sample_ising(jax.random.PRNGKey(1), 6, temperatures, 5, sweeps=10)
    assert configurations.shape == (2, 5, 6, 6)
    assert set(jnp.unique(configurations).tolist()) <= {-1.0, 1.0}


def test_checkerboard_sweep_updates_both_sublattices():
    # At infinite temperature Metropolis accepts every proposal, and the proposal
    # is "flip", so a correct sweep inverts the whole lattice. If either colour
    # were skipped, half the sites -- one full sublattice -- would come back
    # untouched, which nothing else in the suite would notice.
    spins = jnp.ones((8, 8))
    after = L.checkerboard_sweep(jax.random.PRNGKey(0), spins, beta=0.0)
    index = jnp.arange(8)
    parity = (index[:, None] + index[None, :]) % 2
    for colour in (0, 1):
        assert int(jnp.sum((after < 0) & (parity == colour))) == 32


def test_metropolis_reaches_the_ground_state_at_low_temperature():
    spins = jnp.where(jax.random.bernoulli(jax.random.PRNGKey(3), 0.5, (8, 8)), 1.0, -1.0)
    final = L.metropolis_chain(jax.random.PRNGKey(4), spins, beta=5.0, sweeps=400)
    assert float(jnp.abs(L.ising_magnetization(final))) > 0.95


def test_finite_size_scaling_recovers_the_exact_critical_temperature():
    # End-to-end physics check with no machine learning in it: the Binder
    # cumulant crossing across lattice sizes, extrapolated with the exact nu = 1.
    temperatures = jnp.linspace(2.0, 2.6, 13)
    sizes = (6, 8, 10, 12)
    crossings = []
    reference = None
    for size in sizes:
        configurations = L.sample_ising(
            jax.random.PRNGKey(size), size, temperatures, 400, sweeps=400
        )
        cumulant = O.binder_cumulant(L.ising_magnetization(configurations), axis=-1)
        if reference is None:
            reference = cumulant
            continue
        # Where this size's cumulant crosses the smallest size's.
        crossings.append(O.crossing_temperature(temperatures, cumulant - reference, 0.0))
    estimate, _, _ = O.finite_size_extrapolation(jnp.array(sizes[1:]), jnp.array(crossings))
    assert float(abs(estimate - ISING_TC)) < 0.15

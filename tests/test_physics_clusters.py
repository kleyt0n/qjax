import jax
import jax.numpy as jnp
import pytest

from qjax.physics import clusters as C
from qjax.physics.reference import LJ_PAIR_DISTANCE, LJ_REFERENCE_MINIMA

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("num_atoms", [2, 3, 4])
def test_lj_energy_on_equidistant_clusters(num_atoms):
    # A regular simplex at r = 2**(1/6) sigma puts every pair exactly at the
    # potential minimum, so the energy is -epsilon per pair: -1, -3, -6. These
    # are closed forms, not tabulated values, so they pin the potential exactly.
    positions = C.equidistant_cluster(num_atoms, LJ_PAIR_DISTANCE)
    energy = C.lj_energy(positions)
    assert float(energy) == pytest.approx(LJ_REFERENCE_MINIMA[num_atoms], abs=1e-12)

    # Every edge really is the same length.
    offsets = positions[:, None, :] - positions[None, :, :]
    distances = jnp.sqrt(jnp.sum(offsets**2, axis=-1))
    off_diagonal = distances[~jnp.eye(num_atoms, dtype=bool)]
    assert jnp.allclose(off_diagonal, LJ_PAIR_DISTANCE, atol=1e-12)


def test_lj_energy_invariances():
    positions = C.equidistant_cluster(4, LJ_PAIR_DISTANCE)
    reference = C.lj_energy(positions)

    translated = positions + jnp.array([3.0, -1.0, 2.0])
    assert float(abs(C.lj_energy(translated) - reference)) < 1e-12

    angle = 0.7
    rotation = jnp.array(
        [
            [jnp.cos(angle), -jnp.sin(angle), 0.0],
            [jnp.sin(angle), jnp.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert float(abs(C.lj_energy(positions @ rotation.T) - reference)) < 1e-12

    permuted = positions[jnp.array([2, 0, 3, 1])]
    assert float(abs(C.lj_energy(permuted) - reference)) < 1e-12


def test_lj_energy_scales_with_epsilon_and_sigma():
    positions = C.equidistant_cluster(3, 2.0 * LJ_PAIR_DISTANCE)
    # Doubling sigma with the geometry doubled leaves the reduced distance fixed.
    assert float(C.lj_energy(positions, sigma=2.0)) == pytest.approx(-3.0, abs=1e-12)
    # The well depth is linear in epsilon.
    assert float(C.lj_energy(positions, epsilon=2.5, sigma=2.0)) == pytest.approx(-7.5, abs=1e-11)


def test_lj_energy_is_batched_and_has_no_singular_gradient():
    batch = jax.random.normal(jax.random.PRNGKey(0), (5, 7, 3)) * 1.5
    assert C.lj_energy(batch).shape == (5,)
    # Coincident atoms are the r^-12 singularity; the softened, masked form must
    # not back-propagate NaN into every coordinate.
    gradient = jax.grad(C.lj_energy)(jnp.zeros((3, 3)))
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert jnp.isfinite(C.lj_energy(jnp.zeros((3, 3))))


def test_lj_energy_confined_is_inert_inside_the_container():
    positions = C.equidistant_cluster(4, LJ_PAIR_DISTANCE)
    bare = C.lj_energy(positions)
    # Inside the wall the confined energy is the bare energy exactly, so a
    # minimum found strictly inside is a minimum of the real potential.
    assert float(C.lj_energy_confined(positions, container_radius=10.0)) == pytest.approx(
        float(bare), abs=1e-12
    )
    # Outside, the wall is a strictly positive penalty.
    escaped = positions.at[0].set(jnp.array([20.0, 0.0, 0.0]))
    assert float(C.lj_energy_confined(escaped, container_radius=3.0)) > float(C.lj_energy(escaped))


def test_lj_random_cluster_stays_inside_the_ball():
    positions = C.lj_random_cluster(jax.random.PRNGKey(1), 200, radius=4.0)
    assert positions.shape == (200, 3)
    radii = jnp.linalg.norm(positions, axis=-1)
    assert float(jnp.max(radii)) <= 4.0 + 1e-9
    # u**(1/3) makes the law uniform in volume, so the mean radius is 3R/4.
    assert float(jnp.mean(radii)) == pytest.approx(3.0, rel=0.08)


def test_lj_quench_reaches_the_dimer_minimum():
    start = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    final, energies = C.lj_quench(start, steps=3000, learning_rate=0.01)
    assert energies.shape == (3001,)
    assert float(energies[-1]) == pytest.approx(-1.0, abs=1e-4)
    assert float(energies[-1]) < float(energies[0])
    separation = float(jnp.linalg.norm(final[1] - final[0]))
    assert separation == pytest.approx(LJ_PAIR_DISTANCE, abs=1e-4)


def test_lj_quench_never_reports_an_impossible_energy():
    # LJ7 has a known global minimum; no local quench may go below it.
    for seed in range(6):
        start = C.lj_random_cluster(jax.random.PRNGKey(seed), 7, radius=1.6)
        _, energies = C.lj_quench(start, steps=1500, learning_rate=0.01)
        assert float(jnp.min(energies)) > LJ_REFERENCE_MINIMA[7] - 1e-6


def test_equidistant_cluster_rejects_more_than_four_vertices():
    with pytest.raises(ValueError, match="at most 4 vertices"):
        C.equidistant_cluster(5, 1.0)
    with pytest.raises(ValueError, match="at most 4 vertices"):
        C.equidistant_cluster(0, 1.0)
    assert C.equidistant_cluster(1, 1.0).shape == (1, 3)


def test_coordination_numbers():
    # In a regular tetrahedron every atom touches the other three.
    positions = C.equidistant_cluster(4, LJ_PAIR_DISTANCE)
    assert jnp.all(C.coordination_numbers(positions) == 3)
    # Below the edge length nothing is a neighbour.
    assert jnp.all(C.coordination_numbers(positions, cutoff=1.0) == 0)
    # Batched.
    assert C.coordination_numbers(jnp.stack([positions, positions])).shape == (2, 4)


def test_confined_energy_is_differentiable_at_the_origin():
    # The soft wall goes through sqrt(sum x^2), whose derivative is infinite at
    # zero; the ``maximum`` outside it then multiplies that by zero. Without
    # sanitizing the radicand first, an atom at the origin -- the centre of a
    # re-centred icosahedron -- gives a finite energy and a NaN gradient.
    positions = jnp.array([[0.0, 0.0, 0.0], [1.12, 0.0, 0.0], [0.0, 1.12, 0.0]])
    assert bool(jnp.isfinite(C.lj_energy_confined(positions, 5.0)))
    gradient = jax.grad(lambda x: C.lj_energy_confined(x, 5.0))(positions)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    # Strictly inside the container the wall contributes nothing at all.
    assert float(C.lj_energy_confined(positions, 5.0)) == pytest.approx(
        float(C.lj_energy(positions))
    )
    bare = jax.grad(C.lj_energy)(positions)
    assert jnp.allclose(gradient, bare)

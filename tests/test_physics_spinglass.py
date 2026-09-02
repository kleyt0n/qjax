import jax
import jax.numpy as jnp
import pytest

from qjax.physics import spinglass as S

jax.config.update("jax_enable_x64", True)


def test_sk_couplings_are_symmetric_with_zero_diagonal():
    couplings = S.sk_couplings(jax.random.PRNGKey(0), 64)
    assert couplings.shape == (64, 64)
    assert jnp.allclose(couplings, couplings.T)
    assert jnp.allclose(jnp.diag(couplings), 0.0)
    # Variance 1/N off the diagonal is the scaling that makes E/N intensive.
    off_diagonal = couplings[~jnp.eye(64, dtype=bool)]
    assert float(jnp.std(off_diagonal)) == pytest.approx(1.0 / 8.0, rel=0.06)


def test_sk_energy_closed_form_and_symmetries():
    coupling = 0.83
    couplings = jnp.array([[0.0, coupling], [coupling, 0.0]])
    # E(s) = -J s_0 s_1 for two spins.
    spins = jnp.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
    assert jnp.allclose(S.sk_energy(spins, couplings), jnp.array([-1.0, 1.0, 1.0, -1.0]) * coupling)
    # Z2 symmetry, and batching over leading axes.
    bigger = S.sk_couplings(jax.random.PRNGKey(1), 12)
    batch = jnp.sign(jax.random.normal(jax.random.PRNGKey(2), (5, 3, 12)))
    assert S.sk_energy(batch, bigger).shape == (5, 3)
    assert jnp.allclose(S.sk_energy(batch, bigger), S.sk_energy(-batch, bigger))


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_sk_exact_free_energy_matches_the_two_spin_closed_form(temperature):
    coupling = 0.83
    couplings = jnp.array([[0.0, coupling], [coupling, 0.0]])
    exact = S.sk_exact_observables(couplings, temperature, chunk=4)
    # Z = 2 e^{beta J} + 2 e^{-beta J} = 4 cosh(beta J).
    expected = jnp.log(4.0 * jnp.cosh(coupling / temperature))
    assert float(abs(exact["log_z"] - expected)) < 1e-12
    # <s_0 s_1> = tanh(beta J), and <s_i s_i> = 1.
    assert float(exact["correlations"][0, 1]) == pytest.approx(
        float(jnp.tanh(coupling / temperature))
    )
    assert jnp.allclose(jnp.diag(exact["correlations"]), 1.0)
    assert float(exact["ground_state_energy_per_spin"]) == pytest.approx(-coupling / 2.0)
    # F = E - T S <= E, so the free energy per spin never exceeds the mean energy.
    assert float(exact["free_energy_per_spin"]) <= float(exact["energy_per_spin"]) + 1e-12


def test_sk_exact_observables_are_chunk_invariant():
    # The streamed logsumexp must give the same answer whatever the chunking.
    couplings = S.sk_couplings(jax.random.PRNGKey(3), 12)
    reference = S.sk_exact_observables(couplings, 0.4, chunk=4096)
    for chunk in (2, 64, 1000, 1 << 20):
        other = S.sk_exact_observables(couplings, 0.4, chunk=chunk)
        assert float(abs(other["log_z"] - reference["log_z"])) < 1e-11
        assert jnp.allclose(other["correlations"], reference["correlations"], atol=1e-11)


def test_sk_exact_observables_low_temperature_limit():
    couplings = S.sk_couplings(jax.random.PRNGKey(4), 10)
    cold = S.sk_exact_observables(couplings, 0.01)
    ground = cold["ground_state_energy_per_spin"]
    # As T -> 0 both f and u collapse onto the ground-state energy.
    assert float(cold["energy_per_spin"]) == pytest.approx(float(ground), abs=1e-9)
    assert float(cold["free_energy_per_spin"]) == pytest.approx(float(ground), abs=1e-3)
    # Correlations saturate at +/- 1 on a (twofold degenerate) ground state.
    assert bool(jnp.all(jnp.abs(cold["correlations"]) > 0.999))


def test_sk_exact_observables_high_temperature_limit():
    num_spins = 12
    couplings = S.sk_couplings(jax.random.PRNGKey(5), num_spins)
    hot = S.sk_exact_observables(couplings, 1e5)
    assert float(hot["log_z"]) == pytest.approx(num_spins * jnp.log(2.0), rel=1e-8)
    assert float(hot["energy_per_spin"]) == pytest.approx(0.0, abs=1e-4)
    # Spins decorrelate: only the diagonal survives.
    off_diagonal = hot["correlations"][~jnp.eye(num_spins, dtype=bool)]
    assert float(jnp.max(jnp.abs(off_diagonal))) < 1e-3


def test_sk_exact_correlations_matches_the_full_observables():
    couplings = S.sk_couplings(jax.random.PRNGKey(6), 10)
    direct = S.sk_exact_correlations(couplings, 0.7)
    assert jnp.allclose(direct, S.sk_exact_observables(couplings, 0.7)["correlations"])
    assert jnp.allclose(direct, direct.T)


def test_sk_ground_state_is_near_the_parisi_value_on_average():
    # A single realization fluctuates strongly at these sizes, so average over
    # disorder. This is a sanity check on the scaling of sk_couplings, not a
    # precision claim about the N -> inf limit.
    energies = [
        float(
            S.sk_exact_observables(S.sk_couplings(jax.random.PRNGKey(seed), 18), 0.01)[
                "ground_state_energy_per_spin"
            ]
        )
        for seed in range(8)
    ]
    assert -0.95 < sum(energies) / len(energies) < -0.60


def test_sk_rejects_oversized_systems():
    couplings = jnp.zeros((23, 23))
    with pytest.raises(ValueError, match="the limit is"):
        S.sk_exact_observables(couplings, 1.0)


def test_sk_exact_observables_is_jittable():
    couplings = S.sk_couplings(jax.random.PRNGKey(7), 10)
    jitted = jax.jit(lambda c: S.sk_exact_observables(c, 0.6)["log_z"])
    assert float(jitted(couplings)) == pytest.approx(
        float(S.sk_exact_observables(couplings, 0.6)["log_z"])
    )

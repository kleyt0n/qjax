import jax
import jax.numpy as jnp
import numpy as np
import pytest

import qjax
import qjax.physics as qp
from qjax.nn import bounded_q, made_init, made_log_prob, made_masks, made_sample

jax.config.update("jax_enable_x64", True)

EXAMPLES = (
    "ising_phases",
    "tsallis_free_energy",
    "generalized_annealing",
    "anomalous_diffusion",
    "pinn_fokker_planck",
)


# --------------------------------------------------------------------------- #
# The identity the free-energy example rests on
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("q", [0.4, 0.8, 1.0, 1.2, 1.8])
def test_deformed_entropy_is_the_dual_index(q):
    # -E_p[ln_q p] = S_{2-q}(p), NOT S_q(p). The escort weight p^q would give
    # S_q; summing against p itself flips the index. Getting this backwards
    # inverts the direction of every conclusion in the example, so it is pinned.
    probabilities = jax.random.dirichlet(jax.random.PRNGKey(0), jnp.ones(64) * 0.4)
    deformed = -jnp.sum(probabilities * qjax.q_log(probabilities, q))
    assert float(deformed) == pytest.approx(
        float(qjax.tsallis_entropy(probabilities, 2.0 - q)), rel=1e-12
    )
    if q != 1.0:
        assert float(deformed) != pytest.approx(float(qjax.tsallis_entropy(probabilities, q)))


@pytest.mark.parametrize("q", [0.6, 1.0, 1.4])
def test_free_energy_gradient_matches_the_analytic_estimator(q, example):
    # The example's central claim: jax.grad of the surrogate *is* the REINFORCE
    # estimator E[(E + T ln_q p + T p^{1-q}) grad log p], never hand-derived.
    module = example("tsallis_free_energy")
    num_spins, hidden, temperature = 8, (16,), 1.3

    masks = made_masks(num_spins, hidden)
    params = made_init(jax.random.PRNGKey(0), num_spins, hidden)
    spins = made_sample(jax.random.PRNGKey(1), params, masks, 512)
    couplings = qp.sk_couplings(jax.random.PRNGKey(2), num_spins)
    energies = qp.sk_energy(spins, couplings)
    baseline = jnp.mean(energies)

    automatic = jax.grad(module.free_energy_surrogate)(
        params, masks, spins, energies, temperature, q, baseline
    )

    def analytic(p):
        log_p = made_log_prob(p, masks, spins)
        probability = jnp.exp(jax.lax.stop_gradient(log_p))
        weight = (
            energies
            + temperature * qjax.q_log(probability, q)
            + temperature * probability ** (1.0 - q)
            - baseline
        )
        return jnp.mean(jax.lax.stop_gradient(weight) * log_p)

    expected = jax.grad(analytic)(params)
    for key in ("weights", "biases"):
        for got, want in zip(automatic[key], expected[key], strict=True):
            assert jnp.allclose(got, want, atol=1e-9, rtol=1e-7)


def test_free_energy_estimator_reduces_to_van_at_q_one(example):
    # At q = 1 the deformed estimator differs from the plain VAN one by exactly
    # T * grad E[log p], and E[grad log p] = 0, so the two agree in expectation.
    # That identity is exact for *any* batch, which is the checkable version of
    # "the q -> 1 limit is the published algorithm"; the raw gradients themselves
    # differ at O(1/sqrt(B)) on a finite sample, which is not a defect.
    module = example("tsallis_free_energy")
    num_spins, hidden, temperature = 8, (16,), 1.3
    masks = made_masks(num_spins, hidden)
    params = made_init(jax.random.PRNGKey(3), num_spins, hidden)
    spins = made_sample(jax.random.PRNGKey(4), params, masks, 2048)
    couplings = qp.sk_couplings(jax.random.PRNGKey(5), num_spins)
    energies = qp.sk_energy(spins, couplings)
    baseline = jnp.mean(energies)

    automatic = jax.grad(module.free_energy_surrogate)(
        params, masks, spins, energies, temperature, 1.0, baseline
    )

    def van(p):
        log_p = made_log_prob(p, masks, spins)
        weight = energies + temperature * jax.lax.stop_gradient(log_p) - baseline
        return jnp.mean(jax.lax.stop_gradient(weight) * log_p)

    def score_correction(p):
        return temperature * jnp.mean(made_log_prob(p, masks, spins))

    plain = jax.grad(van)(params)
    correction = jax.grad(score_correction)(params)
    for key in ("weights", "biases"):
        for got, base, extra in zip(automatic[key], plain[key], correction[key], strict=True):
            assert jnp.allclose(got, base + extra, atol=1e-10, rtol=1e-8)

    # And the correction really is a mean-zero quantity: it shrinks as 1/sqrt(B).
    magnitudes = []
    for batch in (256, 4096):
        drawn = made_sample(jax.random.PRNGKey(6), params, masks, batch)

        def mean_log_prob(p, drawn=drawn):
            return temperature * jnp.mean(made_log_prob(p, masks, drawn))

        gradient = jax.grad(mean_log_prob)(params)
        magnitudes.append(float(jnp.sqrt(sum(jnp.sum(g**2) for g in gradient["weights"]))))
    assert magnitudes[1] < magnitudes[0]


def test_variational_free_energy_is_an_upper_bound_at_q_one(example):
    # F_1 is a variational bound, so it can never fall below the exact value.
    # Checked by exhaustive enumeration on a 2x2 lattice for random parameters.
    example("tsallis_free_energy")  # the module under test defines the objective
    size, temperature = 2, 1.4
    num_spins = size * size
    masks = made_masks(num_spins, (16,))
    configurations = qp.ising_all_configurations(size).reshape(-1, num_spins)
    energies = qp.ising_energy(configurations.reshape(-1, size, size))
    exact = float(qp.ising_exact_observables(size, temperature)["free_energy_per_site"] * num_spins)

    for seed in range(6):
        params = made_init(jax.random.PRNGKey(seed), num_spins, (16,))
        probabilities = jnp.exp(made_log_prob(params, masks, configurations))
        # Exact expectation over the whole state space, not a sample estimate.
        entropy_term = jnp.sum(probabilities * qjax.q_log(probabilities, 1.0))
        variational = float(jnp.sum(probabilities * energies) + temperature * entropy_term)
        assert variational >= exact - 1e-9


def test_entropic_index_scaling_is_size_matched(example):
    module = example("tsallis_free_energy")
    for num_spins in (12, 16, 20):
        for scaled in (-2.0, 0.0, 4.0):
            q = module.entropic_index(scaled, num_spins)
            assert (q - 1.0) * num_spins == pytest.approx(scaled)
    assert module.entropic_index(0.0, 20) == 1.0


# --------------------------------------------------------------------------- #
# The Ising example's estimators
# --------------------------------------------------------------------------- #
def test_order_parameter_reference_measures_the_crossover(example):
    module = example("ising_phases")
    temperatures = jnp.linspace(1.6, 3.0, 12)
    smaller = qp.sample_ising(jax.random.PRNGKey(0), 6, temperatures, 200, 300)
    larger = qp.sample_ising(jax.random.PRNGKey(1), 12, temperatures, 200, 500)

    noises = []
    for configurations in (smaller, larger):
        threshold, noise, labels = module.order_parameter_reference(configurations, temperatures)
        assert 0.0 < float(threshold) < 1.0
        assert noise.shape == (temperatures.shape[0],)
        assert bool(jnp.all((noise >= 0.0) & (noise <= 1.0)))
        assert labels.shape == configurations.shape[:2]
        # The disagreement is concentrated at the transition, not spread evenly.
        peak = int(jnp.argmax(noise))
        assert abs(float(temperatures[peak]) - qp.ISING_TC) < 0.35
        noises.append(float(jnp.mean(noise)))

    # And it shrinks as the lattice grows, which is the premise of the example.
    assert noises[1] < noises[0]


def test_balanced_accuracy_pins_an_uninformative_split(example):
    module = example("ising_phases")
    targets = jnp.concatenate([jnp.zeros(95, dtype=jnp.int32), jnp.ones(5, dtype=jnp.int32)])
    # A constant predictor scores 95% plain but 50% balanced -- which is why the
    # confusion scan uses the balanced form.
    constant = jnp.zeros(100, dtype=jnp.int32)
    assert module.balanced_accuracy(constant, targets) == pytest.approx(0.5)
    assert module.balanced_accuracy(targets, targets) == pytest.approx(1.0)


def test_crossover_width_shrinks_with_lattice_size(example):
    module = example("ising_phases")
    temperatures = jnp.linspace(1.8, 2.8, 40)
    # A planted sigmoid whose width halves: the estimator must see the factor 2.
    widths = []
    for steepness in (6.0, 12.0):
        probability = 1.0 / (1.0 + jnp.exp((temperatures - qp.ISING_TC) * steepness))
        widths.append(float(module.crossover_width(temperatures, probability)))
    assert widths[0] / widths[1] == pytest.approx(2.0, rel=0.05)


# --------------------------------------------------------------------------- #
# The annealing example's engine
# --------------------------------------------------------------------------- #
def test_bounded_step_survives_the_heavy_tail(example):
    module = example("generalized_annealing")
    limit = 2.0 * 2.8 * float(jnp.sqrt(7.0))
    keys = jax.random.split(jax.random.PRNGKey(0), 4000)

    # At q_V = 2.7 the underlying sampler returns +inf for about 1 draw in 2000 in
    # float32, and the tail is so heavy that most draws exceed the container
    # anyway. The step must stay finite and bounded regardless -- that is the
    # regression this guards.
    wild = jax.vmap(lambda k: module.bounded_step(k, 2.7, 0.9, 7, 2.8))(keys)
    assert bool(jnp.all(jnp.isfinite(wild)))
    magnitudes = jnp.linalg.norm(wild.reshape(4000, -1), axis=-1)
    assert float(jnp.max(magnitudes)) <= limit + 1e-6
    # It really is that wild: a majority of proposals hit the cap, which is the
    # measured reason q_V = 2.7 performs badly.
    assert float(jnp.mean(magnitudes > 0.95 * limit)) > 0.3

    # At a moderate index the step is heavy-tailed but the cap is rarely reached,
    # so the annealing schedule still controls the step length.
    moderate = jax.vmap(lambda k: module.bounded_step(k, 1.5, 0.9, 7, 2.8))(keys)
    magnitudes = jnp.linalg.norm(moderate.reshape(4000, -1), axis=-1)
    assert float(jnp.mean(magnitudes > 0.95 * limit)) < 0.02
    assert float(jnp.max(magnitudes) / jnp.median(magnitudes)) > 5.0


def test_gsa_solves_lj7_and_never_beats_the_reference(example):
    module = example("generalized_annealing")
    settings = {"atoms": 7, "trials": 24, "steps": 600, "quench": 25}
    _, polished, traces = module.sweep(jax.random.PRNGKey(0), settings, 1.5, 1.0)
    polished = np.asarray(polished)
    reference = qp.LJ_REFERENCE_MINIMA[7]
    # No run may report an energy below the tabulated global minimum.
    assert polished.min() > reference - 1e-2
    # And at least one must find it, or the engine is not working at all.
    assert polished.min() < reference + 1e-2
    # The best-so-far trace can only improve.
    assert bool(np.all(np.diff(np.asarray(traces), axis=1) <= 1e-9))


def test_visiting_scan_is_ordered_by_tail_index(example):
    module = example("generalized_annealing")
    assert module.VISITING_SCAN[0] == 1.0
    tails = [(3.0 - q) / (q - 1.0) for q in module.VISITING_SCAN if q > 1.0]
    # The scan sweeps the tail index monotonically downward, which is the axis the
    # example actually reasons about.
    assert all(later < earlier for earlier, later in zip(tails, tails[1:], strict=False))


# --------------------------------------------------------------------------- #
# The diffusion example's estimators
# --------------------------------------------------------------------------- #
def test_fit_q_beta_recovers_a_planted_q_gaussian(example):
    module = example("anomalous_diffusion")
    true_q, true_beta = 1.6, 0.8
    samples = qjax.sample(jax.random.PRNGKey(0), q=true_q, beta=true_beta, shape=(120_000,))
    q_hat, beta_hat, sigma, trace = module.fit_q_beta(samples, 1.3, 1.0, 3000)
    assert float(sigma) > 0.0
    assert abs(float(q_hat) - true_q) < 4.0 * float(sigma) + 0.02
    assert float(beta_hat) == pytest.approx(true_beta, rel=0.12)
    # The negative log-likelihood must have gone down.
    assert float(trace[-1, 0]) < float(trace[0, 0])


def test_fit_q_beta_recovers_the_gaussian_limit(example):
    module = example("anomalous_diffusion")
    samples = jax.random.normal(jax.random.PRNGKey(1), (120_000,)) / jnp.sqrt(2.0)
    # A standard Gaussian is the q = 1, beta = 1 q-Gaussian; the fit is bounded
    # below at q = 1, so it should sit against that bound.
    # bounded_q maps onto an *open* interval, so q = 1 is approached and never
    # attained; the estimate is an upper bound that tightens with more steps.
    loose, _, _, _ = module.fit_q_beta(samples, 1.4, 1.0, 900)
    tight, beta_hat, _, _ = module.fit_q_beta(samples, 1.4, 1.0, 4000)
    assert 1.0 < float(tight) < float(loose)
    assert float(tight) == pytest.approx(1.0, abs=0.06)
    assert float(beta_hat) == pytest.approx(1.0, rel=0.12)


def test_sisyphus_segment_reaches_its_exact_stationary_index(example):
    module = example("anomalous_diffusion")
    friction, diffusion, momentum_scale = 2.7, 0.5, 1.0
    target = float(qp.saturating_langevin_q(diffusion, friction, momentum_scale))
    target_beta = float(qp.saturating_langevin_beta(diffusion, friction))

    momenta = jnp.zeros((60_000,))
    key = jax.random.PRNGKey(2)
    for _ in range(30):
        momenta, key = module.sisyphus_segment(
            key, momenta, friction, diffusion, momentum_scale, 1e-3, 300
        )
    q_hat, beta_hat, sigma, _ = module.fit_q_beta(momenta, 1.3, target_beta, 3000)
    assert float(q_hat) == pytest.approx(target, abs=0.06)
    assert float(beta_hat) == pytest.approx(target_beta, rel=0.15)


def test_brownian_control_is_normal_diffusion(example):
    module = example("anomalous_diffusion")
    config = module.configuration(quick=True, full=False)
    snapshots, times = module.simulate(
        jax.random.PRNGKey(3), config | {"particles": 20_000}, "brownian", diffusivity=1.0
    )
    displacement = qp.mean_squared_displacement(jnp.asarray(snapshots))
    exponent, _, sigma = qp.fit_power_law(jnp.asarray(times), displacement, low=2)
    assert float(exponent) == pytest.approx(1.0, abs=4.0 * float(sigma) + 0.05)


# --------------------------------------------------------------------------- #
# Every script runs end to end
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", EXAMPLES)
def test_scripts_run_in_quick_mode(name, example, tmp_path, monkeypatch):
    # The CI `examples` job runs only a couple of the original scripts, so
    # without this the five statistical-physics ones would never execute in CI.
    import matplotlib

    matplotlib.use("Agg")
    module = example(name)
    monkeypatch.setattr(module, "FIG_DIR", tmp_path)
    module.main(quick=True)
    written = sorted(tmp_path.glob("*.pdf"))
    assert written, f"{name} wrote no figure"
    assert all(path.stat().st_size > 1000 for path in written)


# --------------------------------------------------------------------------- #
# The physics-informed neural network
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("q_loss", [1.2, 1.5, 2.0, 2.5])
def test_student_t_em_score_is_the_q_gaussian_score(q_loss, example):
    # The load-bearing claim of the whole example: the Student-t residual model of
    # Abijuru et al. (ICML 2026) *is* a q-Gaussian likelihood, at qjax's own
    # nu = (3-q)/(q-1) -- the same relation `qjax.sample` uses to draw q-Gaussian
    # variates. If this ever stopped holding, the example would be a claim about a
    # different method.
    module = example("pinn_fokker_planck")
    residual = jnp.linspace(-6.0, 6.0, 241)
    beta = 1.0

    ours = jax.vmap(jax.grad(lambda r, q=q_loss: -qjax.q_gaussian_logpdf(r, q, beta)))(residual)
    degrees, scale = module.matched_student_t(q_loss, beta)
    theirs = module.student_t_weight(residual, degrees, scale) * residual / scale**2

    assert float(degrees) == pytest.approx((3.0 - q_loss) / (q_loss - 1.0), rel=1e-12)
    assert float(jnp.max(jnp.abs(ours - theirs))) < 1e-12
    # And it is not vacuous: the score is genuinely non-trivial.
    assert float(jnp.max(jnp.abs(ours))) > 0.1


def test_deformed_loss_is_a_weighted_mean_squared_residual(example):
    # The deformed objective is a weighted MSE whose weights fall off with the
    # residual -- which is precisely the M-step of the EM algorithm.
    module = example("pinn_fokker_planck")
    residual = jnp.linspace(-6.0, 6.0, 241)
    beta = 0.7

    for q_loss in (1.0, 1.3, 1.9):
        score = jax.vmap(jax.grad(lambda r, q=q_loss: -qjax.q_gaussian_logpdf(r, q, beta)))(
            residual
        )
        weight = module.influence_weight(residual, q_loss, beta)
        assert jnp.allclose(score, weight * 2.0 * beta * residual, atol=1e-12)

    # q_loss = 1 is the mean-squared residual exactly: no reweighting at all.
    assert jnp.allclose(module.influence_weight(residual, 1.0, beta), 1.0, atol=0.0, rtol=0.0)
    # Above 1 the weight is a bounded, strictly decreasing function of |r| --
    # the definition of bounded influence, and the reason it is called robust.
    heavy = module.influence_weight(jnp.abs(residual), 1.8, beta)
    order = jnp.argsort(jnp.abs(residual))
    sorted_weight = heavy[order]
    assert bool(jnp.all(jnp.diff(sorted_weight) <= 1e-12))
    assert float(jnp.max(heavy)) <= 1.0


def test_fit_index_recovers_a_planted_tail(example):
    # The independent measurement the learned index is checked against.
    module = example("pinn_fokker_planck")
    for planted in (1.3, 1.6, 2.0):
        samples = qjax.sample(jax.random.PRNGKey(0), q=planted, beta=1.5, shape=(60_000,))
        q_hat, _, sigma = module.fit_index(samples, 3000)
        assert float(sigma) > 0.0
        assert abs(float(q_hat) - planted) < 4.0 * float(sigma) + 0.03

    # Gaussian samples sit against the q = 1 bound, so the fit reports "not
    # heavy-tailed" rather than inventing a tail.
    gaussian = jax.random.normal(jax.random.PRNGKey(1), (60_000,))
    q_hat, _, _ = module.fit_index(gaussian, 3000)
    assert float(q_hat) == pytest.approx(module.Q_LOSS_MIN, abs=0.05)


def test_every_constant_density_solves_the_equation(example):
    # The spurious family the collapse falls into, and the reason mass is a
    # held-out diagnostic: for a constant c both dc/dt and d^2 c^m/dx^2 vanish, so
    # the residual cannot tell a constant from the true solution.
    module = example("pinn_fokker_planck")
    for q in (0.5, 1.0, 1.5):
        for value in (0.0, 0.05, 1.0, 7.0):
            residual = jax.vmap(
                lambda x, value=value, q=q: qp.nlfp_residual(
                    lambda y, t: value + 0.0 * y + 0.0 * t, x, 0.4, q, module.DIFFUSIVITY
                )
            )(jnp.linspace(-2.0, 2.0, 21))
            assert float(jnp.max(jnp.abs(residual))) < 1e-12


def test_network_residual_matches_a_finite_difference(example):
    # The PINN plumbing: the autodiff residual of an arbitrary callable has to be
    # the thing the equation says it is.
    module = example("pinn_fokker_planck")
    q = 1.2

    def smooth(x, t):
        return 0.4 * jnp.exp(-((x / (1.0 + t)) ** 2)) + 0.05

    step = 1e-4
    for x in (-1.3, 0.0, 0.8, 2.5):
        for t in (0.1, 0.6):
            automatic = float(qp.nlfp_residual(smooth, x, t, q, module.DIFFUSIVITY))
            rate = (smooth(x, t + step) - smooth(x, t - step)) / (2.0 * step)
            pressure = [smooth(x + d * step, t) ** (2.0 - q) for d in (-1, 0, 1)]
            curvature = (pressure[0] - 2.0 * pressure[1] + pressure[2]) / step**2
            assert automatic == pytest.approx(
                float(rate - module.DIFFUSIVITY * curvature), rel=1e-4, abs=1e-6
            )


def test_domain_rule_contains_the_front_and_the_mass(example):
    module = example("pinn_fokker_planck")
    for q in module.INDICES:
        half_width = module.domain_half_width(q)
        front = float(qp.nlfp_front(module.FINAL_TIME, q, module.DIFFUSIVITY, module.BETA_INITIAL))
        if np.isfinite(front):
            assert front < half_width
        grid = jnp.linspace(-half_width, half_width, 40_001)
        inside = float(
            jnp.trapezoid(
                qp.nlfp_density(
                    grid, module.FINAL_TIME, q, module.DIFFUSIVITY, module.BETA_INITIAL
                ),
                grid,
            )
        )
        assert 1.0 - inside <= 2.1e-3


def test_truncated_mass_is_not_one_for_a_heavy_tail(example):
    # Charging the network against 1.0 on a truncated domain would bill it for the
    # domain rather than for its own error.
    module = example("pinn_fokker_planck")
    times = jnp.linspace(0.0, module.FINAL_TIME, 5)
    compact = module.truncated_mass(0.5, module.domain_half_width(0.5), times)
    heavy = module.truncated_mass(1.5, module.domain_half_width(1.5), times)
    assert bool(jnp.all(jnp.abs(compact - 1.0) < 1e-6))
    assert float(jnp.max(1.0 - heavy)) > 1e-4
    assert bool(jnp.all(heavy < 1.0))


def test_loss_index_is_confined_above_one(example):
    # Below q = 1 the q-Gaussian has compact support, so the likelihood is -inf
    # for any residual past the cut-off: an infinitely *un*-robust loss, and the
    # opposite of the intended direction.
    module = example("pinn_fokker_planck")
    assert module.Q_LOSS_MIN == 1.0
    for raw in (-40.0, 0.0, 40.0):
        index = float(bounded_q(raw, module.Q_LOSS_MIN, module.Q_LOSS_MAX))
        assert module.Q_LOSS_MIN <= index <= module.Q_LOSS_MAX
    # And the trap is real: a q < 1 likelihood is -inf outside its support.
    assert bool(jnp.isneginf(qjax.q_gaussian_logpdf(10.0, 0.5, 1.0)))


def test_learning_rate_schedule_spans_its_full_range(example):
    # The training scan runs in blocks, and the schedule is a function of the
    # *global* step. Fed the within-block index it would sit pinned at LR for the
    # whole run, which is what this pins against regressing.
    module = example("pinn_fokker_planck")
    total = 6000
    assert float(module.adam_rate(0, total)) == pytest.approx(module.LR)
    assert float(module.adam_rate(total, total)) == pytest.approx(0.01 * module.LR)
    assert float(module.adam_rate(total // 2, total)) == pytest.approx(0.505 * module.LR)
    # Monotone in between, so no block boundary can hand back a larger step.
    rates = jnp.array([float(module.adam_rate(step, total)) for step in range(0, total, 150)])
    assert bool(jnp.all(jnp.diff(rates) < 0.0))


def test_the_two_boundaries_are_penalized_separately(example):
    # Penalizing the sum of the two ends -- the natural one-liner -- would let an
    # error at +L be cancelled by the opposite error at -L for free.
    module = example("pinn_fokker_planck")
    q, times = 1.5, jnp.linspace(0.0, module.FINAL_TIME, 7)
    half_width = module.domain_half_width(q)
    peak = float(qp.nlfp_density(0.0, 0.0, q, module.DIFFUSIVITY, module.BETA_INITIAL))

    def exact(x, t):
        return qp.nlfp_density(x, t, q, module.DIFFUSIVITY, module.BETA_INITIAL)

    def antisymmetric(x, t):
        # Exactly right on average across the two ends, wrong at each of them.
        return exact(x, t) + 0.1 * peak * jnp.sign(x)

    assert (
        float(jnp.max(jnp.abs(module.boundary_errors(exact, times, half_width, q, peak)))) < 1e-12
    )
    errors = module.boundary_errors(antisymmetric, times, half_width, q, peak)
    assert errors.shape == (times.shape[0], 2)
    assert float(jnp.mean(errors**2)) == pytest.approx(0.01, rel=1e-6)
    # The cancelling form would have scored this perfect.
    assert float(jnp.mean(jnp.sum(errors, axis=-1) ** 2)) == pytest.approx(0.0, abs=1e-24)


def test_only_the_deformed_arms_fit_their_residual_scale(example):
    # A Gaussian likelihood with a fitted scale is not a mean-squared residual:
    # as the residual shrinks the fitted beta grows, so the residual term's weight
    # against the fixed initial and boundary weights drifts. The baseline arm
    # therefore holds its scale, and this checks the gradient really is blocked.
    module = example("pinn_fokker_planck")
    q = 1.5
    half_width = module.domain_half_width(q)
    centre, scale = module.output_map(q, half_width)
    peak = float(qp.nlfp_density(0.0, 0.0, q, module.DIFFUSIVITY, module.BETA_INITIAL))
    config = module.configuration(quick=True, full=False)
    points = module.sample_points(jax.random.PRNGKey(0), config, q, half_width)
    params = module.init_params(jax.random.PRNGKey(1), (8, 8), peak)

    def scale_gradient(q_fixed, learnable, learn_scale):
        grads = jax.grad(module.total_loss)(
            params, points, q, q_fixed, learnable, learn_scale, half_width, peak, centre, scale
        )
        return float(grads["beta_raw"])

    assert scale_gradient(1.0, False, False) == 0.0
    assert abs(scale_gradient(1.5, False, True)) > 0.0
    # Every shipped arm is consistent with that rule.
    for _, q_fixed, learnable, learn_scale in module.LOSS_ARMS:
        blocked = scale_gradient(q_fixed if not learnable else 0.0, learnable, learn_scale)
        assert (blocked == 0.0) is (not learn_scale)

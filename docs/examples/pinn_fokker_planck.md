# Heavy-tailed PINN residuals

> A 2026 ICML paper shows that physics-informed network residuals are heavy-tailed
> and fixes it with a Student-t likelihood. That is a Tsallis method with the name
> removed — and reading it as one shows both why it works and where it does not.

## What it shows

Physics-informed neural networks minimize a PDE residual, and standard practice
minimizes its *mean square*. Abijuru et al. (ICML 2026) point out what that
silently assumes — "independent Gaussian residuals with a fixed global variance" —
and show that PINN residuals are instead "heterogeneous and heavy-tailed", so that
"a small number of large residuals can disproportionately dominate both the loss
and gradient". Their remedy is a **Student-t residual model**, fitted by an
expectation–maximization loop.

**The Student-t is the $q$-Gaussian**, and the correspondence is exact at qjax's
own relation $\nu = (3-q)/(q-1)$ — the one already used by `qjax.sample` to draw
$q$-Gaussian variates. Their EM weight

$$w(r) = \frac{\nu+1}{\nu + r^2/s^2}$$

is, term for term, the score of `qjax.q_gaussian_logpdf` divided by the gradient
of a squared residual:

$$\frac{\mathrm d}{\mathrm dr}\big[-\log \mathcal G_{q_L}(r)\big]
= w(r)\,\frac{\mathrm d}{\mathrm dr}\big[\beta r^2\big],
\qquad w(r) = \frac{1}{1 + (q_L-1)\beta r^2}.$$

The two agree to $\sim10^{-15}$, which the test suite pins, and at $q_L = 1$ the
weight is identically one — so **the mean-squared residual is the
Boltzmann–Gibbs member of the family**, not a separate baseline.

What qjax adds is that the entropic index need not be estimated by EM at all. The
EM alternation exists because the tail index is a latent variable; in qjax $q$ is
an ordinary differentiable argument, so one `jax.grad` trains the network *and*
its own residual model together.

Two entropic indices appear and they are unrelated, so they are named apart: $q$
is the **equation's** index, fixed by the physics
($\partial_t p = D\,\partial_{xx}p^{2-q}$), and $q_L$ is the **residual model's**,
learned.

## Result, in three parts

### 1. The premise holds — by more than the original paper claims

Collecting the residuals of a mean-squared-trained PINN and fitting a
$q$-Gaussian by maximum likelihood:

| equation $q$ | $\hat q_L$ from residuals | $\sigma$ | from Gaussian | excess kurtosis | Student-t $\nu$ |
| --- | --- | --- | --- | --- | --- |
| 0.5 | **2.194** | 0.026 | 46σ | 24.4 | 0.67 |
| 1.5 | **2.323** | 0.018 | 73σ | 17.5 | 0.51 |

$\nu < 1$ means **fewer than one degree of freedom — heavier-tailed than a Cauchy
distribution**, so the residuals do not merely have a heavy tail, they have no
finite mean. Because this PDE has a closed-form solution
(`qjax.physics.nlfp_density`, whose own residual vanishes at $10^{-15}$), that is
measured against ground truth rather than against another approximation.

### 2. The remedy backfires here, and the reason is instructive

| equation $q$ | arm | relative $L^2$ | vs MSE |
| --- | --- | --- | --- |
| 0.5 | MSE ($q_L=1$) | **0.0472** | — |
| 0.5 | fixed $q_L = 1.5$ | 0.8961 | +1797 % |
| 0.5 | learnable $q_L$ | 0.8685 | +1739 % |
| 1.5 | MSE ($q_L=1$) | **0.0244** | — |
| 1.5 | fixed $q_L = 1.5$ | 0.0245 | +0.8 % |
| 1.5 | learnable $q_L$ | 0.0276 | +13 % (seed range 45 %) |

A robust loss is, by construction, a loss that **tolerates large residuals**. In a
*forward* PDE problem the residual is not a noise model: it is the only thing
propagating the initial condition into the interior. Downweighting it removes that
force, and the solution decays into the spurious family this equation carries —
*every* spatially uniform density solves $\partial_t p = D\,\partial_{xx}p^{2-q}$
exactly, a fact the test suite checks directly.

The mechanism, measured:

| arm | peak $p(\cdot,0)$ | mass at $t=1$ | RMS residual |
| --- | --- | --- | --- |
| $q=0.5$, MSE | 1.311 | 1.089 | 0.055 |
| $q=0.5$, fixed $q_L=1.5$ | 1.279 | **0.003** | **1.795** |
| $q=0.5$, learnable $q_L$ | 1.272 | **0.005** | 0.438 |
| exact | 1.326 | 1.000 | 0 |

All three arms fit the initial condition. The deformed arms then let the solution
**decay to nothing**, ending with residuals thirty times *larger* than the
mean-squared arm and a loss that does not mind. Where the solution is smooth
($q=1.5$) the effect is neutral; where there is a free boundary ($q=0.5$) it is
catastrophic.

### 3. Joint descent is not EM

The learned index settles at 1.07–1.15 while the index fitted to the residuals
says 2.19–2.32 — a gap of 41σ and 69σ. Alternating (fit the residual model to a
*fixed* network, then the network to a fixed model, as EM does) is not the same as
descending on both at once, because a network free to change its residuals can
move them to suit the index rather than the other way round. That is an argument
*for* the EM structure, arrived at by removing it.

## Result

<figure markdown>
  ![the solution, the residual survival function, the score correspondence, the learned index, the held-out error, and the scoreboard](../img/examples/pinn_fokker_planck.png)
  <figcaption markdown>
  (a) The solution at both equation indices. (b) The ICML claim, measured: the survival function $P(|r|>u)$ of the residuals against the Gaussian a mean-squared objective assumes, and against the fitted $q$-Gaussian. (c) **The correspondence**: the $q$-Gaussian score (coloured) and the Student-t EM weight at $\nu=(3-q_L)/(q_L-1)$ (thick grey) coincide; $q_L=1$ is flat at one, i.e. no reweighting. (d) The learned $q_L$ against the $q_L$ fitted to the residuals (dashed) — they do not meet. (e) Held-out error against the exact solution; the deformed arms at $q=0.5$ never descend. (f) Scoreboard, median and seed range.
  </figcaption>
</figure>

## Validation

| quantity | measured | exact |
| --- | --- | --- |
| residual of the exact solution | $5.8\times10^{-15}$ | 0 |
| … relative to $\max|\partial_t p|$ | $1.3\times10^{-15}$ | 0 |
| $q$-Gaussian score vs. Student-t EM score | $1.8\times10^{-15}$ | 0 |
| weight at $q_L=1$ (no reweighting) | 1.000000 | 1 |

## What this does and does not say

It does **not** refute Abijuru et al. Their setting is heavy tails arising from
heterogeneity and noisy or misspecified data, where downweighting is the right
move, and their EM keeps the two estimates separate — both differences matter, and
both cut in their favour here. The distinction this example draws is narrower and,
we think, generally useful:

**Heavy-tailed residuals do not by themselves license a robust loss.** What
matters is whether the tail is *noise* or *signal*. At a free boundary it is
signal — the residual is large because the solution genuinely has a kink there —
and discounting it means declining to fit the only part of the domain that is
hard.

Two caveats on our side: at 6000 steps the mean-squared arms are still descending,
so this is a comparison at equal budget rather than at convergence; and the loss
weights are fixed across arms, so a deformed arm that downweights its residual
term is also shifting the balance against the initial and boundary terms. That
shift *is* the mechanism, but a practitioner could compensate for it, and we did
not — compensating per arm would have measured the compensation.

## Takeaways

- The Student-t residual model of a 2026 ICML paper is a $q$-Gaussian likelihood,
  exactly, at qjax's own $\nu = (3-q)/(q-1)$ — and the mean-squared residual is
  its $q_L\to1$ member, not a separate thing.
- PINN residuals really are heavy-tailed: $\hat q_L \approx 2.2$–2.3, i.e. $\nu<1$,
  heavier than Cauchy, measured against an exact solution.
- Because `q_gaussian_logpdf` is differentiable in its index, the EM loop is
  unnecessary — but removing it is not free, and the resulting learned index
  disagrees with the fitted one by tens of standard errors.
- A robust residual loss and a forward PDE are a bad match: robustness means
  tolerating large residuals, and in a forward problem the residual is the only
  thing carrying the solution forward.
- As with the [$q\leftrightarrow2-q$ duality](../theory.md#the-q-leftrightarrow-2-q-duality),
  the deformation has a *direction* and a domain of validity, and both are
  checkable in advance.

## References

- J. Abijuru, M. K. Nagda, J. Tauberschmidt, P. S. Ostheimer, S. Vollmer,
  S. Mandt, M. Kloft & S. Fellenz, *Heavy-tailed Physics-Informed Neural
  Networks*, ICML (2026).
- M. Raissi, P. Perdikaris & G. E. Karniadakis, *Physics-informed neural
  networks*, J. Comput. Phys. **378**, 686 (2019).
- C. Tsallis & D. J. Bukman, *Anomalous diffusion in the presence of external
  forces*, Phys. Rev. E **54**, R2197 (1996).
- A. R. Plastino & A. Plastino, Physica A **222**, 347 (1995).
- W. Thistleton, J. A. Marsh, K. Nelson & C. Tsallis, *Generalized Box–Müller
  method for generating q-Gaussian random deviates*, IEEE Trans. Inf. Theory
  **53**, 4805 (2007) — the $\nu \leftrightarrow q$ correspondence qjax uses.

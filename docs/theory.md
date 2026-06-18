# A short primer

This note collects the definitions implemented in `qjax` and their `q → 1`
limits. Throughout, $q \in \mathbf{R}$ is the *entropic index*; `q = 1` recovers ordinary
Boltzmann–Gibbs statistics.

## 1. Deformed logarithm and exponential

The two foundational maps are

$$
\ln_q(x) = \frac{x^{1-q} - 1}{1 - q}, \qquad
\exp_q(x) = \big[1 + (1-q)\,x\big]_+^{\frac{1}{1-q}},
$$

where $[\cdot]_+ = \max(\cdot, 0)$ enforces the **Tsallis cut-off**. They are
mutual inverses and satisfy $\ln_q \to \ln$, $\exp_q \to \exp$ as $q \to 1$.

`qjax` evaluates both with the *double-`where`* trick: the indeterminate
$0/0$ at $q = 1$ is replaced by the analytic limit, and the unused branch is fed
a sanitized argument so that **gradients stay finite everywhere**, including
exactly at $q = 1$.

### q-algebra

The deformed logarithm turns products into a deformed sum, and the deformed
exponential turns sums into a deformed product:

$$
a \oplus_q b = a + b + (1-q)\,ab, \qquad
a \otimes_q b = \big[a^{1-q} + b^{1-q} - 1\big]_+^{\frac{1}{1-q}}.
$$

These give `q_add`/`q_diff` and `q_prod`/`q_div`. They obey
$\ln_q(xy) = \ln_q x \oplus_q \ln_q y$ and
$\exp_q(x+y) = \exp_q x \otimes_q \exp_q y$.

## 2. Tsallis entropy and divergences

The Tsallis entropy of a distribution $p$ is

$$
S_q(p) = \frac{1 - \sum_i p_i^{\,q}}{q - 1}
\;\xrightarrow{q \to 1}\; -\sum_i p_i \ln p_i .
$$

It is concave, non-negative for probability vectors, and maximized by the uniform
distribution. The associated **relative entropy** (q-divergence) is

$$
D_q(p \,\|\, r) = \frac{\sum_i p_i^{\,q} r_i^{1-q} - 1}{q - 1}
\;\xrightarrow{q \to 1}\; \mathrm{KL}(p \,\|\, r),
$$

and the **cross-entropy** used as a classification loss is
$H_q(y, p) = -\sum_i y_i \ln_q p_i$.

## 3. The q-Gaussian

Maximizing $S_q$ under a fixed second moment yields the **q-Gaussian**:

$$
p(x) = \frac{\sqrt{\beta}}{C_q}\,\exp_q\!\big(-\beta x^2\big).
$$

- $q < 1$: compact support.
- $q = 1$: the ordinary Gaussian, $C_1 = \sqrt{\pi}$.
- $1 < q < 3$: heavy (power-law) tails; for $q = 2$ this is the **Cauchy**
  distribution, and in general it is a rescaled **Student-$t$** with
  $\nu = (3-q)/(q-1)$ degrees of freedom.

The variance is finite only for $q < 5/3$, where
$\operatorname{Var} = 1/\big((5 - 3q)\,\beta\big)$. `qjax.sample` exploits the
Student-$t$ relationship, $X = T_\nu / \sqrt{(3-q)\beta}$, which reproduces this
variance exactly (supported for $1 \le q < 3$).

## 4. Tsallis entmax (q-deformed softmax)

Regularizing the maximum-score problem with Tsallis entropy,

$$
\mathrm{entmax}_q(z) = \arg\max_{p \in \Delta}\;
    \langle p, z \rangle + S_q^{T}(p),
$$

gives a probability map with the closed form
$p_i = \big[(q-1) z_i - \tau\big]_+^{1/(q-1)}$, where $\tau$ enforces
$\sum_i p_i = 1$. Then:

- $q = 1$ → **softmax** (dense),
- $q = 2$ → **sparsemax** (sparse; many coordinates are exactly zero),
- intermediate $q$ → a tunable trade-off between smoothness and sparsity.

`qjax` solves for $\tau$ by bisection on the tight bracket
$[(q-1)\max_i z_i - 1,\; (q-1)\max_i z_i]$, which is fully compatible with
`jit`, `grad`, and `vmap`.

## References

- C. Tsallis, *Introduction to Nonextensive Statistical Mechanics*, Springer, 2009.
- W. Thistleton, J. Marsh, K. Nelson, C. Tsallis, "Generalized Box–Müller method
  for generating q-Gaussian random deviates", *IEEE Trans. Inf. Theory*, 2007.
- B. Peters, V. Niculae, A. Martins, "Sparse Sequence-to-Sequence Models"
  (entmax), *ACL*, 2019.

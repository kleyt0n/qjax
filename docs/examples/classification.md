# Label-noise robustness

> Bounded Tsallis cross-entropy (`q < 1`) degrades gracefully under label noise,
> and a *learnable* `q` discovers the robust regime with no grid search.

## What it shows

The same over-parameterized MLP is trained on a 4-class problem while a growing
fraction $\eta$ of the training labels is corrupted at random. The only knob is
the entropic index `q` of the {func}`~qjax.tsallis_cross_entropy` loss, which
replaces the logarithm of ordinary cross-entropy with the deformed `q`-logarithm,

$$
H_q(y, p) = -\sum_i y_i \ln_q p_i
\;\xrightarrow{q \to 1}\; -\sum_i y_i \log p_i .
$$

At $q = 1$ this is *exactly* the Shannon baseline: the per-example loss
$-\log p_c$ is unbounded, so a confidently mislabeled example produces an enormous
gradient and the network memorizes the noise. For $q < 1$ the loss is **bounded**
(the generalized cross-entropy of Zhang & Sabuncu, 2018) and its gradient
saturates on hard or mislabeled points, so clean-set accuracy degrades far more
gracefully. Making `q` **learnable** — training it jointly with the network —
goes one step further: minimizing the bounded loss favors smaller `q` (it
down-weights unfittable points), so `q` *descends* into the robust regime on its
own, with no grid search.

Every configuration is averaged over seeds and evaluated on a **clean** test set.

## How it works

The loss is `tsallis_cross_entropy` on softmax probabilities; making `q`
learnable is just mapping a free parameter into the desired range:

```python
import jax, jax.numpy as jnp
import qjax

def loss_fn(params, x, y_onehot):
    # learnable q in (0.3, 1.3): spans the robust (q<1) and Shannon (q=1) regimes
    q = 0.3 + 1.0 * jax.nn.sigmoid(params["q_raw"])
    p = jnp.clip(jax.nn.softmax(logits(params, x), axis=-1), 1e-7, 1.0)
    return jnp.mean(qjax.tsallis_cross_entropy(p, y_onehot, q=q, axis=-1))
```

Set a fixed `q < 1` instead of the learnable one to get the bounded baselines.

## Result

```{figure} /_static/examples/classification.png
:alt: accuracy vs label noise, the q learning trajectory, and the task
:width: 100%

(a) Clean-test accuracy vs. label-noise rate `η`: bounded losses (`q < 1`) and
the learnable `q` stay far above the Shannon baseline as noise grows. (b) The
learnable `q` descends from its initialization into the robust regime during
training. (c) The synthetic task: four overlapping Gaussian classes.
```

## Decision boundaries across shapes

A second figure (also produced by the script) makes the robustness *visible*. A
compact 3-class classifier is trained on two shapes — **blobs** and a **spiral** —
from clean data up to 40% label noise, comparing the Shannon baseline (`q = 1`)
with the bounded Tsallis loss (`q = 0.3`). The comparison is **fair**: within each
shape both losses share the same initialization, data, noisy labels and optimizer
— only `q` differs.

```{figure} /_static/examples/classification_boundaries.png
:alt: decision boundaries for blobs and spiral across noise levels, BGS vs Tsallis
:width: 100%

Decision regions at 0%, 20% and 40% label noise. The **Tsallis** (robust) columns
are framed in teal. Without noise both losses match (≈98–99%); as noise grows the
**BGS** baseline memorizes the mislabeled points and carves spurious wrong-class
islands, while **Tsallis** keeps clean regions and higher clean-test accuracy.
```

At 0% noise the two losses are indistinguishable (blobs 98% vs 98%, spiral 99% vs
99%) — confirming the baseline is not handicapped. The gap opens only once noise
is added: at 20% noise, **95% vs 78%** (blobs) and **93% vs 83%** (spiral); at 40%,
**67% vs 59%** and **69% vs 55%**.

## Takeaways

A single $q < 1$ turns cross-entropy into a bounded, noise-robust loss, and a
**learnable `q`** discovers that robust regime automatically — matching or
slightly beating the best hand-tuned fixed `q` at every noise level. This mirrors
the other learnable-`q` results in
[reinforcement learning](reinforcement_learning.md) and
[attention](attention_mlp.md): the right amount of non-extensivity is found by
gradient descent rather than by hand.

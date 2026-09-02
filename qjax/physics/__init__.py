"""Physical systems with exact reference values, for validating Tsallis methods.

`qjax` proper is a library of ``q``-deformed primitives. This subpackage is
the counterpart the primitives are *tested against*: a small set of statistical
models -- the 2-D Ising model, the Sherrington-Kirkpatrick spin glass,
Lennard-Jones clusters, anomalous diffusion -- each paired with something exact.

The scope rule is deliberate and worth stating, because it is what keeps the
subpackage small: **this module holds pure, cheap, exactly-testable kernels; the
long runs, the controlled comparisons and the figures live in ``examples/``.**
Nothing that takes more than a second to exercise belongs here. That boundary is
affordable precisely because every kernel has a closed form, an exhaustive
enumeration, or a published high-precision value to check it against -- collected
in `qjax.physics.reference`.

`qjax.physics.annealing` is the one module that is a ``q``-deformation rather
than generic physics: the Tsallis-Stariolo cooling schedule is $0/0$ at
$q = 1$, and `qjax.q_log` resolves it with no branch and a correct
derivative in $q$.

Imported explicitly (``from qjax import physics`` or
``import qjax.physics as qp``) rather than re-exported at top level, so the flat
``qjax.*`` namespace stays the ``q``-primitives.
"""

from qjax.physics import annealing, clusters, diffusion, lattice, observables, reference, spinglass
from qjax.physics.annealing import (
    acceptance_temperature,
    tsallis_schedule,
    visiting_temperature,
)
from qjax.physics.clusters import (
    coordination_numbers,
    equidistant_cluster,
    lj_energy,
    lj_energy_confined,
    lj_quench,
    lj_random_cluster,
)
from qjax.physics.diffusion import (
    NLFP_MAX_INDEX,
    fit_power_law,
    histogram_density,
    interpolate_density,
    lutz_q,
    mean_squared_displacement,
    nlfp_density,
    nlfp_exponent,
    nlfp_front,
    nlfp_index,
    nlfp_offset,
    nlfp_rate,
    nlfp_residual,
    nlfp_scaling_beta,
    nlfp_width,
    saturating_langevin_beta,
    saturating_langevin_q,
)
from qjax.physics.lattice import (
    checkerboard_sweep,
    ising_all_configurations,
    ising_boltzmann_probabilities,
    ising_energy,
    ising_exact_observables,
    ising_magnetization,
    ising_transfer_matrix_log_z,
    metropolis_chain,
    neighbour_sum,
    onsager_energy_per_site,
    onsager_free_energy_per_site,
    onsager_magnetization,
    sample_ising,
    wolff_chain,
    wolff_update,
)
from qjax.physics.observables import (
    binder_cumulant,
    crossing_temperature,
    finite_size_extrapolation,
    half_width,
    peak_temperature,
)
from qjax.physics.reference import (
    ISING_BETA_EXP,
    ISING_ENERGY_AT_TC,
    ISING_NU,
    ISING_TC,
    LJ38_ICOSAHEDRAL,
    LJ_PAIR_DISTANCE,
    LJ_REFERENCE_MINIMA,
    SK_PARISI_GROUND_STATE,
)
from qjax.physics.spinglass import (
    sk_couplings,
    sk_energy,
    sk_exact_correlations,
    sk_exact_observables,
)

__all__ = [
    # submodules
    "annealing",
    "clusters",
    "diffusion",
    "lattice",
    "observables",
    "reference",
    "spinglass",
    # reference values
    "ISING_TC",
    "ISING_BETA_EXP",
    "ISING_NU",
    "ISING_ENERGY_AT_TC",
    "SK_PARISI_GROUND_STATE",
    "LJ_REFERENCE_MINIMA",
    "LJ38_ICOSAHEDRAL",
    "LJ_PAIR_DISTANCE",
    # lattice
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
    # observables
    "binder_cumulant",
    "crossing_temperature",
    "peak_temperature",
    "half_width",
    "finite_size_extrapolation",
    # spin glass
    "sk_couplings",
    "sk_energy",
    "sk_exact_observables",
    "sk_exact_correlations",
    # clusters
    "lj_energy",
    "lj_energy_confined",
    "lj_random_cluster",
    "equidistant_cluster",
    "lj_quench",
    "coordination_numbers",
    # annealing
    "tsallis_schedule",
    "visiting_temperature",
    "acceptance_temperature",
    # diffusion
    "NLFP_MAX_INDEX",
    "nlfp_exponent",
    "nlfp_index",
    "nlfp_scaling_beta",
    "nlfp_rate",
    "nlfp_offset",
    "nlfp_width",
    "nlfp_density",
    "nlfp_front",
    "nlfp_residual",
    "saturating_langevin_q",
    "saturating_langevin_beta",
    "lutz_q",
    "mean_squared_displacement",
    "fit_power_law",
    "histogram_density",
    "interpolate_density",
]

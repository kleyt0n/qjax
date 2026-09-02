r"""Exact and independently-established reference values for the physics examples.

Every number here comes from a closed form, an exhaustive enumeration, or a
published high-precision calculation, and carries its citation. They exist so
that the examples in ``examples/`` can *validate* themselves rather than merely
illustrate: a curve is only evidence if there is something exact to compare it
against.

References:
    Onsager, L. (1944). Crystal statistics I. *Phys. Rev.* **65**, 117.
    Parisi, G. (1980). A sequence of approximated solutions to the SK model.
        *J. Phys. A* **13**, L115.
    Wales, D. J. & Doye, J. P. K. (1997). Global optimization by basin-hopping.
        *J. Phys. Chem. A* **101**, 5111. See also the Cambridge Cluster
        Database, https://www-wales.ch.cam.ac.uk/CCD.html.
    Doye, J. P. K., Miller, M. A. & Wales, D. J. (1999). The double-funnel
        energy landscape of the 38-atom Lennard-Jones cluster.
        *J. Chem. Phys.* **110**, 6896.
"""

from __future__ import annotations

import math

#: Critical temperature of the 2-D Ising model on a square lattice, in units of
#: ``J / k_B``: ``T_c = 2 / ln(1 + sqrt(2))``. Onsager (1944), exact.
ISING_TC: float = 2.0 / math.log1p(math.sqrt(2.0))

#: Exact 2-D Ising magnetization exponent, ``beta = 1/8``.
ISING_BETA_EXP: float = 0.125

#: Exact 2-D Ising correlation-length exponent, ``nu = 1``.
ISING_NU: float = 1.0

#: Internal energy per site at the critical point, ``u(T_c) / J = -sqrt(2)``.
#: Onsager (1944), exact.
ISING_ENERGY_AT_TC: float = -math.sqrt(2.0)

#: Ground-state energy per spin of the Sherrington-Kirkpatrick model in the
#: thermodynamic limit, from Parisi's replica solution. Exact ground states up
#: to ``N = 90`` give ``-0.7637 +/- 0.0004`` (Kobe, 2003), consistent with this.
SK_PARISI_GROUND_STATE: float = -0.7633

#: Global-minimum energies of Lennard-Jones clusters in units of ``epsilon``,
#: keyed by atom count. ``n <= 4`` are closed forms (a regular simplex at
#: ``r = 2**(1/6) sigma`` contributes ``-1`` per pair); the rest are the
#: Cambridge Cluster Database values.
LJ_REFERENCE_MINIMA: dict[int, float] = {
    2: -1.0,  # closed form
    3: -3.0,  # closed form
    4: -6.0,  # closed form
    5: -9.103852,
    6: -12.712062,
    7: -16.505384,
    13: -44.326801,  # Mackay icosahedron
    19: -72.659782,
    38: -173.928427,  # fcc truncated octahedron, point group O_h
}

#: Second-lowest LJ38 minimum, an incomplete Mackay icosahedron. The global
#: minimum is *not* icosahedral, and the two funnels differ by only 0.38 % in
#: energy while the icosahedral basin is far wider -- which is what makes LJ38
#: the standard hard case for global optimization. Doye, Miller & Wales (1999).
LJ38_ICOSAHEDRAL: float = -173.252378

#: Pair separation minimizing the Lennard-Jones potential, ``2**(1/6) sigma``.
LJ_PAIR_DISTANCE: float = 2.0 ** (1.0 / 6.0)

__all__ = [
    "ISING_TC",
    "ISING_BETA_EXP",
    "ISING_NU",
    "ISING_ENERGY_AT_TC",
    "SK_PARISI_GROUND_STATE",
    "LJ_REFERENCE_MINIMA",
    "LJ38_ICOSAHEDRAL",
    "LJ_PAIR_DISTANCE",
]

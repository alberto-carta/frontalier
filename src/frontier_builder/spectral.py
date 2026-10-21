"""
DOS, projected DOS and COHP, from eigenvalues and projectors.

Everything is a weighted sum of Gaussians centred on the band energies, so the
only inputs are `energies` and `proj` of a `PwDump` -- the same two arrays the
rest of the analysis uses.

Conventions
-----------
* The stored weights are per spin channel, so every curve is multiplied by
  `dump.spin_degeneracy`; integrating a DOS over all energies then gives the
  number of electrons per cell.
* COHP between two orbital sets A and B is

      COHP_b(k) = 2 Re sum_{mu in A, nu in B} P^*_{mu b} H_{mu nu}(k) P_{nu b}

  with H_{mu nu}(k) = sum_{b'} eps_{b'k} P_{mu b'} P^*_{nu b'}, which is the
  projected Hamiltonian the projectors themselves define.  Substituting gives a
  band-space expression that needs no H(R):

      COHP_b(k) = 2 sum_{b'} eps_{b'k} Re[ (P_A^dag P_A)_{b b'}
                                           (P_B^dag P_B)_{b' b} ]

  Negative COHP means the A-B matrix elements lower the energy, i.e. bonding.
  `cohp` returns COHP itself; the plots show -COHP so that bonding is positive.
* `band_weights` gives each band to a group of states in proportion to its
  character in that band, which is how a single COHP curve is split between the
  bonding and the antibonding combinations (the shares sum to one per band).
"""

import numpy as np

from .qe_dump import PwDump


def energy_grid(mu: float, below: float = 10.0, above: float = 8.0,
                step: float = 0.01) -> np.ndarray:
    """Energy axis around the Fermi level."""
    return np.arange(mu - below, mu + above + step, step)


def _broaden(grid, energies, coefficients, sigma):
    """sum_k sum_b coefficients[k, b] * gaussian(grid - energies[k, b]).

    Bands that lie more than four sigma outside the grid are skipped, which is
    most of them when the window is a few eV around the Fermi level.
    """
    result = np.zeros_like(grid)
    norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    inside = ((energies.min(axis=0) < grid[-1] + 4 * sigma)
              & (energies.max(axis=0) > grid[0] - 4 * sigma))
    for band in np.flatnonzero(inside):
        offset = (grid[:, None] - energies[:, band][None, :]) / sigma
        result += np.exp(-0.5 * offset ** 2) @ coefficients[:, band]
    return result * norm


def dos(dump: PwDump, grid: np.ndarray, sigma: float) -> np.ndarray:
    """Total DOS, states per eV per cell (both spins)."""
    per_band = np.ones((dump.nk, dump.nbnd))
    coefficients = dump.weights[:, None] * dump.spin_degeneracy * per_band
    return _broaden(grid, dump.energies, coefficients, sigma)


def projected_dos(dump: PwDump, grid: np.ndarray, sigma: float,
                  indices) -> np.ndarray:
    """DOS resolved on the given orbitals (0-based rows of the projector)."""
    weight = np.sum(np.abs(dump.proj[:, :, list(indices)]) ** 2, axis=2)
    coefficients = dump.weights[:, None] * dump.spin_degeneracy * weight
    return _broaden(grid, dump.energies, coefficients, sigma)


def band_weights(dump: PwDump, columns: np.ndarray) -> np.ndarray:
    """|(P U)_{s b}|^2 for every state s of `columns`, shape (nk, s, nbnd).

    `columns` is (nk, norb, n_states) as returned by `bloch_columns`; the
    amplitude of state s in band b is sum_mu P_{b mu} U_{mu s}, the same
    contraction `operators.k_to_R` applies, so that sum_b eps_b |amplitude|^2
    reproduces the state's energy.  The transposed conjugation
    sum_mu conj(U_{mu s}) P_{b mu} is a different object: it scrambles the band
    attribution and moves the states by about 1 eV.
    """
    amplitude = np.einsum("kbm,kmn->knb", dump.proj, columns)
    return np.abs(amplitude) ** 2


def state_dos(dump: PwDump, grid: np.ndarray, sigma: float,
              columns: np.ndarray) -> np.ndarray:
    """DOS of each state in `columns`, shape (n_states, len(grid))."""
    weight = band_weights(dump, columns)
    spectra = np.empty((weight.shape[1], len(grid)))
    for state in range(weight.shape[1]):
        coefficients = (dump.weights[:, None] * dump.spin_degeneracy
                        * weight[:, state, :])
        spectra[state] = _broaden(grid, dump.energies, coefficients, sigma)
    return spectra


def orthonormal(dump: PwDump, columns: np.ndarray) -> np.ndarray:
    """Loewdin-orthonormalised columns, one rotation per k, null space dropped.

    A set of states is only a partial DOS once it is orthonormal.  Fragment
    states are not (they share ligand orbitals), so their per-state weights
    overcount and the rotation is what makes the curves additive.

    The null space has to be tolerated here, which is why this is a
    pseudo-inverse and not `local_model.orthonormal_per_k`: a fragment built
    from several cells contains the same Wannier function in different cells,
    and in the Bloch picture those are one function up to a phase.  In NiO the
    23 states of one octahedron are rank 11 at every k, so the redundant
    directions carry no spectral weight and are zeroed.
    """
    gram = np.einsum("kma,kmb->kab", columns.conj(), columns)
    values, vectors = np.linalg.eigh(gram)
    keep = values > 1e-6 * np.maximum(values.max(axis=1, keepdims=True), 1.0)
    inverse = np.where(keep, 1.0 / np.sqrt(np.where(keep, values, 1.0)), 0.0)
    rotation = np.einsum("kab,kb,kcb->kac", vectors, inverse, vectors.conj())
    return np.einsum("kmn,knp->kmp", columns, rotation)


def group_dos(dump: PwDump, grid: np.ndarray, sigma: float,
              columns: np.ndarray, states) -> np.ndarray:
    """DOS summed over the states `states` of `columns`."""
    weight = band_weights(dump, columns)[:, list(states), :].sum(axis=1)
    coefficients = dump.weights[:, None] * dump.spin_degeneracy * weight
    return _broaden(grid, dump.energies, coefficients, sigma)


def cohp(dump: PwDump, grid: np.ndarray, sigma: float,
         rows, cols, weight: np.ndarray = None) -> np.ndarray:
    """COHP between the orbital sets `rows` and `cols` (0-based).

    `weight`, if given, is a (nk, nbnd) array multiplying the contribution of
    each band -- this is how a group of states is given its share of the bond
    (see `band_weights`).
    """
    first = dump.proj[:, :, list(rows)]
    second = dump.proj[:, :, list(cols)]

    # (P_A^dag P_A)_{b b'} and (P_B^dag P_B)_{b b'}, summed over the orbitals
    left = np.einsum("kbx,kcx->kbc", first.conj(), first)
    right = np.einsum("kbx,kcx->kbc", second.conj(), second)
    per_band = 2.0 * np.einsum("kbc,kc,kcb->kb", left, dump.energies,
                               right).real
    if weight is not None:
        per_band = per_band * weight

    coefficients = dump.weights[:, None] * dump.spin_degeneracy * per_band
    return _broaden(grid, dump.energies, coefficients, sigma)


def character_split(dump: PwDump, columns: np.ndarray, states, rows,
                    threshold: float = 0.5) -> float:
    """Fraction of the weight of `states` that sits in `rows`-dominated bands.

    Each band is classified once, k-averaged: it counts as dominated by `rows`
    when their projection weight is above `threshold`.  Unlike an energy cut
    this needs no knowledge of where the manifolds are.
    """
    character = (np.abs(dump.proj[:, :, list(rows)]) ** 2).sum(axis=2)
    per_band = np.einsum("k,kb->b", dump.weights, character)
    weight = np.einsum("k,ksb->sb", dump.weights, band_weights(dump, columns))
    share = weight[list(states)].sum(axis=0)
    return float(share[per_band > threshold].sum() / share.sum())


def running_integral(grid: np.ndarray, curve: np.ndarray) -> np.ndarray:
    """Cumulative integral of a spectrum on a uniform grid."""
    return np.concatenate([[0.0],
                           np.cumsum(0.5 * (curve[1:] + curve[:-1])
                                     * np.diff(grid))])


def integrate(grid: np.ndarray, curve: np.ndarray,
              energy: float = None) -> float:
    """Integral of a spectrum up to the absolute energy `energy`."""
    stop = len(grid) if energy is None else np.searchsorted(grid, energy) + 1
    if stop < 2:
        return 0.0
    return float(np.trapz(curve[:stop], grid[:stop]))


def at_energy(grid: np.ndarray, curve: np.ndarray, energy: float) -> float:
    """Value of a spectrum at the absolute energy `energy`."""
    return float(np.interp(energy, grid, curve))

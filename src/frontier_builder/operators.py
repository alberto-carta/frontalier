"""
From a `PwDump` to a real-space operator.

One transform does all the work,

    A_{mu nu}(R) = sum_k w_k exp(-2 pi i k.R)
                       sum_b  c_{k b}  P_{mu,k b}  P^*_{nu,k b}

where `c` is a per-band coefficient.  Only `c` changes between the two
quantities we need:

    c = occupations   ->  the occupation matrix  n_{mu nu}(R)
    c = eps_{k b}     ->  the Hamiltonian        H_{mu nu}(R)

The sign of the phase is fixed by the convention in `qe_dump`:

    A(R) = sum_k exp(-2 pi i k.R) A(k)      A(k) = sum_R exp(+2 pi i k.R) A(R)

so that `A(R = 0)` is the plain k-sum, i.e. the local (on-site) operator
n_{mu nu}(R=0) = <mu 0| n |nu 0> of DFT+DMFT.
"""

import numpy as np

from .qe_dump import PwDump


def fermi_dirac(energies, mu, kT):
    """Fermi-Dirac occupations; kT <= 0 gives a step at `mu`."""
    if kT <= 0.0:
        return np.where(energies < mu, 1.0, 0.0)
    x = np.clip((energies - mu) / kT, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(x))


def k_to_R(dump: PwDump, coeff, R=(0, 0, 0), orbital_indices=None,
           transform=None):
    """A_{mu nu}(R) for a per-band coefficient array `coeff` of shape (nk, nbnd).

    `orbital_indices` selects orbitals (0-based, in .amn order); the result is
    then the corresponding square sub-block, of size len(orbital_indices).

    `transform` replaces the surviving projector columns by linear combinations
    of them, and is where a change of basis enters.  It must have shape
    (nk, n_columns, n_new): column m of the result is the combination
    U_{mu m}(k) of the orbitals, so U itself carries the Bloch phase of the
    cells involved.  The transform is applied at the projector level, so no
    real-space matrix is ever rotated.

    For R != 0 the result is not Hermitian -- only A(R)^dagger = A(-R) holds.
    Use R = 0 for a proper local operator.
    """
    if coeff.shape != dump.energies.shape:
        raise ValueError(f"coeff has shape {coeff.shape}, "
                         f"expected {dump.energies.shape} = (nk, nbnd)")
    proj = dump.proj if orbital_indices is None else \
        dump.proj[:, :, list(orbital_indices)]
    if transform is not None:
        if transform.shape[:2] != (dump.nk, proj.shape[2]):
            raise ValueError(f"transform has shape {transform.shape}, expected "
                             f"(nk, {proj.shape[2]}, n_new) = "
                             f"({dump.nk}, {proj.shape[2]}, n_new)")
        proj = np.einsum("kbm,kmn->kbn", proj, transform)

    phase = np.exp(-2j * np.pi * (dump.kfrac @ np.asarray(R, dtype=float)))
    weighted = (dump.weights * phase)[:, None] * coeff       # (nk, nbnd)
    return np.einsum("kb,kbm,kbn->mn", weighted, proj, proj.conj())


def density_matrix(dump: PwDump, mu: float, kT: float,
                   R=(0, 0, 0), orbital_indices=None, transform=None):
    """Occupation matrix n_{mu nu}(R), traced with the Fermi-Dirac weights."""
    return k_to_R(dump, fermi_dirac(dump.energies, mu, kT), R, orbital_indices,
                  transform)


def hamiltonian(dump: PwDump, R=(0, 0, 0), orbital_indices=None,
                transform=None):
    """H_{mu nu}(R) from the band energies (no occupations), in eV.

    This is `sum_k w_k e^{-2 pi i k.R} sum_b eps_{k b} P P^*`, the same
    quantity projwfc_nloc.py prints for R = 0.
    """
    return k_to_R(dump, dump.energies, R, orbital_indices, transform)

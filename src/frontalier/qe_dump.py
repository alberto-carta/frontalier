"""
Readers for the three files written by the patched `projwfc.x`.

    <prefix>.kpt   ik  kx ky kz  kxf kyf kzf  wk
    <prefix>.eig   ik  ibnd  energy[eV]
    <prefix>.amn   ik  ibnd  iorb  Re P  Im P

Conventions, stated once and used everywhere else
-------------------------------------------------
* `k` is in crystal (fractional) coordinates.
* energies and the Fermi level are in eV.
* weights are normalised per spin channel, `sum_k w_k = 1`.  The raw dump of a
  non-magnetic run carries the spin degeneracy (`sum_k w_k = 2`), which
  `load(..., spin_degeneracy=2)` divides out.
* projectors are `P[ik, iband, iorb] = <phi_iorb | psi_{ik,iband}>`, complex,
  with the orbital ordering of the `.amn` header (0-based here, 1-based in the
  file).
* the Hilbert space of the dump is the set of atomic orbitals listed in the
  `.amn`; a band projector is `sum_iorb |P|^2 <= 1`.

Everything below is a thin wrapper: parse, check the shape, return an array.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from . import timing
from .orbitals import Orbital, read_orbitals


@dataclass(frozen=True)
class PwDump:
    """A projwfc dump: k-points, weights, band energies and projectors."""

    kfrac: np.ndarray        # (nk, 3)   crystal coordinates
    weights: np.ndarray      # (nk,)     normalised, sum = 1
    energies: np.ndarray     # (nk, nbnd)  eV
    proj: np.ndarray         # (nk, nbnd, norb) complex
    orbitals: List[Orbital]
    nbands_total: int        # bands present in the file before any selection
    spin_degeneracy: float = 2.0   # weights are per spin channel

    @property
    def nk(self) -> int:
        return self.kfrac.shape[0]

    @property
    def nbnd(self) -> int:
        return self.energies.shape[1]

    @property
    def norb(self) -> int:
        return self.proj.shape[2]


def _loadtxt(path: str, ncol: int) -> np.ndarray:
    """np.loadtxt with a message that names the file when it goes wrong."""
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.size == 0:
        raise ValueError(f"{path}: file is empty")
    if data.shape[1] != ncol:
        raise ValueError(f"{path}: expected {ncol} columns, found {data.shape[1]}")
    return data


def grid_shape(kfrac: np.ndarray, weights: np.ndarray = None) -> tuple:
    """(n1, n2, n3) of the k mesh, read off the k-points themselves.

    Nothing is assumed to be cubic: a tetragonal 3x3x5 mesh has three different
    counts, and the number of distinct k values per direction gives them.

    Two conditions have to hold, and the counts alone are not enough: the set
    must be the full product grid, and every point must carry the same weight.
    A symmetry-reduced set can have unique counts that multiply to its size by
    coincidence (48 points of a 7x7x5 mesh look like a 4x4x3 grid) while its
    weights differ by the multiplicities, and then it is not a mesh whose dual
    is a box of cells.
    """
    counts = tuple(len(np.unique(np.round(kfrac[:, axis], 10)))
                   for axis in range(3))
    if int(np.prod(counts)) != len(kfrac):
        raise ValueError(
            f"the {len(kfrac)} k-points are not a {counts[0]}x{counts[1]}x"
            f"{counts[2]} grid -- a symmetry-reduced set cannot be dual to a "
            f"box of cells, use the full mesh")
    if weights is not None and not np.allclose(weights, weights[0]):
        raise ValueError(
            f"the {len(kfrac)} k-points do not carry equal weights "
            f"({weights.min():.6f} .. {weights.max():.6f}): the set is "
            f"symmetry-reduced, so its dual is not a box of cells")
    return counts


def mesh_shape(dump: PwDump) -> tuple:
    """(n1, n2, n3) of the k mesh of a dump -- see `grid_shape`."""
    return grid_shape(dump.kfrac, dump.weights)


def read_kpoints(path: str):
    """-> (kfrac (nk,3), weights (nk,))."""
    data = _loadtxt(path, 8)
    return data[:, 4:7], data[:, 7]


def read_eigenvalues(path: str) -> np.ndarray:
    """-> energies (nk, nbnd) in eV."""
    data = _loadtxt(path, 3)
    nk = int(data[:, 0].max())
    nbnd = int(data[:, 1].max())
    if len(data) != nk * nbnd:
        raise ValueError(f"{path}: {len(data)} rows, expected nk*nbnd = {nk * nbnd}")
    energies = np.zeros((nk, nbnd))
    energies[data[:, 0].astype(int) - 1, data[:, 1].astype(int) - 1] = data[:, 2]
    return energies


def read_projections(path: str) -> np.ndarray:
    """-> proj (nk, nbnd, norb) complex."""
    data = _loadtxt(path, 5)
    nk = int(data[:, 0].max())
    nbnd = int(data[:, 1].max())
    norb = int(data[:, 2].max())
    if len(data) != nk * nbnd * norb:
        raise ValueError(f"{path}: {len(data)} rows, expected nk*nbnd*norb"
                         f" = {nk * nbnd * norb}")
    proj = np.zeros((nk, nbnd, norb), dtype=complex)
    proj[data[:, 0].astype(int) - 1,
         data[:, 1].astype(int) - 1,
         data[:, 2].astype(int) - 1] = data[:, 3] + 1j * data[:, 4]
    return proj


def load(prefix: str, spin_degeneracy: int = 2) -> PwDump:
    """Read <prefix>.kpt / .eig / .amn into one object.

    `spin_degeneracy` is the number of spin channels the weights already carry:
    2 for a non-magnetic run (sum_k w_k = 2 in the file, 1 after normalisation).
    """
    with timing.stage(f"reading the projector dump   ({prefix}.kpt/.eig/.amn)"):
        kfrac, weights = read_kpoints(f"{prefix}.kpt")
        energies = read_eigenvalues(f"{prefix}.eig")
        proj = read_projections(f"{prefix}.amn")
        orbitals = read_orbitals(f"{prefix}.amn")

    if not energies.shape[0] == proj.shape[0] == kfrac.shape[0]:
        raise ValueError(f"{prefix}: k-points disagree between .kpt/.eig/.amn "
                         f"({kfrac.shape[0]}/{energies.shape[0]}/{proj.shape[0]})")
    if energies.shape[1] != proj.shape[1]:
        raise ValueError(f"{prefix}: bands disagree between .eig/.amn "
                         f"({energies.shape[1]} vs {proj.shape[1]})")
    if proj.shape[2] != len(orbitals):
        raise ValueError(f"{prefix}: {proj.shape[2]} projectors but "
                         f"{len(orbitals)} labelled orbitals")
    total_weight = weights.sum()
    if not np.isclose(total_weight, spin_degeneracy, rtol=1e-6):
        raise ValueError(f"{prefix}: sum_k w_k = {total_weight:.6f}, "
                         f"expected the spin degeneracy {spin_degeneracy}")

    return PwDump(kfrac=kfrac, weights=weights / spin_degeneracy,
                  energies=energies, proj=proj, orbitals=orbitals,
                  nbands_total=energies.shape[1],
                  spin_degeneracy=float(spin_degeneracy))

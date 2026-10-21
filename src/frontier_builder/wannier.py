"""
Readers for a wannier90 run, used the same way as the projwfc dump.

The analysis reads **eigenvalues and projectors**, exactly as for the pw dump:

    <prefix>.eig    band eigenvalues, written as 'band k energy'
    <prefix>_u.mat  the Wannier rotation U(k)

and the projector of band b on Wannier function n is

    P[k, b, n] = <w_nk | psi_bk> = conj(U[k, b, n])

With that P and the .eig energies, `operators` builds n(R) and H(R) exactly as
in the ortho-atomic case -- nothing else changes downstream.  Checked: the
H(R) rebuilt this way reproduces <prefix>_hr.dat to 1e-15 eV, so the _hr.dat is
a *check* on the construction, not its input.

Orbital labels
--------------
A Wannier function is called d or p because of how it was built: the projection
list of the wannier90 input.  That is the "different definition" of d and p
compared with the pw dump, where the label comes from the .amn header.
`WF_SHELLS` below is that list; its order is confirmed by the on-site energies
of H(R=0), which split as 10 d-like (~13.2 eV) and 6 p-like (~9.84 eV).
"""

from typing import Dict, List, Tuple

import numpy as np
from pymatgen.core import Lattice, Structure

from . import qe_dump
from .orbitals import Orbital
from .qe_dump import PwDump

#  (atom, l, number of functions, n) in the order wannier90 was given them.
WF_SHELLS = [(1, 2, 5, 3),     # Ni1 d
             (2, 2, 5, 3),     # Ni2 d
             (3, 1, 3, 2),     # O3  p
             (4, 1, 3, 2)]     # O4  p


def wf_orbitals(shells=WF_SHELLS) -> List[Orbital]:
    """The Wannier functions as `Orbital` entries, so the usual tools apply."""
    orbitals, index = [], 0
    for atom, l, count, n in shells:
        label = f"Ni{atom}" if l == 2 else "O"
        for m in range(1, count + 1):
            orbitals.append(Orbital(index=index, atom=atom, label=label,
                                    n=n, l=l, m=m))
            index += 1
    return orbitals


def read_hr_dat(path: str):
    """-> (num_wann, {R: H(R)}, {R: ndegen}) from a 7-column <prefix>_hr.dat.

    The file stores H(R)/ndegen(R) (the standard wannier90 convention), so the
    degeneracies are returned as well: H(k) = sum_R exp(ikR) H_file(R) / ndegen.
    """
    lines = open(path).read().splitlines()
    nw, nrpts = int(lines[1]), int(lines[2])
    header = " ".join(lines[3:3 + (nrpts + 14) // 15]).split()
    degeneracy = [int(value) for value in header][:nrpts]

    body = [line.split() for line in lines[3 + (nrpts + 14) // 15:] if line.strip()]
    values = np.asarray(body, dtype=float)
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError(f"{path}: expected 7 columns per line, found "
                         f"{values.shape[1] if values.ndim == 2 else '?'}")

    hamiltonian: Dict[Tuple[int, int, int], np.ndarray] = {}
    ndegen: Dict[Tuple[int, int, int], int] = {}
    for r1, r2, r3, m, n, real, imag in values:
        R = (int(r1), int(r2), int(r3))
        if R not in hamiltonian:
            hamiltonian[R] = np.zeros((nw, nw), dtype=complex)
            ndegen[R] = degeneracy[len(hamiltonian) - 1]
        hamiltonian[R][int(m) - 1, int(n) - 1] = real + 1j * imag
    return nw, hamiltonian, ndegen


def read_eig(path: str) -> np.ndarray:
    """wannier90 .eig, written as 'band k energy' -> (nk, nbnd) in eV."""
    data = np.loadtxt(path)
    nk = int(data[:, 1].max())
    nbnd = int(data[:, 0].max())
    energies = np.zeros((nk, nbnd))
    energies[data[:, 1].astype(int) - 1, data[:, 0].astype(int) - 1] = data[:, 2]
    return energies


def read_amn(path: str) -> np.ndarray:
    """wannier90 .amn -> A[k, band, projector], shape (nk, num_bands, nproj).

    Layout: a comment line, 'num_bands num_kpoints nproj', then one line per
    (band, projector, k) of '<m> <n> <k> Re Im'.

    The values are <phi_n|psi_mk> for the projection orbitals of *that* run, so
    these are complex amplitudes in the same band gauge as the `_u.mat` of the
    same run.  This is what makes a real-space decomposition of a Wannier
    function possible: the projwfc dump of a different nscf is not in this
    gauge, and mixing the two produces noise rather than a smaller number.
    """
    with open(path) as handle:
        next(handle)                                   # creation stamp
        num_bands, nk, nproj = [int(value) for value in next(handle).split()]
        data = np.loadtxt(handle)
    amplitude = np.zeros((nk, num_bands, nproj), dtype=complex)
    amplitude[data[:, 2].astype(int) - 1,
              data[:, 0].astype(int) - 1,
              data[:, 1].astype(int) - 1] = data[:, 3] + 1j * data[:, 4]
    return amplitude


def read_nnkp_projections(path: str):
    """(fractional centres, l, mr) of the projection list in a wannier90 .nnkp.

    Each projection occupies two lines: the centre (fractional coordinates) with
    l, mr and r, then the z axis, the x axis and zona.
    """
    lines = open(path).read().splitlines()
    start = next(index for index, line in enumerate(lines)
                 if line.strip().lower().startswith("begin projections"))
    count = int(lines[start + 1].split()[0])
    centres, angular, magnetic = [], [], []
    for index in range(count):
        parts = lines[start + 2 + 2 * index].split()
        centres.append([float(value) for value in parts[:3]])
        angular.append(int(parts[3]))
        magnetic.append(int(parts[4]))
    return np.array(centres), np.array(angular), np.array(magnetic)


def read_win_kpoints(path: str) -> np.ndarray:
    """The k points of a `begin kpoints ... end kpoints` block, shape (nk, 3).

    A .win listing its k points defines the order of the `.amn` and `_u.mat` of
    that run, which is not necessarily the order of the nscf `.kpt`.
    """
    lines = open(path).read().splitlines()
    start = next(index for index, line in enumerate(lines)
                 if line.strip().lower().startswith("begin kpoints"))
    points = []
    for line in lines[start + 1:]:
        if line.strip().lower().startswith("end"):
            break
        if line.split():
            points.append([float(value) for value in line.split()[:3]])
    return np.array(points)


def read_win_structure(path: str) -> Structure:
    """The cell of a run, read from its own `.win`.

    A wannier90 input carries the geometry it was run with, so an analysis of
    two runs needs no pw input file at all.  The atom order is the one the run
    used, which is the order of the projection list and therefore of the Wannier
    functions -- exactly what the `atom` index of `Orbital` must match.

    Both blocks are read: `unit_cell_cart` (with its optional unit line, `Ang`
    or `Bohr`) and `atoms_frac`.  `alat` is refused: the lattice parameter is
    not in the file.
    """
    lines = open(path).read().splitlines()
    stripped = [line.strip() for line in lines]

    def block(name: str):
        start = next(index for index, line in enumerate(stripped)
                     if line.lower().startswith(f"begin {name}"))
        rows, index = [], start + 1
        while not stripped[index].lower().startswith("end"):
            if stripped[index]:                 # the blocks contain blank lines
                rows.append(stripped[index])
            index += 1
        return rows

    cell = block("unit_cell_cart")
    scale = 1.0
    if cell[0] and cell[0][0].isalpha():
        unit = cell[0].lower()
        if unit.startswith("bohr"):
            scale = 0.529177210903
        elif not unit.startswith("ang"):
            raise ValueError(f"{path}: unit '{cell[0]}' in unit_cell_cart is not "
                             f"supported (Ang or Bohr)")
        cell = cell[1:]
    vectors = np.array([[float(x) for x in row.split()[:3]] for row in cell])

    species, positions = [], []
    for row in block("atoms_frac"):
        parts = row.split()
        species.append(parts[0])
        positions.append([float(x) for x in parts[1:4]])
    return Structure(Lattice(vectors * scale), species, positions)


def read_spreads(path: str):
    """Final WF centres and spreads out of a wannier90 .wout.

    -> (centres (nw, 3) cartesian, spreads (nw,) in Angstrom^2).  The file
    prints one block per iteration, so only the last block is kept.

    This is the one spatial number a Wannierisation gives away for free: the
    spread <r^2> - |<r>|^2 does not need the projectors and does not depend on
    the gauge, so it can be compared with a model even when the underlying
    runs used different band phases.
    """
    centres, spreads, labels = [], [], []
    for line in open(path):
        if "WF centre and spread" not in line:
            continue
        labels.append(int(line.split("WF centre and spread")[1].split()[0]))
        numbers = line.split("(")[-1].split(")")[0].replace(",", " ").split()
        centres.append([float(value) for value in numbers])
        spreads.append(float(line.split(")")[-1].split()[0]))

    if not centres:
        raise ValueError(f"no 'WF centre and spread' lines in {path}")
    # the file repeats the block at every iteration; the last block ends with
    # the highest index, and that index is the number of Wannier functions
    last = labels[-1]
    centres, spreads = centres[-last:], spreads[-last:]
    if labels[-last:] != list(range(1, last + 1)):
        raise ValueError(f"the last block in {path} is incomplete")
    return np.asarray(centres), np.asarray(spreads)


def read_u_mat(path: str):
    """wannier90 _u.mat / _u_dis.mat -> (kpoints (nk,3), rotations (nk, rows, cols)).

    Layout: a comment line, 'nk rows cols', a blank line, then per k a line with
    the three k components followed by rows*cols lines of 'Re Im'.  The blocks
    are unitary rotations (checked to 2e-10 on the file used here).

    `_u.mat` is square (num_wann x num_wann); `_u_dis.mat` is not -- it has the
    bands as rows and the Wannier functions as columns -- so both dimensions are
    read from the header rather than assumed equal.
    """
    header = open(path).read().splitlines()
    rows, cols = (int(value) for value in header[1].split()[1:3])
    lines = [line for line in header[2:] if line.strip()]

    kpoints, rotations, current, index = [], [], None, 0
    for line in lines:
        parts = line.split()
        if len(parts) == 3:
            if current is not None:
                rotations.append(current)
            current = np.zeros((rows, cols), dtype=complex)
            index = 0
            kpoints.append([float(x) for x in parts])
        else:
            current[index // cols, index % cols] = \
                float(parts[0]) + 1j * float(parts[1])
            index += 1
    rotations.append(current)
    return np.asarray(kpoints), np.asarray(rotations)


def read_kpoint_weights(path: str):
    """k-points and weights of a run, from its `.win` or from a QE `.kpt`.

    The list inside the `.win` is what wannier90 actually used, so it is the
    right thing to pair with a `.eig` and a `_u.mat`: the row order matches by
    construction.  A QE `.kpt` is still accepted, for runs driven from a pw
    dump, but note that `<seed>_band.kpt` is a *band path* written by the kpath
    tool and must not be used here.
    """
    if path.endswith(".win"):
        points = read_win_kpoints(path)
        return points, np.full(len(points), 1.0 / len(points))
    return qe_dump.read_kpoints(path)


def projector_dump(eig_file: str, u_file: str, kpt_file: str) -> PwDump:
    """A `PwDump` built from the eigenvalues and the Wannier rotation.

    The projectors are `conj(U)`, i.e. P[k, b, n] = <w_nk|psi_bk>.  The k-points
    and weights come from the `.win` (or a QE `.kpt`) of the same mesh; the
    rotation blocks and the .eig rows follow that same k ordering.
    """
    energies = read_eig(eig_file)
    _, rotation = read_u_mat(u_file)
    projectors = np.conj(np.transpose(rotation, (0, 2, 1)))
    kfrac, raw_weights = read_kpoint_weights(kpt_file)

    if not (len(kfrac) == energies.shape[0] == projectors.shape[0]):
        raise ValueError(f"k-points disagree: .kpt {len(kfrac)}, .eig "
                         f"{energies.shape[0]}, _u.mat {projectors.shape[0]}")
    if energies.shape[1] != projectors.shape[2]:
        raise ValueError(f"bands disagree: .eig {energies.shape[1]}, "
                         f"_u.mat {projectors.shape[2]}")

    return PwDump(kfrac=kfrac, weights=raw_weights / raw_weights.sum(),
                  energies=energies, proj=projectors,
                  orbitals=wf_orbitals(), nbands_total=energies.shape[1],
                  spin_degeneracy=2.0)

"""
The local model of one fragment: its Hamiltonian and its eigenstates.

A fragment is an origin atom plus its coordination shell (see `fragments`).
Its basis states are the orbitals of the origin in cell 0 and the orbitals of
each neighbour in the cell where the neighbour was found:

    fragment basis = { (orbital mu, cell R) }

so the fragment Hamiltonian follows from the crystal one by

    H[{mu, R_a}, {nu, R_b}] = H_{mu nu}(R_b - R_a)

with the convention of `operators`, H_{mu nu}(R) = <mu 0|H|nu R>.  Nothing is
orthogonalised here: the fragment lives in the orthonormal ortho-atomic basis,
so sub-diagonalising it is a plain `eigh`.

Sub-diagonalisation returns, for every eigenstate, the coefficients on the
fragment states; `mixing` turns those into weights |c|^2, which is the measure
of how each molecular state is built out of the fragment's atomic states.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import fragments, operators, timing
from .fragments import Neighbour
from .orbitals import Orbital, select
from .qe_dump import PwDump

ORIGIN_CELL = (0, 0, 0)


@dataclass(frozen=True)
class FragmentState:
    """One basis state of a fragment: an orbital sitting in a given cell."""

    orbital: Orbital
    image: Tuple[int, int, int]

    @property
    def label(self) -> str:
        """'#5 Ni1 3dz2' in the origin cell, '#22 O3 2pz R(-1, 0, 1)' elsewhere."""
        if not any(self.image):
            return f"#{self.orbital.index + 1} {self.orbital.name}"
        return f"#{self.orbital.index + 1} {self.orbital.name} R{self.image}"


def _orbitals_by_atom(dump: PwDump, selection: Sequence[str]):
    """{atom index: [Orbital]} restricted to the requested shells."""
    chosen = set(select(dump.orbitals, selection))
    by_atom = {}
    for orbital in dump.orbitals:
        if orbital.index in chosen:
            by_atom.setdefault(orbital.atom, []).append(orbital)
    return by_atom


def fragment_states(dump: PwDump, origin_site: int, shell: Sequence[Neighbour],
                    selection: Sequence[str]) -> List[FragmentState]:
    """Origin orbitals in cell 0, then each neighbour's orbitals in its cell.

    `origin_site` is the 1-based .amn atom index; `selection` decides which
    shells enter the fragment (and therefore what gets sub-diagonalised).
    """
    by_atom = _orbitals_by_atom(dump, selection)
    if origin_site not in by_atom:
        raise ValueError(f"selection {list(selection)} contains no orbital of "
                         f"atom {origin_site}, which defines the fragment")

    states = [FragmentState(orbital, ORIGIN_CELL) for orbital in by_atom[origin_site]]
    for neighbour in shell:
        states += [FragmentState(orbital, neighbour.image)
                   for orbital in by_atom.get(neighbour.site, [])]
    return states


def fragment_hamiltonian(dump: PwDump,
                         states: Sequence[FragmentState]) -> np.ndarray:
    """H_{ab} for the fragment, in eV, shape (len(states), len(states)).

    The H(R) blocks are built for the fragment's own orbitals only.  Asking for
    the full orbital array costs O(norb^2) per R vector instead of O(n_states^2)
    -- on a 168-orbital cell with 20-atom fragments that is 50x more work for
    rows and columns that are then thrown away (12 s per fragment instead of
    0.3 s).
    """
    indices = sorted({state.orbital.index for state in states})
    position = {index: number for number, index in enumerate(indices)}
    offsets = {tuple(int(x) for x in np.subtract(b.image, a.image))
               for a in states for b in states}

    with timing.timed(f"H(R) blocks, {len(offsets)} offsets x "
                      f"{len(indices)} orbitals"):
        blocks = {offset: operators.hamiltonian(dump, R=offset,
                                               orbital_indices=indices)
                  for offset in offsets}

    matrix = np.zeros((len(states), len(states)), dtype=complex)
    for row, a in enumerate(states):
        for col, b in enumerate(states):
            offset = tuple(int(x) for x in np.subtract(b.image, a.image))
            matrix[row, col] = blocks[offset][position[a.orbital.index],
                                              position[b.orbital.index]]
    return matrix


def diagonalise(matrix: np.ndarray):
    """(eigenvalues, eigenvectors as columns) of a Hermitian matrix."""
    hermitian = 0.5 * (matrix + matrix.conj().T)
    return np.linalg.eigh(hermitian)


def mixing(eigenvectors: np.ndarray) -> np.ndarray:
    """|c_{mu e}|^2: shape (n_fragment_states, n_eigenstates)."""
    return np.abs(eigenvectors) ** 2


def shell_mixing(states: Sequence[FragmentState],
                 eigenvectors: np.ndarray):
    """(shell names, table): weights summed per shell, shape (n_shells, n_states).

    'Ni 3d' / 'O 2p' are the coarse channels that tell how much of an
    eigenstate belongs to the origin and how much to the ligands.
    """
    weight = mixing(eigenvectors)
    shells: List[str] = []
    for state in states:
        if state.orbital.shell not in shells:
            shells.append(state.orbital.shell)

    table = np.zeros((len(shells), weight.shape[1]))
    for index, state in enumerate(states):
        table[shells.index(state.orbital.shell)] += weight[index]
    return shells, table


def leading_states(states: Sequence[FragmentState], weights: np.ndarray,
                   count: int = 3) -> List[Tuple[int, float]]:
    """The `count` largest (1-based fragment index, weight) pairs of one eigenstate."""
    order = np.argsort(weights)[::-1][:count]
    return [(int(index) + 1, float(weights[index])) for index in order if weights[index] > 1e-6]


def highest_states(eigenvalues: np.ndarray, eigenvectors: np.ndarray,
                   count: int):
    """The `count` highest eigenstates, returned lowest-first.

    -> (energies, vectors, original column indices), with the vectors of shape
    (n_fragment_states, count).
    """
    order = np.argsort(eigenvalues)[::-1][:count]
    order = order[np.argsort(eigenvalues[order])]
    return eigenvalues[order], eigenvectors[:, order], order


def d_character(model: "Fragment") -> np.ndarray:
    """Weight of the d orbitals in each fragment eigenstate."""
    weight = np.abs(model.eigenvectors) ** 2
    is_d = np.array([state.orbital.l == 2 for state in model.states])
    return weight[is_d].sum(axis=0)


def character_groups(model: "Fragment", count: int):
    """Split the fragment states into (antibonding, bonding, ligand) indices.

    The splitting is by d character, exactly as in the frontier model: the
    `count` most d-like states are the antibonding partners of the origin, the
    next `count` are the p-derived bonding partners, the rest is the remaining
    ligand manifold.
    """
    order = np.argsort(d_character(model))[::-1]
    return (list(order[:count]), list(order[count:2 * count]),
            list(order[2 * count:]))


def bloch_columns(states: Sequence[FragmentState], eigenvectors: np.ndarray,
                  columns: Sequence[int], kfrac: np.ndarray,
                  norb: int) -> np.ndarray:
    """U(k) of the chosen fragment eigenstates, shape (nk, norb, len(columns)).

    `k_to_R` multiplies the projectors from the right by this matrix, so its
    columns are the *bra* coefficients: the projector of a state on the orbital
    mu in cell R is conj(c) * exp(+2 pi i k.R) times P_mu(k).  Both the
    conjugation and the sign are needed -- with exp(-2 pi i k.R) the on-site
    energy of a fragment state comes out 0.49 eV wrong with this cell.

    Checked: k_to_R(..., transform=U)[m, m] reproduces the fragment
    eigenvalues of the chosen columns to 5e-14.
    """
    coefficient = np.zeros((len(kfrac), norb, len(columns)), dtype=complex)
    for position, column in enumerate(columns):
        for index, state in enumerate(states):
            phase = np.exp(+2j * np.pi * (kfrac @ fragments.image_array(state)))
            coefficient[:, state.orbital.index, position] += \
                np.conj(eigenvectors[index, column]) * phase
    return coefficient


def overlap(coefficient: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """S = sum_k w_k U(k)^dagger U(k), the k-averaged overlap of the columns."""
    return np.einsum("k,kam,kan->mn", weights, coefficient.conj(), coefficient)


def orthonormal_per_k(columns: np.ndarray, tolerance: float = 1e-8,
                      verbose: bool = True) -> np.ndarray:
    """Loewdin inside each k: U(k) -> U(k) S(k)^{-1/2}, S(k) = U(k)^dag U(k).

    This is the only correct way to orthonormalise a set of model states.  The
    states of different origins overlap through the shared ligands, so a
    *k-averaged* rotation leaves them orthonormal on average only; then at any
    single k they are not a projector, and every occupation built from them --
    n_occ, its trace, a partial DOS -- is undercounted.  It is a quiet error: it
    changed 7.98 electrons into 7.65 and an e_g occupation of 0.520 into 0.452.

    Raises when the set is linearly dependent, which is what happens when the
    fragment grows until its functions stop being independent.

    `verbose` prints the overlap spectrum of every k (default: on when
    FRONTALIER_TIMING is set).  There is no iteration to follow -- it is one
    symmetric diagonalisation per k, not a self-consistent step -- so what is
    worth watching is how far S is from the unit matrix, i.e. how hard the
    rotation is actually working.
    """
    gram = np.einsum("kma,kmb->kab", columns.conj(), columns)
    values, vectors = np.linalg.eigh(gram)
    if verbose:
        spread = values[:, -1] / values[:, 0]
        print(f"   {gram.shape[0]} rotations of {gram.shape[-1]} states"
              f"   S eigenvalues {values.min():.6f} .. {values.max():.6f}"
              f"   worst condition {spread.max():.6f}", flush=True)
        stride = max(1, len(gram) // 20)
        for index in range(0, len(gram), stride):
            print(f"      k {index + 1:4d}   S {values[index, 0]:9.6f} .. "
                  f"{values[index, -1]:9.6f}   condition {spread[index]:9.6f}",
                  flush=True)
        if stride > 1:
            print(f"      (every {stride}th of {len(gram)} k-points)", flush=True)
    if values.min() <= tolerance * max(values.max(), 1.0):
        raise ValueError(
            f"the {columns.shape[2]} states are linearly dependent at some k "
            f"(smallest overlap eigenvalue {values.min():.1e}): the fragment "
            f"is too redundant to define them, try a smaller radius")
    rotation = np.einsum("kab,kb,kcb->kac", vectors, 1.0 / np.sqrt(values),
                         vectors.conj())
    return np.einsum("kmn,knp->kmp", columns, rotation)


def off_diagonal_max(matrix: np.ndarray) -> float:
    """Largest off-diagonal element, a check that a diagonalisation worked."""
    size = matrix.shape[0]
    mask = ~np.eye(size, dtype=bool)
    return float(np.abs(matrix[mask]).max()) if size > 1 else 0.0


# --------------------------------------------------------------------------
#  the two models a driver script asks for
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Fragment:
    """One origin, its coordination shell, and its sub-diagonalised model."""

    origin: str
    radius: float
    selection: List[str]
    shell: List[Neighbour]
    states: List[FragmentState]
    hamiltonian: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray

    @property
    def size(self) -> int:
        return len(self.states)


def fragment(cell, dump: PwDump, origin: str, radius: float,
             selection: Sequence[str], coupling: Optional[float] = None) -> Fragment:
    """Geometry + local Hamiltonian of the fragment around `origin`.

    `coupling` rescales every hopping between *different* shells -- in practice
    the d-p one -- and leaves d-d and p-p untouched.  It is the knob used to
    bring the model to a prescribed filling: 1.0 is the octahedron as it comes
    out of the projection, 0.0 switches hybridisation off altogether.
    """
    with timing.stage(f"identifying the fragment of {origin}"
                      f"   (radius {radius} A, {list(selection)})"):
        shell = fragments.coordination_shell(cell, origin, radius)
        origin_site = fragments.resolve_atoms(cell, [origin])[0] + 1
        with timing.timed(f"fragment {origin}: basis"):
            states = fragment_states(dump, origin_site, shell, selection)
        with timing.timed(f"fragment {origin}: H(R) blocks"):
            hamiltonian = fragment_hamiltonian(dump, states)
        if coupling is not None:
            shell_of = np.array([state.orbital.shell for state in states])
            between = shell_of[:, None] != shell_of[None, :]
            hamiltonian = hamiltonian * np.where(between, coupling, 1.0)
        with timing.timed(f"fragment {origin}: diagonalise"):
            eigenvalues, eigenvectors = diagonalise(hamiltonian)
    return Fragment(origin=origin, radius=radius, selection=list(selection),
                    shell=shell, states=states, hamiltonian=hamiltonian,
                    eigenvalues=eigenvalues, eigenvectors=eigenvectors)


@dataclass(frozen=True)
class Frontier:
    """The highest states of every origin, Loewdin-orthonormalised together."""

    origins: List[str]
    count: int
    selection: List[str]
    levels: np.ndarray      # fragment levels of the chosen states
    overlap: np.ndarray     # S before the Loewdin rotation, k-averaged
    columns: np.ndarray     # the orthonormal columns U(k), (nk, norb, size)
    hloc: np.ndarray
    nocc: np.ndarray

    @property
    def size(self) -> int:
        return self.hloc.shape[0]


def frontier(cell, dump: PwDump, origins: Sequence[str], radius: float,
             selection: Sequence[str], count: int,
             mu: float, kT: float,
             coupling: Optional[float] = None,
             verbose: bool = True,
             models: Optional[Sequence["Fragment"]] = None) -> Frontier:
    """Lowdin-orthonormalised model of the `count` highest states per origin.

    The states of different origins overlap (they share ligand orbitals), so
    the set is only orthonormal once rotated.  The rotation acts on the
    projectors: U(k) carries the cell phase, A = S^{-1/2} is applied on the
    right, and `operators` does the rest.  No real-space matrix is rotated.

    `coupling` is passed to `fragment`; leave it alone for the octahedron as it
    comes out of the projection.  `models` are fragments already built (in the
    order of `origins`), so a driver that prints them does not build them a
    second time here.
    """
    with timing.stage(f"Loewdin orthonormalisation   ({len(origins)} origins"
                      f" x {count} states)"):
        columns, levels = _frontier_columns(cell, dump, origins, radius,
                                            selection, count, coupling, models)
        combined = np.concatenate(columns, axis=2)   # (nk, norb, origins*count)
        with timing.timed("frontier: overlap S (k-averaged)"):
            overlap_matrix = overlap(combined, dump.weights)
        with timing.timed("frontier: Loewdin, one rotation per k"):
            rotated = orthonormal_per_k(combined, verbose=verbose)
        with timing.timed("frontier: Hloc"):
            hloc = operators.hamiltonian(dump, transform=rotated)
        with timing.timed("frontier: n_occ"):
            nocc = operators.density_matrix(dump, mu, kT, transform=rotated)

    return Frontier(origins=list(origins), count=count,
                    selection=list(selection),
                    levels=np.concatenate(levels),
                    overlap=overlap_matrix,
                    columns=rotated,
                    hloc=hloc, nocc=nocc)


def _frontier_columns(cell, dump, origins, radius, selection, count, coupling,
                      models=None):
    """The Bloch columns of the `count` highest states of every origin."""
    if models is None:
        with timing.timed(f"frontier: {len(origins)} fragment(s)"):
            models = [fragment(cell, dump, origin, radius, selection, coupling)
                      for origin in origins]
    columns, levels = [], []
    for model in models:
        energies, _, chosen = highest_states(model.eigenvalues,
                                             model.eigenvectors, count)
        with timing.timed(f"frontier {model.origin}: Bloch columns"):
            columns.append(bloch_columns(model.states, model.eigenvectors,
                                         chosen, dump.kfrac, dump.norb))
        levels.append(energies)
    return columns, levels

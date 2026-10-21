"""
Where a state lives, in real space.

A state is known through its band amplitudes A[n, b] = <b|n>; the projector
file gives P[b, mu] = <phi_mu|b> for the orbitals of the primitive cell.  Their
product, Fourier transformed, assigns to every orbital of every cell the
coefficient

    C[n, mu, R] = sum_k w_k exp(-2 pi i k.R) sum_b A[n, b] conj(P[b, mu])

so |C|^2 is the weight of the state on that atom image.  Grouping the images by
distance from an atom answers the question a fragment model silently assumes a
priori: is the state inside one octahedron, or does it spill onto neighbours?

Two properties make the fractions trustworthy:

* the report uses only |C|^2, so the sign of the Fourier phase cannot change a
  shell weight -- the wrong sign moves weight between equivalent images of the
  same atom, and
* sum_{mu R} |C|^2 is the share of the state the projector manifold can
  represent.  It is printed next to the fractions, so a state the projectors
  miss shows up as a total below 1 instead of as a wrong distribution.

The same function serves Wannier functions (amplitudes from `_u.mat`) and the
cluster states (amplitudes from `local_model.bloch_columns`), which is what
makes the two comparable.
"""

import itertools
from typing import Sequence

import numpy as np
from pymatgen.core import Structure

from .qe_dump import PwDump

DEFAULT_EDGES = (0.6, 2.5, 3.6, 5.2)


def mesh_cells(dump: PwDump) -> np.ndarray:
    """The lattice translations dual to the k mesh, shape (nk, 3).

    A Fourier transform on the mesh cannot tell apart cells that differ by a
    mesh vector, so the mesh itself is the complete set of atom images.
    Requires the same number of points in each direction.
    """
    side = int(round(dump.nk ** (1.0 / 3.0)))
    if side ** 3 != dump.nk:
        raise ValueError(f"the k mesh has {dump.nk} points, "
                         f"which is not a cube -- no dual cell list")
    grid = np.arange(side)
    return np.array(list(itertools.product(grid, grid, grid)))


def orbital_coefficients(dump: PwDump, amplitudes: np.ndarray) -> np.ndarray:
    """C[n, mu, R] = <n|phi_{mu R}>, shape (n_states, n_orb, n_cells).

    `amplitudes` is <b|n> per k, shape (nk, n_states, nbnd) -- the same array
    `spectral.band_weights` consumes, so a state enters here exactly as it
    enters a projected DOS.

    The conjugation is on the projector, not outside the sum:
    <n|phi> = sum_b <n|b><b|phi> and the library's amplitude is
    A = P U = conj(<b|n>), while <b|phi> = conj(P).  So the sum is
    sum_b A[n, b] conj(P[b, mu]).  Conjugating the whole sum instead picks
    sum_b A P, which is the transpose product: it looks equally innocent and
    produces a state spread evenly over every cell (max ~0.005 instead of
    0.58 for a state built from one octahedron), because the two factors are
    then in opposite gauges.
    """
    if amplitudes.shape[0] != dump.nk or amplitudes.shape[2] != dump.nbnd:
        raise ValueError(f"amplitudes have shape {amplitudes.shape}, "
                         f"expected ({dump.nk}, n_states, {dump.nbnd})")
    band_sum = np.einsum("knb,kbm->knm", amplitudes, dump.proj.conj())
    phase = np.exp(-2j * np.pi * (dump.kfrac @ mesh_cells(dump).T))
    return np.einsum("k,knm,kr->nmr", dump.weights, band_sum, phase)


def orbital_atoms(dump: PwDump) -> np.ndarray:
    """0-based cell site of every .amn row."""
    return np.array([orbital.atom - 1 for orbital in dump.orbitals])


def atom_weights(dump: PwDump, coefficients: np.ndarray) -> np.ndarray:
    """Weight of each state on each atom of the cell, shape (n_states, natoms).

    Summed over the orbitals of the atom and over all its images, so the column
    is "how much of this state belongs to that atom", the quantity that decides
    which atom a state is centred on.
    """
    weight = np.abs(coefficients) ** 2
    atom = orbital_atoms(dump)
    table = np.zeros((weight.shape[0], int(atom.max()) + 1))
    for index, site in enumerate(atom):
        table[:, site] += weight[:, index, :].sum(axis=1)
    return table


def centre_atoms(dump: PwDump, coefficients: np.ndarray) -> np.ndarray:
    """Per state, the 0-based site carrying the largest weight."""
    return atom_weights(dump, coefficients).argmax(axis=1)


def atom_cell_weights(dump: PwDump, coefficients: np.ndarray) -> np.ndarray:
    """(n_states, n_atoms, n_cells) weight of every state on every atom image.

    This is the "one number per (atom, R)" table: `table[state, atom, cell]` is
    how much of that state sits on that atom of that cell, which is exactly the
    question "does the state stay in the neighbouring cells or not".
    """
    weight = np.abs(coefficients) ** 2
    atom = orbital_atoms(dump)
    table = np.zeros((weight.shape[0], int(atom.max()) + 1, weight.shape[2]))
    for index, site in enumerate(atom):
        table[:, site, :] += weight[:, index, :]
    return table


def atom_cell_distances(structure: Structure, dump: PwDump,
                        site: int) -> np.ndarray:
    """(n_atoms, n_cells) distance of every atom image from site `site`.

    The distance companion of `atom_cell_weights`: with both tables an entry
    (atom, cell) says how far that image is and how much of the state sits on
    it, which is the whole "where does this function live" question.
    """
    positions = _folded_positions(structure, dump, site)
    distance = np.linalg.norm(positions, axis=-1)          # (n_orb, n_cells)
    atom = orbital_atoms(dump)
    table = np.full((int(atom.max()) + 1, positions.shape[1]), np.inf)
    for orbital, index in enumerate(atom):
        table[index] = np.minimum(table[index], distance[orbital])
    return table


def _folded_positions(structure: Structure, dump: PwDump, site: int) -> np.ndarray:
    """Cartesian position of every orbital image, folded around `site`.

    The mesh cells run over 0..5, so an image can be six cells away from the
    centre and still be the nearest one: the fractional offset is folded into
    [-0.5, 0.5) first, and only then is the surviving ambiguity of a skewed
    cell resolved by trying the 27 neighbouring images.
    """
    lattice = structure.lattice
    sites = np.array([s.frac_coords for s in structure])
    offset = (sites[orbital_atoms(dump)][:, None, :] + mesh_cells(dump)[None]
              - sites[site])
    offset -= np.round(offset)
    cartesian = lattice.get_cartesian_coords(
        offset.reshape(-1, 3)).reshape(offset.shape)
    shifts = np.array(list(itertools.product((-1, 0, 1), repeat=3))) @ lattice.matrix
    candidates = cartesian[:, :, None, :] + shifts[None, None, :, :]
    nearest = np.linalg.norm(candidates, axis=-1).argmin(axis=-1)
    return np.take_along_axis(candidates, nearest[..., None, None], axis=2)[..., 0, :]


def distance_shells(structure: Structure, dump: PwDump,
                    coefficients: np.ndarray,
                    edges: Sequence[float] = DEFAULT_EDGES):
    """(labels, fractions, totals) of each state in distance shells.

    The shells are measured from the state's own centre atom (the one with the
    most weight), so a Ni1-centred and a Ni2-centred state are described the
    same way and their rows are directly comparable.

    `fractions` sums to 1 over the shells; `totals` is the weight the projector
    manifold could represent, so read the fractions as "of the part we see".
    """
    limits = np.concatenate([[0.0], np.asarray(edges, dtype=float), [np.inf]])
    weight = np.abs(coefficients) ** 2
    centres = centre_atoms(dump, coefficients)

    fractions = np.zeros((weight.shape[0], len(limits) - 1))
    for site in range(len(structure)):
        rows = np.where(centres == site)[0]
        if not len(rows):
            continue
        distance = np.linalg.norm(_folded_positions(structure, dump, site), axis=-1)
        for index in range(len(limits) - 1):
            inside = (distance >= limits[index]) & (distance < limits[index + 1])
            fractions[rows, index] = weight[rows][:, inside].sum(axis=1)

    totals = fractions.sum(axis=1)
    return _labels(limits), fractions / totals[:, None], totals


def spread(structure: Structure, dump: PwDump,
           coefficients: np.ndarray) -> np.ndarray:
    """<r^2> - |<r>|^2 of each state in Angstrom^2, the wannier90 spread.

    Measured from the state's own centre atom, which removes the cell of the
    centre but not the extent, so the number is directly comparable with the
    spreads wannier90 prints in `_u.mat`-based runs (`nio.wout`).
    """
    weight = np.abs(coefficients) ** 2
    centres = centre_atoms(dump, coefficients)
    values = np.zeros(weight.shape[0])
    for site in range(len(structure)):
        rows = np.where(centres == site)[0]
        if not len(rows):
            continue
        position = _folded_positions(structure, dump, site).reshape(-1, 3)
        part = weight[rows].reshape(len(rows), -1)
        norm = part.sum(axis=1)
        mean = (part @ position) / norm[:, None]
        square = (part @ np.sum(position ** 2, axis=1)) / norm
        values[rows] = square - np.sum(mean ** 2, axis=1)
    return values


def _labels(limits: np.ndarray) -> list:
    """('own atom', '< 2.5 A', ..., '> 5.2 A') for the shells of `limits`."""
    labels = ["own atom"]
    for index in range(1, len(limits) - 2):
        labels.append(f"< {limits[index + 1]:.1f} A")
    labels.append(f"> {limits[-2]:.1f} A")
    return labels

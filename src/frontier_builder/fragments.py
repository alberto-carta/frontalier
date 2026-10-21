"""
Coordination shells: which atoms sit around a given atom, within a radius.

The user names the atoms to centre on (e.g. "Ni1", "Ni2") and a radius;
pymatgen's lattice search does the periodic work and we name and group the
result.

Why not `Structure.get_neighbors`
---------------------------------
That goes through `pymatgen.optimization.neighbors.find_points_in_spheres`,
whose Cython implementation **misses a genuine neighbour in this cell**.
For Ni2 at (-0.5, 1.5, -0.5) it returns 5 O at 2.0894 A while there are 6:

    brute force               6 : O3(-2,2,0) O3(-1,1,0) O3(-1,2,-1)
                                  O4(0,1,0) O4(0,2,-1) O4(1,1,-1)
    get_neighbors             5 : the O4(1,1,-1) image is dropped
    get_points_in_sphere_py   6 : finds all of them

So we call `Lattice.get_points_in_sphere_py`, the pure-Python reference
implementation of the same search, and skip the centre itself (the only entry
at zero distance).  It is slower, but the cells of interest have a handful of
atoms and the result is verifiable.

Why `image` matters
-------------------
A neighbour is returned as a cell site plus a lattice translation.  With the
centre at tau_c in cell 0 and the neighbour at tau_n in cell R, the actual
separation is

    (tau_n + R - tau_c) @ lattice

so `image` is exactly the R that H_{mu nu}(R) = <mu 0|H|nu R> needs, and
`site` is the atom whose orbitals (its .amn rows) belong to that R.

Atom names follow the .amn label convention: element + 1-based cell index, so
atom 1 of a Ni2O2 cell is "Ni1" and atom 3 is "O3", matching
`Orbital.atom_name`.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from pymatgen.core import Structure

ZERO_DISTANCE = 1e-8


@dataclass(frozen=True)
class Neighbour:
    """One atom of a coordination shell."""

    site: int                     # 1-based index in the cell (.amn atom index)
    name: str                     # "O3"
    species: str                  # "O"
    image: Tuple[int, int, int]   # lattice translation relative to the centre cell
    distance: float               # Angstrom


def atom_names(structure: Structure) -> List[str]:
    """['Ni1', 'Ni2', 'O3', 'O4'] -- element + 1-based cell index."""
    return [f"{site.specie}{index}" for index, site in enumerate(structure, start=1)]


def resolve_atoms(structure: Structure, names: List[str]) -> List[int]:
    """Turn atom names into 0-based site indices."""
    table = {name: index for index, name in enumerate(atom_names(structure))}
    missing = [name for name in names if name not in table]
    if missing:
        raise ValueError(f"no atom(s) {missing} in the cell; "
                         f"available: {list(table)}")
    return [table[name] for name in names]


def coordination_shell(structure: Structure, name: str,
                       radius: float) -> List[Neighbour]:
    """Neighbours of the named atom within `radius` Angstrom, nearest first."""
    index = resolve_atoms(structure, [name])[0]
    centre_cartesian = structure.lattice.get_cartesian_coords(
        structure[index].frac_coords)
    found = structure.lattice.get_points_in_sphere_py(
        structure.frac_coords, centre_cartesian, radius)

    names = atom_names(structure)
    shell = []
    for _, distance, site_index, image in found:
        if distance < ZERO_DISTANCE:                  # the centre itself
            continue
        shell.append(Neighbour(
            site=site_index + 1,
            name=names[site_index],
            species=str(structure[site_index].specie),
            image=tuple(int(i) for i in image),
            distance=float(distance),
        ))
    return sorted(shell, key=lambda neighbour: (neighbour.distance, neighbour.name))


def coordination(structure: Structure, names: List[str],
                 radius: float) -> Dict[str, List[Neighbour]]:
    """{atom name: shell} for every requested centre."""
    return {name: coordination_shell(structure, name, radius) for name in names}


def composition(shell: List[Neighbour]) -> Dict[str, int]:
    """Number of neighbours per species, e.g. {'O': 6}."""
    counts: Dict[str, int] = {}
    for neighbour in shell:
        counts[neighbour.species] = counts.get(neighbour.species, 0) + 1
    return counts


def distances(shell: List[Neighbour]) -> Tuple[float, float]:
    """(shortest, longest) neighbour distance in the shell."""
    return (min(n.distance for n in shell), max(n.distance for n in shell))


def image_array(neighbour: Neighbour) -> np.ndarray:
    """`image` as an integer numpy array, ready for the k -> R phase."""
    return np.asarray(neighbour.image, dtype=int)

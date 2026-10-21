"""
Which row of a projector file is which atomic orbital.

A `projwfc.x` `.amn` file starts with a comment block that labels every
projection, e.g.

    #        5 : atom    1 (Ni1) n= 3 l=2 m= 1

`read_orbitals` turns that block into a list of `Orbital`, one per row of the
projector array, and `select` picks the rows belonging to a shell such as
"Ni 3d".

Note on `n`
-----------
`projwfc.x` writes the *index of the atomic wavefunction inside the
pseudopotential* as `n`, not necessarily the physical principal quantum
number.  For the Ni/O pseudopotentials used here, the O s wavefunction comes
out as `n=1` although it is physically the O 2s.  `Orbital.n` reports the file
value unchanged; do not read it as chemistry.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# Real spherical harmonics in the order used by projwfc.x (m = 1, 2, ...).
LM_NAMES = {
    0: ["s"],
    1: ["pz", "px", "py"],
    2: ["dz2", "dxz", "dyz", "dx2-y2", "dxy"],
    3: ["fz3", "fxz2", "fyz2", "fzx2-zy2", "fxyz", "fx3-3xy2", "f3yx2-y3"],
}
SHELL_LETTERS = "spdfg"

# "#        5 : atom    1 (Ni1) n= 3 l=2 m= 1"
_HEADER_LINE = re.compile(
    r"#\s*(\d+)\s*:\s*atom\s+(\d+)\s*\(\s*(\w+)\s*\)"
    r"\s*n=\s*(\d+)\s*l=\s*(\d+)\s*m=\s*(-?\d+)"
)


@dataclass(frozen=True)
class Orbital:
    """One row of the projector array."""

    index: int          # 0-based row
    atom: int           # 1-based, as in the pw input
    label: str          # atom label as written by projwfc: "Ni1", "O"
    n: int
    l: int
    m: int              # 1-based, QE ordering

    @property
    def species(self) -> str:
        """'Ni1' -> 'Ni'."""
        return re.sub(r"\d+$", "", self.label)

    @property
    def atom_name(self) -> str:
        """Unique per atom: 'Ni1' stays, 'O' becomes 'O3' / 'O4'."""
        return self.label if self.label[-1].isdigit() else f"{self.label}{self.atom}"

    @property
    def name(self) -> str:
        """'Ni1 3dz2'."""
        return f"{self.atom_name} {self.n}{lm_name(self.l, self.m)}"

    @property
    def shell(self) -> str:
        """'Ni 3d'."""
        return f"{self.species} {self.n}{SHELL_LETTERS[self.l]}"

    @property
    def atom_shell(self) -> str:
        """'Ni1 3d' -- the shell of one specific atom."""
        return f"{self.atom_name} {self.n}{SHELL_LETTERS[self.l]}"


def lm_name(l: int, m: int) -> str:
    """'dz2' for (2, 1); falls back to 'l2m1' for unknown shells."""
    names = LM_NAMES.get(l)
    if names is None or not 1 <= m <= len(names):
        return f"l{l}m{m}"
    return names[m - 1]


def read_orbitals(amn_path: str) -> List[Orbital]:
    """Orbital table from the comment header of a `.amn` file."""
    found = {}
    with open(amn_path) as fh:
        for line in fh:
            if not line.lstrip().startswith("#"):
                continue
            match = _HEADER_LINE.match(line.strip())
            if match:
                index, atom, label, n, l, m = match.groups()
                found[int(index)] = Orbital(
                    index=int(index) - 1, atom=int(atom), label=label,
                    n=int(n), l=int(l), m=int(m),
                )
    orbitals = [found[i] for i in sorted(found)]
    if not orbitals:
        raise ValueError(f"{amn_path}: no orbital header lines found")
    if [o.index for o in orbitals] != list(range(len(orbitals))):
        raise ValueError(f"{amn_path}: orbital labels are not contiguous from 1")
    return orbitals


def parse_shell(spec: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """'Ni 3d' -> ('Ni', 3, 2);  'Ni' -> ('Ni', None, None);  '3d' -> (None, 3, 2)."""
    match = re.fullmatch(r"\s*([A-Za-z]+)?\s*(\d+)?\s*([spdfg])?\s*", spec)
    if not match or not spec.strip():
        raise ValueError(f"cannot read orbital selection {spec!r} (use 'Ni 3d')")
    species, n, letter = match.groups()
    l = SHELL_LETTERS.index(letter) if letter else None
    return species, (int(n) if n else None), l


def matches(orbital: Orbital, spec: Tuple[Optional[str], Optional[int],
                                          Optional[int]]) -> bool:
    """Does the orbital satisfy an (species, n, l) pattern? None = any."""
    species, n, l = spec
    return ((species is None or species.lower() == orbital.species.lower())
            and (n is None or n == orbital.n)
            and (l is None or l == orbital.l))


def select(orbitals: Sequence[Orbital], specs: Sequence[str]) -> List[int]:
    """Indices (0-based) of the orbitals matching any of the shell specs."""
    patterns = [parse_shell(spec) for spec in specs]
    chosen = [o.index for o in orbitals
              if any(matches(o, pattern) for pattern in patterns)]
    if not chosen:
        raise ValueError(f"orbital selection {list(specs)} matched nothing")
    return chosen


def group_by_atom_shell(orbitals: Sequence[Orbital],
                        indices: Sequence[int]):
    """[(atom_name, shell, [indices])] in file order, one entry per atom-shell."""
    groups = {}
    for index in indices:
        orbital = orbitals[index]
        key = (orbital.atom_name, orbital.shell)
        groups.setdefault(key, []).append(index)
    return [(atom, shell, members) for (atom, shell), members in groups.items()]

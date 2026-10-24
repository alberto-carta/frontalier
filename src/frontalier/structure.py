"""
The cell, read from a Quantum ESPRESSO pw input file with pymatgen.

We do not parse the input ourselves.  `pymatgen.io.pwscf.PWInput` already
understands it, but only after three textual normalisations that QE accepts
and pymatgen does not:

  1. `ATOMIC_POSITIONS crystal` / `CELL_PARAMETERS (angstrom)` /
     `K_POINTS automatic` must carry braces: `{crystal}`, `{angstrom}`,
     `{automatic}` — otherwise pymatgen raises IndexError looking for "{";
  2. numbers written as `-.5` must become `-0.5`, otherwise pymatgen's atom
     regex (which requires a digit before the decimal point) silently skips
     the atom;
  3. `CELL_PARAMETERS` must come *before* `ATOMIC_POSITIONS`, because pymatgen
     decides "cartesian or crystal" from the last structural keyword it saw.

One block is replaced rather than repaired: `K_POINTS`.  An explicit list
(`K_POINTS crystal` followed by 216 rows) makes pymatgen raise
`UnboundLocalError` from a branch that never assigns its local grid, and this
reader only wants the cell, so the block becomes a one-point automatic grid.
The k points of the dump come from `<prefix>.kpt`, not from here.

`read_structure` applies those fixes to the text and hands it to
pymatgen.  The atom order of the resulting Structure is the atom order of the
pw input, which is also the `atom` index of the .amn header.
"""

import re

from pymatgen.core import Structure
from pymatgen.io.pwscf import PWInput

from . import timing


def _normalise(text: str) -> str:
    """Make a QE input acceptable to pymatgen (see the module docstring)."""
    text = _resolve_alat(text)
    text = _braced_modes(text)
    text = re.sub(r"(?<![\d.])-?\.(\d)",
                  lambda m: m.group(0).replace(".", "0."), text)
    return _cell_before_positions(_replace_kpoints(text))


BOHR_TO_ANGSTROM = 0.529177210903

#  CELL_PARAMETERS (alat= 7.26120000)  +  the three rows that follow it
_ALAT_CARD = re.compile(
    r"^([ \t]*)CELL_PARAMETERS[ \t]*\([ \t]*alat[ \t]*=[ \t]*"
    r"([-+]?[\d.]+(?:[eEdD][-+]?\d+)?)[ \t]*\)[^\n]*\n"
    r"([^\n]*)\n([^\n]*)\n([^\n]*)\n",
    re.MULTILINE | re.IGNORECASE)


def _qe_float(text: str) -> float:
    """A QE number, which may carry a Fortran `d` exponent."""
    return float(text.replace("d", "e").replace("D", "E"))


def _resolve_alat(text: str) -> str:
    """Rewrite `CELL_PARAMETERS (alat=X)` as explicit Angstrom rows.

    QE reads the three numbers in units of `X` Bohr; pymatgen ignores the
    factor and would build a cell that is `X` bhor too small -- the SrVO3 input
    came out with a = 1.0044 A instead of 3.8596 A, quietly.  The conversion is
    done here and the card becomes `{angstrom}`, which also stops
    `_braced_modes` from mangling the `=` sign.
    """
    def repl(match):
        indent, scale = match.group(1), _qe_float(match.group(2))
        scale *= BOHR_TO_ANGSTROM
        rows = []
        for group in (3, 4, 5):
            values = [_qe_float(value) * scale for value in match.group(group).split()]
            rows.append(f"{indent}  {values[0]:18.10f}{values[1]:18.10f}"
                        f"{values[2]:18.10f}")
        return f"{indent}CELL_PARAMETERS {{angstrom}}\n" + "\n".join(rows) + "\n"

    return _ALAT_CARD.sub(repl, text)


#  the cards whose mode pymatgen only accepts in braces
MODE_CARDS = ("ATOMIC_POSITIONS", "CELL_PARAMETERS", "K_POINTS")


def _braced_modes(text: str) -> str:
    """`CARD mode`, `CARD (mode)` and `CARD {mode}` all become `CARD {mode}`.

    QE accepts the three spellings, pymatgen only the braced one: a bare
    `CELL_PARAMETERS angstrom` is what made it raise IndexError on the FeO
    input.  Only letters and underscores are taken as a mode, so a card whose
    mode is omitted altogether (QE then means alat) is left alone.
    """
    for card in MODE_CARDS:
        text = re.sub(
            rf"^([ \t]*{card})[ \t]*[\(\{{]?[ \t]*([A-Za-z_]+)[ \t]*[\)\}}]?",
            r"\1 {\2}", text, flags=re.MULTILINE)
    return text


def _replace_kpoints(text: str) -> str:
    """Swap the K_POINTS block for a one-point automatic grid.

    The block is not structure and pymatgen cannot read an explicit list, but
    it insists on seeing a K_POINTS card, so a trivial one goes back in.
    """
    kept, skipping = [], False
    for line in text.splitlines():
        if line.strip().upper().startswith("K_POINTS"):
            skipping = True
            kept.append("K_POINTS {automatic}\n1 1 1 0 0 0")
            continue
        if skipping:
            stripped = line.strip()
            if stripped and not re.match(r"^[-+0-9.\s]+$", stripped):
                skipping = False        # a new namelist or card starts here
            else:
                continue                # still inside the blocked list
        kept.append(line)
    return "\n".join(kept)


def _cell_before_positions(text: str) -> str:
    """Move the CELL_PARAMETERS block above ATOMIC_POSITIONS, if it is below."""
    lines = text.splitlines()
    try:
        i_cell = next(i for i, l in enumerate(lines) if l.startswith("CELL_PARAMETERS"))
        i_pos = next(i for i, l in enumerate(lines) if l.startswith("ATOMIC_POSITIONS"))
    except StopIteration:
        return text                       # nothing to reorder
    if i_cell < i_pos:
        return text
    block = lines[i_cell:i_cell + 4]                 # keyword + 3 lattice rows
    rest = lines[:i_cell] + lines[i_cell + 4:]
    return "\n".join(rest[:i_pos] + block + rest[i_pos:])


def read_structure(pw_input_path: str) -> Structure:
    """pymatgen Structure of the cell described by a pw input file."""
    with timing.stage(f"reading the structure   ({pw_input_path})"):
        with open(pw_input_path) as fh:
            return PWInput.from_string(_normalise(fh.read())).structure

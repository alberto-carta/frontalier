"""
Printing helpers.  The only module in the library that writes to stdout.

Every block is printed the same way:

    title   (n orbitals)
      #5 Ni1 3dz2   0.9834   0.0000  ...
      #6 Ni1 3dxz   0.0000   0.9834  ...
      trace                           3.9918
      eigval        0.5205  0.5205  0.9834  ...

The leading `#5` is the row of the full .amn array, so a block can always be
traced back to the file.  The eigenvalues are always printed because the
diagonal of a block depends on the basis (degenerate multiplets, Löwdin
mixing) while its spectrum does not.  The trace is opt-in: it is the electron
count for an occupation matrix, and meaningless for a Hamiltonian block, which
carries an arbitrary reference energy and a gauge.
"""

import numpy as np


def _text(values):
    return " ".join(f"{v:9.6f}" for v in values)


def structure_report(structure, source: str) -> None:
    """Lattice, cell parameters and the per-atom table."""
    lat = structure.lattice
    print(f"\nstructure   ({source}, read with pymatgen)")
    print(f"  formula          {structure.formula}    natoms {len(structure)}")
    print(f"  a, b, c          {lat.a:10.6f} {lat.b:10.6f} {lat.c:10.6f}  Angstrom")
    print(f"  alpha, beta, gamma {lat.alpha:8.3f} {lat.beta:8.3f} {lat.gamma:8.3f}"
          f"  degrees")
    print("  lattice vectors (rows, Angstrom)")
    for vector in lat.matrix:
        print("    " + _text(vector))
    print("   #  species  fractional coordinates          cartesian (Angstrom)")
    for i, site in enumerate(structure, start=1):
        print(f"  {i:2d}  {str(site.specie):<7s} {_text(site.frac_coords)}"
              f"   {_text(site.coords)}")


def matrix(title: str, values, per_strip: int = 8, width: int = 8,
           decimals: int = 3) -> None:
    """A square matrix printed in vertical strips of `per_strip` columns.

    Rows are labelled by their 1-based position, so a wide fragment
    Hamiltonian stays readable; the meaning of each row is printed separately
    as the fragment basis.
    """
    size = values.shape[0]
    row_width = len(str(size))
    indent = " " * (4 + row_width)

    for start in range(0, size, per_strip):
        stop = min(start + per_strip, size)
        print(f"  {title}   columns {start + 1}-{stop} of {size}")
        print(indent + "".join(f"{column + 1:>{width}d}"
                               for column in range(start, stop)))
        for row in range(size):
            entries = "".join(f"{values[row, column].real:{width}.{decimals}f}"
                              for column in range(start, stop))
            print(f"    {row + 1:<{row_width}d}{entries}")
        print()


def block(title: str, matrix, labels, width: int = 9, decimals: int = 4,
          trace: bool = False) -> None:
    """One square block of a matrix, rows and columns named by `labels`.

    The trace is off by default and should only be asked for when it means
    something.  It does for an occupation matrix (it is the electron count); it
    does not for a Hamiltonian block, which is offset by an arbitrary reference
    energy and written in a particular gauge.
    """
    size = len(labels)
    label_width = max(len(label) for label in labels)
    print(f"  {title}   ({size} orbitals)")

    for row, label in enumerate(labels):
        values = "".join(f"{matrix[row, col].real:{width}.{decimals}f}"
                         for col in range(size))
        print(f"    {label:<{label_width}s}{values}")

    if trace:
        pad = " " * ((size - 1) * width)
        print(f"    {'trace':<{label_width}s}{pad}"
              f"{np.trace(matrix).real:{width}.{decimals}f}")
    eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.conj().T)).real
    print(f"    {'eigval':<{label_width}s}"
          + "".join(f"{v:{width}.{decimals}f}" for v in eigenvalues))

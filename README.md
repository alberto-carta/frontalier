# frontier_builder

Hello world, I am wannierizing!

Small tools to read a Quantum ESPRESSO / wannier90 dump and build local frontier
models from it: projectors out of a `.amn` or a `_u.mat`, $n(R)$ and $H(R)$, the
coordination fragments and their sub-diagonalisation, the occupation matrices,
and DOS / PDOS / COHP.  One module, one job; the numerics never print.

## Install

```bash
conda activate pymatgen_env
pip install -e .                 # or: pip install -e ".[test]"
```

Python >= 3.8, `numpy`, `pymatgen`, `matplotlib` (see `requirements.txt`).

## Use

```python
from frontier_builder import local_model, qe_dump, structure
```

The scripts that drive the library live with the calculation they analyse, e.g.
`NiO/run_nloc.py` (a projwfc dump), `NiO/run_nloc_wannier.py` (a wannier90 run)
and `NiO/check_wannier.py` (two runs, compared).

## Test

```bash
pytest
```


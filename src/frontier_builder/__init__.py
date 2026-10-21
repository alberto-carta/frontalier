"""
frontier_builder
================

Small tools to read a Quantum ESPRESSO / wannier90 dump and build local
frontier models from it.

Design rules
------------
* One module, one job.
* Data in, data out: functions take arrays and return arrays.
  Nothing prints inside the numerics.
* 0-based indices everywhere inside the library; 1-based only at the file
  boundary (the .amn header and the .kpt/.eig columns).
* Conventions are stated once, in `qe_dump`, and nowhere else.

Modules
-------
orbitals    which row of a projector file is which atomic orbital
qe_dump     readers for <prefix>.kpt / .eig / .amn
operators   the k -> R transform: occupation matrix and Hamiltonian
structure   the cell, read from a pw input file with pymatgen
fragments   coordination shells: which atoms surround a chosen atom
local_model the fragment Hamiltonian and its sub-diagonalisation
wannier     readers for <prefix>.eig / _u.mat, and the same PwDump from them
spectral    DOS, projected DOS and COHP out of energies and projectors
spatial     where a state lives: weights per atom image and distance shell
analysis    the pipeline: a dump in, the standard report out
plots       the figures; the only module that draws
report      the only module that prints
"""

__all__ = ["orbitals", "qe_dump", "operators", "structure", "fragments",
           "local_model", "wannier", "spectral", "spatial", "analysis",
           "plots", "report"]

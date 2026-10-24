"""
End-to-end checks against the reference calculations.

Each case reads the settings out of the driver that sits next to its data, so
the test uses the same selection, radius, Fermi level and temperature as the
calculation it checks.  Nothing is recomputed from scratch: the numbers are the
ones the drivers printed when the construction was validated.

The data lives outside the repository.  Point FRONTALIER_DATA at the folder
that holds NiO/, FeO/ and KCuF3/, or the cases are skipped (a clone without the
calculations still runs the smoke test).

    FRONTALIER_DATA=/path/to/Ortho_projectors pytest

Tolerances are loose enough to survive a re-run of the nscf and tight enough to
catch a change in the construction.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from frontalier import fragments, local_model, qe_dump, spectral, structure

DATA = Path(os.environ.get("FRONTALIER_DATA",
                           Path(__file__).resolve().parents[2]))

#  electrons in the model per spin channel, and the two frontier levels (eV)
CASES = {
    "NiO": {"electrons": 7.8576, "t2g": 13.542, "eg": 14.374},
    "FeO": {"electrons": 2.9948, "t2g": 14.008, "eg": 15.064},
    "KCuF3": {"electrons": 4.4311, "t2g": 4.670, "eg": 5.174},
}

_CACHE = {}


def _settings(compound):
    """The constants of <compound>/run_nloc.py, without running its main()."""
    path = DATA / compound / "run_nloc.py"
    if not path.exists():
        pytest.skip(f"{path} not found -- set FRONTALIER_DATA")
    spec = importlib.util.spec_from_file_location(f"driver_{compound}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return path.parent, module


def _build(compound):
    """The dump and the frontier model of one compound, built once."""
    directory, s = _settings(compound)
    dump = qe_dump.load(str(directory / s.PREFIX[2:]),
                        spin_degeneracy=s.SPIN_DEGENERACY)
    cell = structure.read_structure(str(directory / s.PW_INPUT[2:]))
    mu = s.FERMI_LEVEL
    if mu is None:
        mu = spectral.chemical_potential(
            dump, s.N_ELECTRONS / s.SPIN_DEGENERACY, s.KT, verbose=False)
    origins = fragments.expand_atoms(cell, s.FRAGMENT_ORIGINS)
    model = local_model.frontier(cell, dump, origins, s.FRAGMENT_RADIUS,
                                 s.FRAGMENT_ORBITALS, s.FRONTIER_COUNT, mu,
                                 s.KT, verbose=False)
    return dump, model


def _case(compound):
    if compound not in _CACHE:
        _CACHE[compound] = _build(compound)
    return _CACHE[compound]


@pytest.mark.parametrize("compound", sorted(CASES))
def test_frontier_model(compound):
    """Occupations and levels of the frontier model, per spin channel."""
    _, model = _case(compound)
    expected = CASES[compound]

    assert np.isclose(np.trace(model.nocc).real, expected["electrons"],
                      atol=0.02), f"{compound}: electron count changed"

    levels = np.sort(np.diag(model.hloc).real.reshape(-1, model.count), axis=1)
    assert np.allclose(levels[:, :3], expected["t2g"], atol=0.05), \
        f"{compound}: t2g levels moved"
    assert np.allclose(levels[:, 3:], expected["eg"], atol=0.06), \
        f"{compound}: e_g levels moved"


@pytest.mark.parametrize("compound", sorted(CASES))
def test_projectors_are_complete(compound):
    """sum_b |<phi_mu|psi_b>|^2 must be 1: the dump has to span its orbitals.

    A nscf with fewer bands than projectors cannot do that, and every weight,
    H(R=0) and occupation built from it silently underestimates.
    """
    dump, _ = _case(compound)
    assert dump.nbnd >= dump.norb, (
        f"{compound}: {dump.norb} projectors in {dump.nbnd} bands")
    completeness = np.einsum("k,kbo->o", dump.weights, np.abs(dump.proj) ** 2)
    assert completeness.min() > 0.95, (
        f"{compound}: projector {completeness.argmin()} carries only "
        f"{completeness.min():.2f} of its weight")

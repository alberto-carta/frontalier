"""
The analysis steps shared by the driver scripts.

Everything here either builds a model (via `local_model`), prints one (via
`report`) or draws one (via `plots`); the numerics live in `operators`,
`local_model` and `spectral`.  A driver script only chooses files, a Fermi
level and an orbital selection, then calls the functions below in order.
"""

import re
from dataclasses import dataclass
from typing import List

import numpy as np

from . import (fragments, local_model, orbitals, operators, plots, qe_dump,
               report, spectral, timing, wannier)


# --------------------------------------------------------------------------
#  the dump
# --------------------------------------------------------------------------
def dump_summary(dump, prefix=""):
    """Size and normalisation of a projector dump."""
    print("\ndump")
    if prefix:
        print(f"  prefix            {prefix}")
    print(f"  k-points          {dump.nk}")
    print(f"  bands             {dump.nbnd}  (of {dump.nbands_total})")
    print(f"  orbitals          {dump.norb}")
    if dump.nbnd < dump.norb:
        print(f"  WARNING           {dump.norb} projectors cannot be complete in "
              f"{dump.nbnd} bands: every weight, H(R=0) and n_occ built from "
              f"them is an underestimate. Raise nbnd (a few hundred) in the "
              f"nscf and re-run projwfc.")
    print(f"  sum_k w_k         {dump.weights.sum():.6f}  (per spin channel)")
    print(f"  energy range      {dump.energies.min():.3f} .. "
          f"{dump.energies.max():.3f} eV")


def atom_order_check(cell, dump):
    """The Structure and the projector dump must list atoms in the same order."""
    seen = {}
    for orbital in dump.orbitals:
        seen.setdefault(orbital.atom, orbital.species)
    from_cell = [str(site.specie) for site in cell]
    from_dump = [seen[atom] for atom in sorted(seen)]
    verdict = "yes" if from_cell == from_dump else "NO -- check the files!"
    print(f"\n  atom order  structure {from_cell}")
    print(f"              dump      {from_dump}    match: {verdict}")


def hr_dat_check(dump, hr_file):
    """Rebuild every H(R) from (eigenvalues, projectors) and compare with a
    written _hr.dat.

    The real-space Hamiltonian is a *derived* quantity here: the analysis only
    ever reads eigenvalues and projectors.  This check proves that the derived
    H(R) is the one projwfc / wannier90 wrote.

    Only the R vectors inside the k-mesh box are compared.  On the box boundary
    the two writers disagree on the degeneracy factor (the projwfc file divides
    those entries by ndegen although its header says 1, the wannier90 file does
    not divide at all), so those entries are not comparable without knowing the
    writer.  Everything inside agrees to the print precision of the file.
    """
    with timing.stage(f"H(R) check against {hr_file}"):
        _hr_dat_check(dump, hr_file)


def _hr_dat_check(dump, hr_file):
    """The body of `hr_dat_check`, so the banner stays out of the way."""
    try:
        nw, reference, _ = wannier.read_hr_dat(hr_file)
        shape = qe_dump.mesh_shape(dump)
    except (OSError, ValueError) as error:
        print(f"\n  {hr_file}: not checked ({error})")
        return
    if nw != dump.norb:
        print(f"\n  {hr_file}: has {nw} orbitals, the dump has {dump.norb}; "
              f"skipping the H(R) check")
        return

    half_box = [n // 2 for n in shape]
    interior = [R for R in reference
                if all(abs(R[axis]) < half_box[axis] for axis in range(3))]
    if not interior:
        print(f"\n  {hr_file}: no R vector inside the k-mesh box")
        return

    blocks = np.einsum("kb,kbm,kbn->kmn", dump.energies, dump.proj,
                       dump.proj.conj())
    worst = 0.0
    for R in interior:
        phase = np.exp(-2j * np.pi * (dump.kfrac @ np.asarray(R, dtype=float)))
        rebuilt = np.einsum("k,kmn->mn", dump.weights * phase, blocks)
        worst = max(worst, float(np.abs(rebuilt - reference[R]).max()))
    print(f"\n  H(R) rebuilt from (eig, projectors) vs {hr_file}:"
          f"   max |dH| = {worst:.2e} eV"
          f"   over {len(interior)} of {len(reference)} R vectors"
          f" (inside the k-mesh box)")


def orbital_table(dump):
    """One line per row of the projector array."""
    print("\n  rows of the projector array")
    print(f"  {'#':>3s}  {'atom':<5s} {'name':<12s} {'shell':<8s} {'l':>2s} {'m':>2s}")
    for orbital in dump.orbitals:
        print(f"  {orbital.index + 1:3d}  {orbital.atom_name:<5s}"
              f" {orbital.name:<12s} {orbital.shell:<8s}"
              f" {orbital.l:2d} {orbital.m:2d}")


def selection_table(dump, selection):
    """Which rows a shell selection picks, one line per atom-shell."""
    chosen = orbitals.select(dump.orbitals, selection)
    print(f"\nselection {selection} -> {len(chosen)} orbitals of {dump.norb}")
    for spec in selection:
        pattern = orbitals.parse_shell(spec)
        if pattern[1] is None or any(orbitals.matches(o, pattern)
                                     for o in dump.orbitals):
            continue
        shell = ("the shell asked for" if pattern[2] is None else
                 f"l={orbitals.SHELL_LETTERS[pattern[2]]}")
        print(f"  note  nothing carries n={pattern[1]} with {shell} here -- the "
              f".amn `n` counts wavefunctions inside the pseudopotential, not "
              f"shells -- so {spec!r} was matched on species and l only")
    for atom, shell, members in orbitals.group_by_atom_shell(dump.orbitals, chosen):
        first, last = members[0] + 1, members[-1] + 1
        span = f"{first}" if first == last else f"{first}-{last}"
        names = " ".join(orbitals.lm_name(dump.orbitals[i].l, dump.orbitals[i].m)
                         for i in members)
        print(f"  {atom:<5s} {shell:<7s} rows {span:<7s} {names}")
    return chosen


# --------------------------------------------------------------------------
#  the selected blocks of n(R=0) and H(R=0)
# --------------------------------------------------------------------------
def block_report(title, matrix, dump, members):
    """One atom-shell block, rows labelled with the full array row index."""
    labels = [f"#{i + 1} {dump.orbitals[i].name}" for i in members]
    report.block(f"{dump.orbitals[members[0]].atom_name} "
                 f"{dump.orbitals[members[0]].shell}",
                 matrix[np.ix_(members, members)], labels)


def matrices(dump, selection, mu, kT):
    """n(R=0) and H(R=0) of a shell selection, one block per atom-shell."""
    chosen = orbitals.select(dump.orbitals, selection)
    with timing.stage(f"n(R=0) and H(R=0) of the selection   "
                      f"({dump.norb} orbitals, mu = {mu:.4f} eV)"):
        occupation = operators.density_matrix(dump, mu, kT)
        hamiltonian = operators.hamiltonian(dump)

    for title, matrix in ((f"n(R=0)   mu = {mu} eV, kT = {kT} eV", occupation),
                          ("H(R=0)   [eV]", hamiltonian)):
        print(f"\n{title}")
        for _, _, members in orbitals.group_by_atom_shell(dump.orbitals, chosen):
            block_report(title, matrix, dump, members)

    print(f"\n  traces over the full {dump.norb}-orbital array:"
          f"   n(R=0) {np.trace(occupation).real:.6f}"
          f"   H(R=0) {np.trace(hamiltonian).real:.6f} eV")


# --------------------------------------------------------------------------
#  a fragment
# --------------------------------------------------------------------------
def fragment_basis(model):
    """The fragment states with their local index, used by the tables."""
    print(f"\n  fragment basis   ({model.size} states)")
    for number, state in enumerate(model.states, start=1):
        print(f"    {number:3d}   {state.label}")


def eigenstate_table(model):
    """Eigenvalues with the mixing in terms of the fragment shells."""
    shells, per_shell = local_model.shell_mixing(model.states, model.eigenvectors)
    print(f"    {'#':>3s} {'E (eV)':>10s}"
          + "".join(f"{shell:>9s}" for shell in shells))
    for column in range(len(model.eigenvalues)):
        print(f"    {column + 1:3d} {model.eigenvalues[column]:10.4f}"
              + "".join(f"{per_shell[s, column]:9.3f}"
                        for s in range(len(shells))))


def fragment_origins(cell, origins):
    """The origin labels of the fragments, one per atom.

    A species name that names several atoms -- "Ti" in a cell with four Ti --
    means all of them, and the expansion is printed, so the report cannot show
    fewer fragments than the driver asked for.
    """
    expanded = fragments.expand_atoms(cell, origins)
    if expanded != list(origins):
        print(f"\n  fragment origins {list(origins)} stand for {expanded}")
    return expanded


def fragment_report(model, show_hamiltonian=True):
    """Shell, basis, Hamiltonian and sub-diagonalisation of one fragment."""
    counts = fragments.composition(model.shell)
    summary = "  ".join(f"{count} x {species}" for species, count in counts.items())
    print(f"\n{'=' * 72}")
    print(f"fragment origin {model.origin}   {len(model.shell)} neighbours"
          f" ({summary})   radius {model.radius:.3f} A")
    print(f"  sub-diagonalising {model.selection}")
    for neighbour in model.shell:
        print(f"    {neighbour.name:<5s} site {neighbour.site}  "
              f"R{neighbour.image}  {neighbour.distance:.4f} A")

    fragment_basis(model)

    if show_hamiltonian:
        print(f"\n  fragment Hamiltonian, before   ({model.size} states, eV)")
        report.matrix("H_fragment", model.hamiltonian, per_strip=8, decimals=3)

    print(f"\n  after sub-diagonalisation   ({model.size} eigenstates, eV)")
    eigenstate_table(model)

    residual = np.linalg.norm(model.hamiltonian @ model.eigenvectors
                              - model.eigenvectors * model.eigenvalues)
    norm_error = np.abs((np.abs(model.eigenvectors) ** 2).sum(axis=0) - 1.0).max()
    print(f"\n  checks   max|H - H^dag| = "
          f"{np.abs(model.hamiltonian - model.hamiltonian.conj().T).max():.1e}"
          f"   ||H V - V E|| = {residual:.1e}"
          f"   max|sum_mu |c|^2 - 1| = {norm_error:.1e}")


# --------------------------------------------------------------------------
#  the frontier model
# --------------------------------------------------------------------------
def frontier_report(model):
    """Overlap, Loewdin rotation and the resulting Hloc / n_occ per origin."""
    size = model.size
    values = np.linalg.eigvalsh(0.5 * (model.overlap + model.overlap.conj().T))
    print(f"\n{'=' * 72}")
    print(f"frontier model   {len(model.origins)} origins x {model.count} states"
          f"   Loewdin orthonormalisation")
    print(f"  sub-diagonalising {model.selection}")
    for origin in model.origins:
        levels = model.levels[model.origins.index(origin) * model.count:
                              (model.origins.index(origin) + 1) * model.count]
        print(f"  {origin}: fragment levels"
              + "".join(f" {level:9.4f}" for level in levels) + "  eV")

    print(f"\n  overlap S of the {size} states, before Loewdin:"
          f"  eigenvalues {values.min():.4f} .. {values.max():.4f}"
          f"   max|S - 1| = {np.abs(model.overlap - np.eye(size)).max():.4f}")
    gram = np.einsum("kma,kmb->kab", model.columns.conj(), model.columns)
    print(f"  after Loewdin, at every k:  max|U^dag U - I| = "
          f"{np.abs(gram - np.eye(size)).max():.1e}")

    for position, origin in enumerate(model.origins):
        block = list(range(position * model.count,
                           (position + 1) * model.count))
        labels = [f"{origin} AB{k + 1}" for k in range(model.count)]
        report.block(f"{origin} Hloc [eV]", model.hloc[np.ix_(block, block)],
                     labels)
    for position, origin in enumerate(model.origins):
        block = list(range(position * model.count,
                           (position + 1) * model.count))
        labels = [f"{origin} AB{k + 1}" for k in range(model.count)]
        report.block(f"{origin} n_occ", model.nocc[np.ix_(block, block)],
                     labels, trace=True)

    print(f"\n  checks:  max|Hloc - Hloc^dag| = "
          f"{np.abs(model.hloc - model.hloc.conj().T).max():.1e}"
          f"   max|n_occ - n_occ^dag| = "
          f"{np.abs(model.nocc - model.nocc.conj().T).max():.1e}"
          f"   total electrons in the model = {np.trace(model.nocc).real:.4f}")


# --------------------------------------------------------------------------
#  spectral quantities: DOS / PDOS and COHP
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Shells:
    """The orbital rows the spectra are projected on, and how to name them.

    The species come from the data, not from a hard-coded compound: the metal
    is the species of the fragment origins ("Ni1" -> Ni, "Fe" -> Fe) and the
    ligand is whatever else the projectors carry, so the same report runs on
    NiO and FeO.  The names come from the driver's own selection ("Fe 3d",
    "O 2p"), so nothing here has to guess which quantum number the
    pseudopotential decided to write in the `.amn` header.
    """

    metal: str
    ligand: str
    d_label: str
    p_label: str
    s_label: str
    d_rows: List[int]
    p_rows: List[int]
    s_rows: List[int]


def _species_of(name: str) -> str:
    """'Ni1' -> 'Ni', 'Fe' -> 'Fe'."""
    return re.sub(r"\d+$", "", name)


def _spec_label(selection, species, letter, fallback):
    """The spec of `selection` naming this species and shell, else `fallback`."""
    angular = orbitals.SHELL_LETTERS.index(letter)
    for spec in selection:
        chosen, _, l = orbitals.parse_shell(spec)
        if (chosen or "").lower() == species.lower() and l == angular:
            return spec
    return fallback


def _shells(dump, origins, selection) -> Shells:
    """Build the `Shells` of a dump out of its fragment origins.

    The ligand is the species the driver's selection calls a p shell on
    ("O 2p"), not "every species that is not the metal": La and Lu carry their
    own semicore p channel, and taking those would label the O 2p panel
    "La p" / "Lu p" and fold La 5p into the ligand DOS and COHP shares.  When
    the selection names no such shell, the species with the most projectors
    wins, which is the anion in every compound here.
    """
    metals = sorted({_species_of(origin) for origin in origins})
    others = [orbital for orbital in dump.orbitals
              if _species_of(orbital.species) not in metals]
    if not others:
        raise ValueError(f"no ligand species besides {metals} in the projectors")

    ligand = None
    for spec in selection:
        chosen, _, l = orbitals.parse_shell(spec)
        if chosen and l == 1 and _species_of(chosen) not in metals:
            ligand = _species_of(chosen)
            break
    if ligand is None:
        counts = {}
        for orbital in others:
            counts[orbital.species] = counts.get(orbital.species, 0) + 1
        ligand = max(counts, key=counts.get)

    d_rows = [orbital.index for orbital in dump.orbitals
              if orbital.l == 2 and _species_of(orbital.species) in metals]
    if not d_rows:
        raise ValueError(f"no d projectors on {metals} to build a model from")
    p_rows = [orbital.index for orbital in dump.orbitals
              if orbital.l == 1 and _species_of(orbital.species) == ligand]
    s_rows = [orbital.index for orbital in dump.orbitals
              if orbital.l == 0 and _species_of(orbital.species) == ligand]

    metal = metals[0]
    return Shells(
        metal=metal, ligand=ligand,
        d_label=_spec_label(selection, metal, "d", f"{metal} d"),
        p_label=_spec_label(selection, ligand, "p", f"{ligand} p"),
        s_label=_spec_label(selection, ligand, "s", f"{ligand} s"),
        d_rows=d_rows, p_rows=p_rows, s_rows=s_rows)


def _dressed_columns(dump, model, count):
    """The fragment states, orthonormalised once and split in groups.

    The rotation is done on the whole set, not group by group, so the groups
    stay a partition of one orthonormal basis: their densities of states and
    their shares of the COHP then add up to the totals.
    """
    columns = spectral.orthonormal(
        dump, local_model.bloch_columns(model.states, model.eigenvectors,
                                        range(model.size), dump.kfrac,
                                        dump.norb))
    return [columns[:, :, list(group)]
            for group in local_model.character_groups(model, count)]


def _character_shares(dump, groups):
    """Per band, the fraction of the character carried by each group.

    The weights are normalised over the groups, so they sum to one for every
    (k, band) and a weighted sum of the COHP adds back up to the total.
    """
    shares = [spectral.band_weights(dump, blocks).sum(axis=1)
              for blocks in groups]
    total = sum(shares)
    return [share / total for share in shares]


def _dressed_dos(dump, grid, sigma, columns, origins):
    """PDOS of every group of dressed states, summed over the origins.

    `columns` is one list of per-group column blocks per origin, as returned by
    `_dressed_columns`; the curves are per cell (all origins added).
    """
    curves = []
    for group in range(len(columns[0])):
        curve = np.zeros_like(grid)
        for per_origin in columns:
            blocks = per_origin[group]
            curve += spectral.group_dos(dump, grid, sigma, blocks,
                                        range(blocks.shape[2]))
        curves.append(curve)
    return curves


def _multiplets(levels, tolerance=0.05):
    """Position of the states grouped by degenerate level, lowest first.

    The sub-diagonalisation splits a fragment into a handful of levels (for NiO
    three t2g-like and two e_g-like states, for instance), and it is those
    levels -- not the whole block -- that carry the chemical information.
    """
    order = np.argsort(levels)
    groups = []
    for position in order:
        if groups and abs(levels[position] - levels[groups[-1][0]]) < tolerance:
            groups[-1].append(position)
        else:
            groups.append([position])
    return groups


def _atom_weights(model, indices):
    """Weight of each state on each atom of the fragment, images summed.

    `indices` are eigenstate indices; the result is {atom name: weight}, so the
    states can be assigned to the atoms rather than to orbital shells -- the
    bonding partners of the d orbitals are, after all, ligand states.
    """
    weights = {}
    for index in indices:
        for coefficient, state in zip(np.abs(model.eigenvectors[:, index]) ** 2,
                                      model.states):
            name = state.orbital.atom_name
            weights[name] = weights.get(name, 0.0) + float(coefficient)
    return weights


def _comparison_report(dump, grid, mu, sigma, columns, models, groups,
                       shells, out_dir, tag):
    """The metal d and ligand p above, the sub-diagonalised levels below.

    The two panels must not be mistaken for a partition of one another.  The
    density of states of a whole block of orthonormal states is basis
    independent -- sum_s |<s|b>|^2 is the projector on the subspace the block
    spans -- so the AB block, which spans the Ni 3d space, *is* the Ni 3d PDOS
    and cannot be moved in energy by any rotation inside the block.  What a
    sub-diagonalisation does resolve is the level structure inside the block,
    so the lower panel shows one curve per level.
    """
    curves = [
        (f"{shells.d_label} ({len(shells.d_rows)})",
         spectral.projected_dos(dump, grid, sigma, shells.d_rows), "C0", False),
        (f"{shells.p_label} ({len(shells.p_rows)})",
         spectral.projected_dos(dump, grid, sigma, shells.p_rows), "C3", False)]
    total = ("total DOS", spectral.dos(dump, grid, sigma))

    palette = ["C1", "C4", "C2", "C5", "C0", "C3"]
    hybrids, sticks, report = [], [], []
    for name, position in (("AB", 0), ("B", 1)):
        levels = models[0].eigenvalues[groups[0][position]]
        for number, members in enumerate(_multiplets(levels)):
            colour = palette[(2 * position + number) % len(palette)]
            curve = np.zeros_like(grid)
            for blocks_per_origin, index in zip(columns, groups):
                curve += spectral.group_dos(
                    dump, grid, sigma, blocks_per_origin[position], members)
            label = (f"{name} {levels[members].mean():6.2f} eV"
                     f" ({len(members)} states)")
            hybrids.append((label, curve, colour, True))
            sticks.append((levels[members], colour))
            share = float(np.mean([
                spectral.character_split(dump, blocks[position], members,
                                         shells.d_rows)
                for blocks in columns]))
            absolute = [groups[0][position][i] for i in members]
            atoms = _atom_weights(models[0], absolute)
            for key in atoms:
                atoms[key] /= len(members)
            report.append((label, grid[np.argmax(curve)] - mu,
                           spectral.integrate(grid, curve, mu), share, atoms))

    path = plots.spectra_stack(
        grid, mu,
        [{"ylabel": "states / eV / cell", "curves": curves,
          "background": total},
         {"ylabel": "states / eV / cell", "curves": hybrids,
          "background": total, "sticks": sticks}],
        f"{out_dir}/{tag}_dos_compare.png",
        f"{shells.d_label} / {shells.p_label} (top) and the "
        "sub-diagonalised levels (bottom)", tag=tag)

    print("  sub-diagonalised levels  |  weight per atom  |  bands it sits in")
    for label, peak, weight, share, atoms in report:
        per_atom = "  ".join(f"{name} {value:4.2f}"
                             for name, value in sorted(atoms.items()))
        print(f"    {label:28s} peak {peak:+6.2f} eV {weight:5.2f} e"
              f"   {per_atom}"
              f"   ->  {100 * share:4.1f} % {shells.d_label} bands,"
              f" {100 * (1 - share):4.1f} % {shells.p_label}")
    return path


def _dos_report(dump, grid, mu, sigma, shells, out_dir, tag):
    """Total DOS with the metal d and ligand p projections."""
    groups = [(f"{shells.d_label} ({len(shells.d_rows)})", shells.d_rows, "C0"),
              (f"{shells.p_label} ({len(shells.p_rows)})", shells.p_rows, "C3")]
    if shells.s_rows:
        groups.append((f"{shells.s_label} ({len(shells.s_rows)})",
                       shells.s_rows, "C2"))
    path = plots.dos_pdos(dump, grid, mu, sigma,
                          f"{out_dir}/{tag}_dos_pdos.png", groups,
                          title="total and projected DOS", tag=tag)

    total = spectral.dos(dump, grid, sigma)
    values = [spectral.at_energy(grid, total, mu)]
    for rows in (shells.d_rows, shells.p_rows):
        values.append(spectral.at_energy(
            grid, spectral.projected_dos(dump, grid, sigma, rows), mu))
    print(f"  DOS at E_F   total {values[0]:6.3f}"
          f"   {shells.d_label} {values[1]:6.3f}"
          f"   {shells.p_label} {values[2]:6.3f}"
          f"   (states / eV / cell)")
    return path


def _cohp_report(dump, cell, grid, mu, sigma, columns, origins, shells,
                 out_dir, tag):
    """Metal d - ligand p COHP: the total, and the share of each dressed group."""
    labels = ["AB (5 most d-like)", "B (the next 5)", "remaining ligand"]
    partial = [np.zeros_like(grid) for _ in labels]
    for position, origin in enumerate(origins):
        atom = fragments.resolve_atoms(cell, [origin])[0] + 1
        rows = [index for index, orbital in enumerate(dump.orbitals)
                if orbital.atom == atom and orbital.l == 2]
        for curve, share in zip(partial,
                                _character_shares(dump, columns[position])):
            curve += spectral.cohp(dump, grid, sigma, rows, shells.p_rows,
                                   weight=share)

    total = spectral.cohp(dump, grid, sigma, shells.d_rows, shells.p_rows)

    def icohp(curve):
        return -spectral.integrate(grid, curve, mu)

    def label(text, curve):
        return f"{text}   (-ICOHP {icohp(curve):.2f} eV)"

    colours = ["C0", "C3", "C2"]
    curves = [(label(f"total {shells.d_label} - {shells.p_label}", total),
               total, "0.15", 1.8)]
    for name, colour, curve in zip(labels, colours, partial):
        curves.append((label(f"share of {name}", curve), curve, colour, 1.2))
    path = plots.cohp_panel(grid, mu, curves, f"{out_dir}/{tag}_cohp.png",
                            f"COHP between {shells.d_label} and {shells.p_label}",
                            tag=tag)

    print(f"  -ICOHP up to E_F   total {icohp(total):7.3f} eV"
          f"   of which"
          f" antibonding {icohp(partial[0]):6.3f}"
          f"   bonding {icohp(partial[1]):6.3f}"
          f"   ligand {icohp(partial[2]):6.3f}"
          f"   (the three shares add up to the total)")
    return path


def spectral_report(dump, cell, origins, radius, selection, count, mu, kT,
                    out_dir="./plots", sigma=0.08, tag="ni",
                    below=8.0, above=4.0, models=None, frontier=None):
    """DOS/PDOS, dressed-state DOS and COHP; saves the figures to `out_dir`.

    Built only from `dump.energies` and `dump.proj` (through `spectral`), so it
    works the same for a pw dump and for a Wannier run.

    `models` (one fragment per origin) and `frontier` are reused when the
    driver has already built them, so the same octahedra are not diagonalised
    three times in one run.
    """
    with timing.stage(f"spectra: DOS, dressed states and COHP   "
                      f"(sigma = {sigma:.3f} eV, -> {out_dir}/)"):
        return _spectral_report(dump, cell, origins, radius, selection, count,
                                mu, kT, out_dir, sigma, tag, below, above,
                                models, frontier)


def _spectral_report(dump, cell, origins, radius, selection, count, mu, kT,
                     out_dir, sigma, tag, below, above, models=None,
                     frontier=None):
    grid = spectral.energy_grid(mu, below=below, above=above, step=0.02)
    shells = _shells(dump, origins, selection)
    paths = [_dos_report(dump, grid, mu, sigma, shells, out_dir, tag)]

    # ---- after sub-diagonalisation ---------------------------------
    if models is None:
        with timing.timed("spectral: fragment models"):
            models = [local_model.fragment(cell, dump, origin, radius,
                                           selection)
                      for origin in origins]
    with timing.timed("spectral: dressed columns (Loewdin per k)"):
        columns = [_dressed_columns(dump, model, count) for model in models]
    with timing.timed("spectral: dressed DOS"):
        curves = _dressed_dos(dump, grid, sigma, columns, origins)   # per cell
        per_origin = [curve / len(origins) for curve in curves]

    paths.append(plots.spectra(
        grid, mu,
        [("AB: 5 most d-like", per_origin[0], "C3", False),
         ("B: the next 5", per_origin[1], "C0", False),
         ("remaining ligand", per_origin[2], "0.55", False)],
        f"{out_dir}/{tag}_dos_fragment.png",
        "dressed states after sub-diagonalisation",
        ylabel="states / eV / fragment", tag=tag))

    groups = [local_model.character_groups(model, count) for model in models]
    levels = [np.concatenate([model.eigenvalues[index[position]]
                              for model, index in zip(models, groups)])
              for position in (0, 1)]
    where = ",  ".join(
        f"{name} {levels[position].min():6.2f} to {levels[position].max():6.2f} eV"
        for position, name in ((0, "AB"), (1, "B")))
    print(f"  sub-diagonalised levels:  {where}"
          f",  ligand {models[0].eigenvalues[groups[0][2]].min():6.2f} to"
          f" {models[0].eigenvalues[groups[0][2]].max():6.2f} eV")
    # only `count` ligand combinations can mix with the `count` d orbitals, the
    # rest are exactly non-bonding: that is what fixes the size of the blocks
    character = local_model.d_character(models[0])
    print(f"  d weight carried by  AB {character[groups[0][0]].sum():5.3f}"
          f" + B {character[groups[0][1]].sum():5.3f}"
          f" = {character[groups[0][0] + groups[0][1]].sum():5.3f}"
          f" of {count} d orbitals;"
          f" the other {models[0].size - 2 * count} states carry"
          f" {character[groups[0][2]].sum():5.3f}")
    print("  states below E_F per origin"
          f"   AB {spectral.integrate(grid, per_origin[0], mu):5.2f}"
          f"   B {spectral.integrate(grid, per_origin[1], mu):5.2f}"
          f"   ligand {spectral.integrate(grid, per_origin[2], mu):5.2f}")

    paths.append(timing.timed_call(
        "spectral: AB/B comparison panel",
        _comparison_report, dump, grid, mu, sigma, columns, models,
        groups, shells, out_dir, tag))

    # ---- after Loewdin ---------------------------------------------
    if frontier is None:
        frontier = local_model.frontier(cell, dump, origins, radius, selection,
                                        count, mu, kT, models=models)
    model = frontier
    spectra = [spectral.group_dos(dump, grid, sigma, model.columns,
                                  range(position * count, (position + 1) * count))
               for position in range(len(origins))]
    frontier = sum(spectra)
    paths.append(plots.spectra(
        grid, mu,
        [(f"frontier states after Loewdin ({len(spectra) * count})",
          frontier, "C0", True)],
        f"{out_dir}/{tag}_dos_lowdin.png",
        "frontier states after Loewdin, over the total DOS", tag=tag,
        background=("total DOS", spectral.dos(dump, grid, sigma))))
    print(f"  frontier DOS   sum over the origins"
          f" {spectral.integrate(grid, frontier, mu):5.2f} states below E_F")
    if len(spectra) > 1:
        spread = np.abs(spectra[0] - spectra[1]).max()
        print(f"  the two origins agree to {spread:.4f} states / eV")

    paths.append(timing.timed_call(
        "spectral: COHP",
        _cohp_report, dump, cell, grid, mu, sigma, columns, origins,
        shells, out_dir, tag))
    for path in paths:
        print(f"  saved {path}")
    return paths

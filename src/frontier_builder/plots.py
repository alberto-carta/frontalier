"""
The figures.  One function per figure, each saving a PNG and returning the path.

Energy is always plotted as E - E_F, so the Fermi level is the vertical line
at zero and the two drivers produce directly comparable pictures.  The name of
the run (`tag`) goes in the title, so the ortho and Wannier versions of the same
quantity can be put side by side.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import spectral

FONT = 9


def _title(title, tag):
    return f"{title}   [{tag}]" if tag else title


def _frame(ax, ylabel, title, tag, xlabel=True):
    ax.axvline(0.0, color="0.25", lw=1.0, zorder=0)
    ax.axhline(0.0, color="0.8", lw=0.8, zorder=0)
    if xlabel:
        ax.set_xlabel(r"$E - E_F$  (eV)", fontsize=FONT)
    ax.set_ylabel(ylabel, fontsize=FONT)
    ax.set_title(_title(title, tag), fontsize=10)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.tick_params(labelsize=8)


def _save(figure, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _fermi_note(ax, height):
    ax.annotate(r"$E_F$", xy=(0.0, height), xytext=(-5, 0),
                textcoords="offset points", ha="right", va="center",
                fontsize=8, color="0.25")


def dos_pdos(dump, grid, mu, sigma, path, groups=(), title="DOS", tag=""):
    """Total DOS with one curve per orbital group.

    groups: [(label, indices, colour), ...]
    """
    figure, ax = plt.subplots(figsize=(7.0, 4.2))

    total = spectral.dos(dump, grid, sigma)
    ax.fill_between(grid - mu, total, color="0.88", lw=0, zorder=1,
                    label="total")
    ax.plot(grid - mu, total, color="0.45", lw=0.8, zorder=2)

    for label, indices, colour in groups:
        curve = spectral.projected_dos(dump, grid, sigma, indices)
        ax.plot(grid - mu, curve, color=colour, lw=1.5, zorder=3, label=label)

    _frame(ax, "states / eV / cell", title, tag)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_xlim(grid[0] - mu, grid[-1] - mu)
    ax.set_ylim(bottom=0.0)
    _fermi_note(ax, 0.94 * total.max())
    return _save(figure, path)


def _draw(ax, grid, mu, curves, background=None, sticks=None):
    """The curves of one panel, over an optional reference spectrum.

    sticks: [(energies, colour), ...] -- vertical marks for the levels of the
            states the curves describe, so a curve can be read against the
            level it comes from.  They take the colour of their curve and are
            left out of the legend.
    """
    if background is not None:
        label, curve = background
        ax.fill_between(grid - mu, curve, color="0.88", lw=0, zorder=1,
                        label=label)
        ax.plot(grid - mu, curve, color="0.45", lw=0.8, zorder=2)

    for label, spectrum, colour, filled in curves:
        if filled:
            ax.fill_between(grid - mu, spectrum, color=colour, alpha=0.25,
                            lw=0, zorder=3)
        ax.plot(grid - mu, spectrum, color=colour, lw=1.5, zorder=4,
                label=label)

    heights = [np.max(curve) for _, curve, _, _ in curves]
    if background is not None:
        heights.append(np.max(background[1]))
    for energies, colour in sticks or ():
        ax.vlines(np.asarray(energies) - mu, 0.0, 0.07 * max(heights),
                  color=colour, lw=1.6, zorder=5)

    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_xlim(grid[0] - mu, grid[-1] - mu)
    ax.set_ylim(bottom=0.0)
    _fermi_note(ax, 0.94 * max(heights))


def spectra(grid, mu, curves, path, title, tag="",
            ylabel="states / eV / cell", background=None, sticks=None):
    """Spectra against E - E_F, over an optional reference curve.

    curves:     [(label, spectrum, colour, filled), ...]
    background: (label, spectrum), drawn as a grey area behind everything --
                normally the total DOS, so the reader can see how much of it
                the curves account for.
    sticks:     (energies, colour, label), see `_draw`
    """
    figure, ax = plt.subplots(figsize=(7.0, 4.2))
    _draw(ax, grid, mu, curves, background, sticks)
    _frame(ax, ylabel, title, tag)
    return _save(figure, path)


def spectra_stack(grid, mu, panels, path, title, tag="", height=6.4):
    """Two or more panels sharing the energy axis.

    panels: [{"ylabel": ..., "curves": [...], "background": ...,
              "sticks": ...}, ...] top to bottom, with the last three optional.
    Used to put a decomposition (B / AB) directly above or below the quantity
    it explains (the Ni 3d / O 2p projections).
    """
    figure, axes = plt.subplots(len(panels), 1, figsize=(7.0, height),
                                sharex=True)
    for index, (ax, panel) in enumerate(zip(axes, panels)):
        last = index == len(panels) - 1
        _draw(ax, grid, mu, panel["curves"], panel.get("background"),
              panel.get("sticks"))
        _frame(ax, panel["ylabel"], title if index == 0 else "",
               tag if index == 0 else "", xlabel=last)
    return _save(figure, path)


def cohp_panel(grid, mu, curves, path, title, tag="",
               ylabel="-COHP  (eV,  bonding > 0)"):
    """-COHP above, its running integral -ICOHP below, sharing the energy axis.

    curves: [(label, cohp_spectrum, colour, linewidth), ...]
    """
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.0, 5.8), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]})

    for label, curve, colour, width in curves:
        top.plot(grid - mu, -curve, color=colour, lw=width, label=label)
        top.fill_between(grid - mu, -curve, 0.0, color=colour, alpha=0.12,
                         lw=0)
        bottom.plot(grid - mu, -spectral.running_integral(grid, curve),
                    color=colour, lw=width)

    top.legend(fontsize=8, frameon=False, loc="upper right")
    _frame(top, ylabel, title, tag, xlabel=False)
    _frame(bottom, "-ICOHP  (eV)", "", "", xlabel=True)
    bottom.set_title("")
    bottom.set_xlim(grid[0] - mu, grid[-1] - mu)
    top.set_xlim(grid[0] - mu, grid[-1] - mu)
    return _save(figure, path)

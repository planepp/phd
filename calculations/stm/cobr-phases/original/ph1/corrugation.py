"""
Plot a corrugation (height, z) map of a molecule read from a structure file.

Supports any format ASE can auto-detect from the filename, most commonly:
    - geometry.in         (FHI-aims)
    - POSCAR / CONTCAR     (VASP)

Usage:
    python plot_corrugation.py                  # reads ./geometry.in
    python plot_corrugation.py POSCAR            # reads ./POSCAR instead
    python plot_corrugation.py CONTCAR --zscale -0.15 0.15
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from ase.data import covalent_radii
from ase.io import read


def plot_corrugation(atoms, zscale=None, fig=None, ax=None,
                      xlim=None, ylim=None, atoms_size=2000, fontsize=18,
                      cmap_name="RdBu"):
    """
    Scatter-plot atoms in the xy-plane, colored and sized to show corrugation.

    atoms      : ASE Atoms object
    zscale     : [vmin, vmax] for the colorbar (in Å, relative to the mean z).
                 If None, auto-scales symmetrically around 0 to the largest
                 |z-deviation| present in the structure.
    xlim, ylim : optional (min, max) tuples to restrict which atoms are shown
    atoms_size : scaling factor for marker size (scaled by covalent radius)
    """

    positions = atoms.positions
    numbers = atoms.numbers

    idx = (positions[:, 0] > xlim[0]) & (positions[:, 0] < xlim[1]) if xlim else True
    idy = (positions[:, 1] > ylim[0]) & (positions[:, 1] < ylim[1]) if ylim else True
    s = np.logical_and(idx, idy) if (xlim or ylim) else slice(None)

    cmap = plt.get_cmap(cmap_name)

    # height relative to the mean z-position, in Angstrom (so 0 = average height,
    # not the topmost atom) — makes it easier to compare corrugation across molecules
    z = positions[:, 2] - positions[:, 2].mean()

    if zscale is not None:
        norm = Normalize(vmin=zscale[0], vmax=zscale[1])
    else:
        zmax_abs = np.abs(z).max()
        norm = Normalize(vmin=-zmax_abs, vmax=zmax_abs)

    colors = cmap(norm(z[s]))

    if fig is None or ax is None:
        fig, ax = plt.subplots(constrained_layout=True)

    ax.scatter(
        positions[s, 0],
        positions[s, 1],
        color=colors,
        s=atoms_size * covalent_radii[numbers[s]],
        edgecolors="k",
        linewidths=0.5,
    )

    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_xlabel(r"x ($\AA$)", fontsize=fontsize)
    ax.set_ylabel(r"y ($\AA$)", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation="vertical")
    cbar.ax.set_ylabel(r"z ($\AA$)", fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.set_ticks(np.linspace(norm.vmin, norm.vmax, 5))
    cbar.ax.yaxis.set_major_formatter("{x:.2f}")

    return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a corrugation map from a structure file")
    parser.add_argument("structure_file", nargs="?", default="geometry.in",
                         help="Structure file to read (default: geometry.in). "
                              "Also accepts POSCAR, CONTCAR, or anything ASE can "
                              "auto-detect from the filename.")
    parser.add_argument("--atoms-size", type=float, default=2000,
                         help="Marker size scale factor (default: 2000)")
    parser.add_argument("--zscale", type=float, nargs=2, default=None,
                         metavar=("VMIN", "VMAX"),
                         help="Fixed colorbar range in Angstrom, e.g. --zscale -0.15 0.15. "
                              "If omitted, auto-scales symmetrically around the mean height "
                              "based on this molecule's own max deviation.")
    args = parser.parse_args()

    atoms = read(args.structure_file)

    fig, ax = plot_corrugation(atoms, zscale=args.zscale, atoms_size=args.atoms_size)

    out_stem = Path(args.structure_file).stem or Path(args.structure_file).name
    out_png = f"corrugation_map_{out_stem}.png"
    plt.savefig(out_png, dpi=300)
    print(f"Saved {out_png}")
    plt.show()

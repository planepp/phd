#!/usr/bin/env python
"""
Scatter plot of the Br atoms in a structure, coloured by height.
Au and Co atoms are ignored. The cell is repeated 3x3.

Usage:
    python corrugation.py                      # reads CONTCAR
    python corrugation.py POSCAR
    python corrugation.py POSCAR --geom 1 2 3  # geom01/POSCAR, geom02/POSCAR, geom03/POSCAR
    python corrugation.py CONTCAR --geom 1 2 3 --save
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read

parser = argparse.ArgumentParser()
parser.add_argument("file", nargs="?", default="CONTCAR")
parser.add_argument("--geom", type=int, nargs="+",
                    help="Plot <file> from these folders side by side: 1 -> geom01/, 2 -> geom02/ ...")
parser.add_argument("--save", action="store_true", help="Save corrugation.png instead of showing.")
args = parser.parse_args()

if args.geom:
    paths = [Path(f"geom{g:02d}") / args.file for g in args.geom]
else:
    paths = [Path(args.file)]
missing = [str(p) for p in paths if not p.exists()]
if missing:
    raise SystemExit("Not found: " + ", ".join(missing))


def br_data(path):
    """Br positions repeated 3x3, their heights relative to the mean Br height, the cell,
    and the Co/Br composition label."""
    atoms = read(path)
    symbols = atoms.get_chemical_symbols()
    label = f"Co{symbols.count('Co')}Br{symbols.count('Br')}"
    br = atoms[[s == "Br" for s in symbols]]
    if len(br) == 0:
        raise SystemExit(f"No Br atoms in {path}")
    cell = atoms.cell.array
    z = br.positions[:, 2] - br.positions[:, 2].mean()
    shifts = [i * cell[0, :2] + j * cell[1, :2] for i in (-1, 0, 1) for j in (-1, 0, 1)]
    xy = np.concatenate([br.positions[:, :2] + s for s in shifts])
    return xy, np.tile(z, 9), cell, label


data = [br_data(p) for p in paths]

# Same colour scale in every panel, so heights can be compared
vmin = min(z.min() for _, z, _, _ in data)
vmax = max(z.max() for _, z, _, _ in data)

n = len(data)
ncol = min(n, 4)
nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 5 * nrow), squeeze=False)
for ax in axes.flat:
    ax.axis("off")

for ax, path, (xy, z, cell, label) in zip(axes.flat, paths, data):
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=z, s=100, cmap="gray", vmin=vmin, vmax=vmax,
                    edgecolors="k", linewidths=0.5)
    a, b = cell[0, :2], cell[1, :2]
    corners = np.array([[0, 0], a, a + b, b, [0, 0]])
    ax.plot(corners[:, 0], corners[:, 1], "k--", lw=0.8)
    ax.set_aspect("equal")

    ax.set_title(f"{path}\n{label}", fontsize=11)

fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02, label="Δz (Å)")

if args.save:
    fig.savefig("corrugation.png", dpi=200, bbox_inches="tight")
    print("Saved corrugation.png")
else:
    plt.show()

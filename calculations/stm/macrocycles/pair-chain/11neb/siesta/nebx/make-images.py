#!/usr/bin/env python3
"""Generate NEB images for a Lua/flos NEB in SIESTA.

Linear interpolation between two relaxed endpoints, using the
minimum-image convention so atoms crossing the cell boundary move the
short way. Writes image_0.xyz (initial) ... image_{N+1}.xyz (final),
plus initial.fdf and final.fdf.

Usage:
    python3 make_images.py INITIAL FINAL [-n 5] [-o OUTDIR]
Example:
    python3 make_images.py geoini/AuCNHSiBr.XV nebx/geofinx/AuCNHSiBr.XV -n 5 -o nebx
"""
import argparse
import os

import numpy as np
import sisl

p = argparse.ArgumentParser(description="Make NEB images by linear interpolation.")
p.add_argument("initial", help="relaxed initial geometry (e.g. AuCNHSiBr.XV)")
p.add_argument("final", help="relaxed final geometry (e.g. AuCNHSiBr.XV)")
p.add_argument("-n", "--n-images", type=int, default=5, help="intermediate images (default 5)")
p.add_argument("-o", "--outdir", default=".", help="output folder (default: current)")
args = p.parse_args()

init = sisl.Geometry.read(args.initial)
fin = sisl.Geometry.read(args.final)

# Sanity checks: same atoms, same order, same cell
if init.na != fin.na:
    raise SystemExit(f"Different number of atoms: {init.na} vs {fin.na}")
if not np.array_equal(init.atoms.Z, fin.atoms.Z):
    raise SystemExit("Atom order/species differ between initial and final")
if not np.allclose(init.cell, fin.cell, atol=1e-4):
    raise SystemExit("Cells differ between initial and final")

# Displacement with minimum-image convention (fractional coordinates)
dfrac = fin.fxyz - init.fxyz
dfrac -= np.round(dfrac)
dxyz = dfrac @ init.cell

os.makedirs(args.outdir, exist_ok=True)
nseg = args.n_images + 1
for i in range(nseg + 1):
    img = init.copy()
    img.xyz = init.xyz + dxyz * i / nseg
    img.write(os.path.join(args.outdir, f"image_{i}.xyz"))
    if i == 0:
        img.write(os.path.join(args.outdir, "initial.fdf"))
    if i == nseg:
        img.write(os.path.join(args.outdir, "final.fdf"))

disp = np.linalg.norm(dxyz, axis=1)
print(f"Wrote image_0.xyz ... image_{nseg}.xyz to '{args.outdir}'")
print(f"Max atom displacement initial->final: {disp.max():.3f} Ang "
      f"(step between images {disp.max() / nseg:.3f} Ang)")

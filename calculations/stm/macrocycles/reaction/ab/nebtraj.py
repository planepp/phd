#!/usr/bin/env python3
"""
Collect 00/POSCAR, 01/POSCAR, ... into one trajectory file.

Usage (run in the NEB folder):
    python nebtraj.py                 # POSCARs -> band.xyz
    python nebtraj.py CONTCAR         # current/converged images -> band_contcar.xyz

Atoms that VASP folded to the other side of the cell are unfolded frame by
frame, so nothing jumps across the cell in the movie.
"""
import glob
import os
import sys
import numpy as np
from ase.io import read, write

name = sys.argv[1] if len(sys.argv) > 1 else "POSCAR"
dirs = sorted(d for d in glob.glob("[0-9][0-9]") if os.path.isdir(d))
frames = []
for d in dirs:
    f = f"{d}/{name}"
    # endpoints (and images that have not written yet) have no or an empty CONTCAR:
    # fall back to POSCAR so the trajectory still has all frames
    if not os.path.isfile(f) or os.path.getsize(f) == 0:
        f = f"{d}/POSCAR"
        if name != "POSCAR":
            print(f"{d}: no {name} (or empty), using POSCAR")
    at = read(f, format="vasp")
    if frames:                                    # unfold relative to previous frame
        fr = (at.positions - frames[-1].positions) @ np.linalg.inv(at.cell.array)
        fr[:, 2] = 0
        at.positions -= np.round(fr) @ at.cell.array
    frames.append(at)

out = "band.xyz" if name == "POSCAR" else "band_" + name.lower() + ".xyz"   # no "POSCAR"/"CONTCAR" in the name: ASE would read it as VASP
write(out, frames, format="extxyz")
print(f"{len(frames)} frames ({dirs[0]}..{dirs[-1]}) -> {out}")

#!/usr/bin/env python3
"""
Duplicate the adsorbed molecule in a POSCAR by a fraction of lattice vector a.

Usage:
    python duplicate_molecule.py POSCAR_initial POSCAR_two_molecules --dist 25.964961
    python duplicate_molecule.py IN OUT --dist 10 --dir 1 0 0 --species C N H Br

Assumes Cartesian coordinates with Selective dynamics (as in your file).
"""
import argparse
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("infile")
p.add_argument("outfile")
p.add_argument("--species", nargs="+", default=["C", "N", "H", "Br"],
               help="species that make up the molecule")
p.add_argument("--dist", type=float, required=True,
               help="how far to move the copy, in angstrom")
p.add_argument("--dir", type=float, nargs=3, default=None,
               help="Cartesian direction of the move (default: along lattice vector a)")
args = p.parse_args()

lines = open(args.infile).read().splitlines()
scale = float(lines[1])
lat = scale * np.array([[float(x) for x in l.split()] for l in lines[2:5]])
species = lines[5].split()
counts = list(map(int, lines[6].split()))
body = lines[9:9 + sum(counts)]

direction = np.array(args.dir) if args.dir else lat[0]
shift = args.dist * direction / np.linalg.norm(direction)
inv = np.linalg.inv(lat)

new_lines, new_counts, start = [], [], 0
for sp, n in zip(species, counts):
    block = body[start:start + n]
    out = list(block)
    if sp in args.species:
        for l in block:
            parts = l.split()
            c = np.array(list(map(float, parts[:3]))) + shift
            fa = (c @ inv)[0]
            if not 0.0 <= fa < 1.0:
                print(f"Warning: shifted {sp} atom has frac a = {fa:.3f} (outside cell)")
            out.append(" %20.16f %20.16f %20.16f   %s" % (*c, "   ".join(parts[3:])))
        n *= 2
    new_lines += out
    new_counts.append(n)
    start += len(block)

header = lines[:6] + [" " + " ".join(f"{c:3d}" for c in new_counts)] + lines[7:9]
with open(args.outfile, "w") as f:
    f.write("\n".join(header + new_lines) + "\n")
print("Counts:", dict(zip(species, new_counts)), "->", args.outfile)

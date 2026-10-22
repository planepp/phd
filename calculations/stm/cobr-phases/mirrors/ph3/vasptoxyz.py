#!/usr/bin/env python3
"""
Convert a VASP POSCAR file to an XYZ file, repeating the cell to make
a larger slab (e.g. for a Blender render of a surface).

Usage:
    python poscar_to_xyz.py [POSCAR] [output.xyz] [nx] [ny] [nz]

Defaults:
    POSCAR file : POSCAR
    output file : output.xyz
    repeat      : 2 2 1   (i.e. 2x2x1 supercell = 4 copies total,
                            good for surfaces where z is the slab normal)

Examples:
    python poscar_to_xyz.py
    python poscar_to_xyz.py POSCAR slab_4x.xyz
    python poscar_to_xyz.py POSCAR slab_4x.xyz 2 2 1
    python poscar_to_xyz.py POSCAR slab_4x.xyz 4 1 1   # repeat 4x along x only
"""

import sys
from ase.io import read, write

def main():
    infile = sys.argv[1] if len(sys.argv) > 1 else "POSCAR"
    outfile = sys.argv[2] if len(sys.argv) > 2 else "output.xyz"
    nx = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    ny = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    nz = int(sys.argv[5]) if len(sys.argv) > 5 else 1

    atoms = read(infile, format="vasp")
    supercell = atoms.repeat((nx, ny, nz))

    write(outfile, supercell)
    print(f"Read {infile}: {len(atoms)} atoms")
    print(f"Repeated ({nx}, {ny}, {nz}) -> {len(supercell)} atoms")
    print(f"Wrote {outfile}")

if __name__ == "__main__":
    main()

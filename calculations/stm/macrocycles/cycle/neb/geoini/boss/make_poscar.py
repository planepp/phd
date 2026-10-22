"""Write the POSCAR for any (x, y, angle) using the same geometry as function.py,
with the molecule shifted to the middle of the cell for easier viewing.

Usage:
  python3 make_poscar.py              # uses the last predicted global minimum in boss.out
  python3 make_poscar.py x y angle    # any point you choose
"""
import sys
import numpy as np
from ase.io import read, write

N_SURF = 9        # the slab is a 9x9 Au(111) supercell
CELL_C = None     # set e.g. 25.0 to change the cell height (A); None keeps surface.vasp's

if len(sys.argv) == 4:
    x, y, angle_deg = map(float, sys.argv[1:])
else:
    lines = open("boss.out").read().splitlines()
    idx = [i for i, l in enumerate(lines) if "Global minimum prediction" in l]
    x, y, angle_deg = map(float, lines[idx[-1] + 1].split()[:3])

# Same steps as in function.py
surface = read("surface.vasp")
mol = read("molecule.vasp").copy()
mol.rotate(a=angle_deg, v='z', center='COM')
original_com = mol.get_center_of_mass()
target_com = np.array([x, y, original_com[2]])
mol.positions += target_com - original_com

# Move the molecule towards the middle of the cell, in whole steps of the
# Au(111) surface lattice vectors, so its adsorption site does not change
a1 = surface.cell[0][:2] / N_SURF
a2 = surface.cell[1][:2] / N_SURF
centre = (surface.cell[0][:2] + surface.cell[1][:2]) / 2
delta = centre - mol.get_center_of_mass()[:2]
n, m = np.round(np.linalg.solve(np.column_stack([a1, a2]), delta))
mol.positions[:, :2] += n * a1 + m * a2

system = surface + mol
if CELL_C is not None:
    cell = system.cell.copy()
    cell[2] = [0.0, 0.0, CELL_C]
    system.set_cell(cell, scale_atoms=False)

out = f"POSCAR_{x:.3f}_{y:.3f}_{angle_deg:.1f}"
write(out, system, format="vasp")
print(f"Wrote {out}  (x={x:.4f} A, y={y:.4f} A, angle={angle_deg:.2f} deg; "
      f"shifted by {int(n)} a1 + {int(m)} a2)")

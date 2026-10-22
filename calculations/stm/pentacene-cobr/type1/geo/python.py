from ase.io import read
import numpy as np
traj = read('OUTCAR', ':')
mol = list(range(196, 232))        # molecule indices (0-based), adjust
slab = list(range(0, 196))         # slab indices, adjust
for i, a in enumerate(traj[::5]):
    f = a.get_forces()
    z_mol = a.positions[mol, 2].min()
    z_top = np.sort(a.positions[slab, 2])[-10:].mean()
    print(i*5, round(f[mol, 2].sum(), 4), round(z_mol - z_top, 4))

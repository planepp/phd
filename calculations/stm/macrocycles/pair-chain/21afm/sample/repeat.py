from ase.io import read, write
import numpy as np

atoms = read("POSCAR")

# Repeat 3 x 1 x 1
atoms = atoms.repeat((2, 1, 1))
atoms.wrap()

# Your existing POTCAR / POSCAR element order
element_order = ["Au", "C", "N", "H", "Si", "Br"]

symbols = np.array(atoms.get_chemical_symbols())

# Group atoms according to element_order
order = np.concatenate([
    np.where(symbols == element)[0]
    for element in element_order
])

atoms = atoms[order]

write(
    "POSCAR_3x1x1",
    atoms,
    format="vasp",
    direct=True,
    sort=False
)

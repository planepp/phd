from ase.io import read, write

atoms = read("POSCAR_mol")
atoms *= (3, 3, 1)
write("POSCAR_331", atoms, format="vasp")

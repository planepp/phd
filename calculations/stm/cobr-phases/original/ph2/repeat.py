from ase.io import read, write

atoms = read("POSCAR")
atoms *= (3, 1, 1)
write("POSCAR_331", atoms, format="vasp")

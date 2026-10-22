from ase.io import read, write

atoms = read("CONTCAR")
atoms *= (3, 3, 1)
write("POSCAR", atoms, format="vasp")

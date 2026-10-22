from ase.io import read

atoms = read("POSCAR")

z = [atom.position[2] for atom in atoms if atom.symbol != "Au"]

print("Average z:", sum(z) / len(z))

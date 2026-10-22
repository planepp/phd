from ase.io import read

atoms = read("POSCAR")

# Seleccionar todos los átomos excepto Au
mol = [atom for atom in atoms if atom.symbol != "Au"]

# Media de la coordenada z
z_mean = sum(atom.position[2] for atom in mol) / len(mol)

print(f"Z media de la molécula (sin Au): {z_mean:.6f} Å")

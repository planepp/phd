from ase.io import read, write
import numpy as np
import subprocess
import os

# Reference energies for binding energy calculation
# Structures taken from a previous full-relaxation run
emol = -353.73974462     # Energy of isolated molecule (eV)
esurf = -727.61950236   # Energy of clean surface (eV)

def f(X):
    if X.ndim == 2:
        x, y, angle_deg = X[0]
    else:
        x, y, angle_deg = X

    # Load structures
    surface = read("surface.xyz")
    molecule = read("molecule.xyz")
    mol = molecule.copy()

    # Rotate molecule around Z-axis at its COM
    mol.rotate(a=angle_deg, v='z', center='COM')

    # Preserve original height (z) of the molecule’s COM
    original_com = mol.get_center_of_mass()
    target_com = np.array([x, y, original_com[2]])
    displacement = target_com - original_com
    mol.positions += displacement

    # Combine system and write POSCAR for VASP
    system = surface + mol
    write("POSCAR", system, format="vasp")

    # Run VASP (adapt command to your environment)
    subprocess.run("srun vasp_std", shell=True)

    # Extract energy from OUTCAR
    energy = None
    with open("OUTCAR", "r") as f_out:
        for line in f_out:
            if "energy  without entropy=" in line:
                parts = line.strip().split()
                energy = float(parts[-1])  # Last value is energy(sigma->0)

    if energy is None:
        raise RuntimeError("Energy not found in OUTCAR.")

    # Compute binding energy
    return energy - esurf - emol

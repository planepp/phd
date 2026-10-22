from ase.io import read, write
import numpy as np
import subprocess
import os

# Reference energies for binding energy calculation
# Structures taken from a previous full-relaxation run
emol = -724.91229       # Energy of isolated molecule (eV)
esurf = -840.04345   # Energy of clean surface (eV)

def f(X):
    if X.ndim == 2:
        x, y, angle_deg = X[0]
    else:
        x, y, angle_deg = X

    # Load structures
    surface = read("surface.vasp")
    molecule = read("molecule.vasp")
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

    # delete the previous OUTCAR, so a crashed run does not mess everything up
    if os.path.exists("OUTCAR"):
        os.remove("OUTCAR")

    # Run VASP (adapt command to your environment)
    subprocess.run("srun /projappl/project_2001912/vasp-env/view/bin/vasp_gam", shell=True, check=True)

    # Extract energy from OUTCAR
    energy = None
    converged = False  # CHANGED
    with open("OUTCAR", "r") as f_out:
        for line in f_out:
            if "aborting loop because EDIFF is reached" in line:  # CHANGED
                converged = True
            if "energy  without entropy=" in line:
                parts = line.strip().split()
                energy = float(parts[-1])  # Last value is energy(sigma->0)

    if energy is None:
        raise RuntimeError("Energy not found in OUTCAR.")
    if not converged:  # CHANGED
        raise RuntimeError("SCF did not converge within NELM.")

    # Compute binding energy
    return energy - esurf - emol

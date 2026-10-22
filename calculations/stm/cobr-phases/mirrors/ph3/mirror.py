from ase.io import read

mol = read("POSCAR")

# Mirror through the plane x = x_center (perpendicular to x, through COM)
x_center = mol.get_center_of_mass()[0]
mol.positions[:, 0] = 2 * x_center - mol.positions[:, 0]
mol.wrap()

# Rotate atoms and cell together by -30° around z, about the center of mass
mol.rotate(-30, 'z', center='COM', rotate_cell=True)

mol.write("mirror.vasp")

from ase.io import read, write
import numpy as np

atoms = read("geo99.in")

# move based on com
com = atoms.get_center_of_mass()
target_com = np.array([0,0,0])
shift = target_com - com

# or choose shift
shift = np.array([-8.816, -11.758, 0])

# Move all atoms
atoms.positions += shift

# Save the moved molecule
write("geo999.in", atoms)


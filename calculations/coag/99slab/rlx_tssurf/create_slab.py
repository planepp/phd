# Create slab
from ase.build import fcc100, fcc110, fcc111
ag100 = fcc100('Ag', a=4.157, size=(9,9,4), vacuum=100)

from ase.io import write
ag100.write('geo99.in', format='aims')

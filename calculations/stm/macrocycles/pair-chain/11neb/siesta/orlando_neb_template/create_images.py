import sisl

# Read initial geometry
init = sisl.Geometry.read('input.fdf')

# Create images (also initial and final [0, 180])
for i, ang in enumerate([0, 30, 60, 90, 120, 150, 180]):
    # Rotate around atom 3 (Oxygen), and only rotate atoms
    #  [4, 5] (rotating 3 is a no-op)
    print("Rotating {} angle H with 4,5 index ".format(str(ang)))
    new = init.rotate(ang,v=[0,0,1], origo=3, atoms=[4, 5], only='xyz')
    new.write('image_{}.xyz'.format(i))
new.write('initial.fdf')
new.write('final.fdf')
print("Image Generated!")

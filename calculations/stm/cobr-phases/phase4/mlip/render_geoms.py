"""
Render a top view of every geometry with the look of a template .blend
(materials, light, world, render settings), framed exactly like the
STM panels made by paper_sheet.py.

Run from the folder containing geom02, geom03, ...:

  blender -b ph4.blend --python render_geoms.py
  blender -b ph4.blend --python render_geoms.py -- --nums 2,3,19 --res 1200
  blender -b ph4.blend --python render_geoms.py -- --radius Co=1.3 Br=1.9
  blender -b ph4.blend --python render_geoms.py -- --omit          # include Au

Everything after "--" goes to this script. The template file is never saved.

Framing
-------
The STM images span --repeat x --repeat POSCAR cells (default 3 x 3).
Same transform as the STM grid: mirror in x, rotate by --angle about z,
square crop of side --side x (cell height), centred on the cell centre,
periodic images included. With the same --side/--angle as paper_sheet.py,
panel (a) of this grid shows the same patch of surface as panel (a) of the
STM grid, in the same orientation and at the same scale.
Use --ortho 39 to frame a fixed width in Angstrom instead.

Output: geomXX/<--outdir>/<--outname> (default geomXX/blender/geom_top.png).
POSCARs are only read, never written.
"""
import argparse
import math
import sys
from pathlib import Path

import bpy
import numpy as np

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
p = argparse.ArgumentParser(prog='render_geoms.py')
p.add_argument('--root', type=Path, default=Path('.'))
p.add_argument('--nums', nargs='+',
               default=['2 3 4 5 6 7 8 9 10 13 14 15 19 20 21 22 23 24'])
p.add_argument('--poscar', default='POSCAR',
               help='Geometry file inside each geomXX (the one used for the STM).')
p.add_argument('--angle', type=float, default=-130)
p.add_argument('--no-mirror', action='store_true')
p.add_argument('--side', type=float, default=0.7,
               help='Square side in units of the cell height (as in paper_sheet.py).')
p.add_argument('--repeat', default='3',
               help='How many POSCAR cells the STM image spans along a and b, '
                    'e.g. 3 or 3,3. Count the repeats of the motif in one STM image.')
p.add_argument('--ortho', type=float, default=0,
               help='Fixed view width in Angstrom (overrides --side).')
p.add_argument('--radius', nargs='*', default=[],
               help='Ball radii in Angstrom, e.g. Au=1.6 Co=1.3. '
                    'Default: the visual size used in the template.')
p.add_argument('--radius-scale', type=float, default=1.0,
               help='Multiply all ball radii, e.g. 0.5 for half-size balls.')
p.add_argument('--omit', nargs='*', default=['Au'],
               help='Elements not to draw (default: Au). Use --omit with no '
                    'arguments to draw everything.')
p.add_argument('--res', type=int, default=0, help='Image size in px (0 = template).')
p.add_argument('--engine', default='', help='BLENDER_EEVEE / CYCLES (default: template).')
p.add_argument('--samples', type=int, default=0, help='Render samples (0 = template).')
p.add_argument('--outdir', default='blender')
p.add_argument('--outname', default='geom_top.png')
args = p.parse_args(argv)

NAMES = {'H': 'Hydrogen', 'C': 'Carbon', 'N': 'Nitrogen', 'O': 'Oxygen',
         'F': 'Fluorine', 'S': 'Sulfur', 'Cl': 'Chlorine', 'Br': 'Bromine',
         'I': 'Iodine', 'Co': 'Cobalt', 'Fe': 'Iron', 'Ni': 'Nickel',
         'Cu': 'Copper', 'Ag': 'Silver', 'Au': 'Gold', 'Pt': 'Platinum',
         'Pd': 'Palladium', 'Mn': 'Manganese', 'Zn': 'Zinc', 'Si': 'Silicon'}
FALLBACK = {'H': ((0.9, 0.9, 0.9), 0.5), 'C': ((0.2, 0.2, 0.2), 0.9),
            'N': ((0.1, 0.2, 0.9), 0.9), 'O': ((0.9, 0.1, 0.1), 0.9)}


# ------------------------------------------------------------ POSCAR reader
def read_poscar(path):
    lines = Path(path).read_text().splitlines()
    scale = float(lines[1].split()[0])
    cell = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)])
    if scale < 0:                                          # volume given
        scale = (-scale / abs(np.linalg.det(cell))) ** (1 / 3)
    cell *= scale
    toks = lines[5].split()
    if toks[0].isdigit():                                  # VASP 4: names in line 1
        syms, counts, i = lines[0].split(), [int(x) for x in toks], 6
    else:
        syms, counts, i = toks, [int(x) for x in lines[6].split()], 7
    if lines[i].strip()[0] in 'sS':                        # selective dynamics
        i += 1
    direct = lines[i].strip()[0] in 'dD'
    n = sum(counts)
    xyz = np.array([[float(x) for x in lines[i + 1 + k].split()[:3]] for k in range(n)])
    pos = xyz @ cell if direct else xyz * scale
    symbols = [s for s, c in zip(syms, counts) for _ in range(c)]
    return cell, pos, symbols


# ------------------------------------------------------------ template look
scene = bpy.context.scene
template = {}                                              # name -> (material, radius)
for o in list(bpy.data.objects):
    if o.name.endswith('_ball') and o.parent is not None:
        name = o.name[:-5]
        mat = o.data.materials[0] if o.data.materials else None
        r = 0.5 * o.dimensions[0] * o.parent.scale[0]      # visual radius in the template
        template[name] = (mat, r)
for o in list(bpy.data.objects):                           # remove template atoms
    if o.name.endswith('_ball') or o.name.endswith('_mesh'):
        bpy.data.objects.remove(o, do_unlink=True)

radius_override = {}
for item in args.radius:
    s, v = item.split('=')
    radius_override[s] = float(v)

cam = scene.camera
cam.rotation_euler = (0.0, 0.0, 0.0)                       # look straight down, x right
cam.data.type = 'ORTHO'
if args.res:
    scene.render.resolution_x = scene.render.resolution_y = args.res
else:
    scene.render.resolution_y = scene.render.resolution_x
scene.render.resolution_percentage = 100
if args.engine:
    scene.render.engine = args.engine
if args.samples:
    if scene.render.engine == 'CYCLES':
        scene.cycles.samples = args.samples
    else:
        scene.eevee.taa_render_samples = args.samples

bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=1.0)
sphere = bpy.context.active_object
sphere.data.name = 'atom_sphere'
for poly in sphere.data.polygons:
    poly.use_smooth = True
sphere_mesh = sphere.data
sphere_mesh.materials.append(None)                         # one slot, filled per object
bpy.data.objects.remove(sphere, do_unlink=True)


def material_for(sym):
    name = NAMES.get(sym, sym)
    if name in template and template[name][0] is not None:
        return template[name][0]
    mat = bpy.data.materials.get(f'auto_{sym}') or bpy.data.materials.new(f'auto_{sym}')
    col = FALLBACK.get(sym, ((0.6, 0.6, 0.6), 1.2))[0]
    mat.diffuse_color = (*col, 1)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (*col, 1)
    return mat


def radius_for(sym):
    if sym in radius_override:
        return radius_override[sym] * args.radius_scale
    name = NAMES.get(sym, sym)
    if name in template:
        return template[name][1] * args.radius_scale
    return FALLBACK.get(sym, (None, 1.2))[1] * args.radius_scale


# ------------------------------------------------------------ per geometry
_rep = [int(x) for x in args.repeat.replace('x', ',').split(',')]
rep_a, rep_b = (_rep[0], _rep[0]) if len(_rep) == 1 else _rep[:2]
nums = [int(x) for s in args.nums for x in s.replace(',', ' ').split()]
t = math.radians(args.angle)
R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
made = []

for n in nums:
    gname = f'geom{n:02d}'
    gdir = args.root / gname
    pfile = gdir / args.poscar
    if not pfile.exists():
        print(f'{gname}: no {args.poscar}, skipped')
        continue
    cell, pos, symbols = read_poscar(pfile)
    keep_el = np.array([s not in args.omit for s in symbols])
    pos, symbols = pos[keep_el], [s for s, k in zip(symbols, keep_el) if k]
    if not symbols:
        print(f'{gname}: nothing left after omitting {args.omit}, skipped')
        continue
    a, b = cell[0, :2], cell[1, :2]
    # the STM image shows rep_a x rep_b POSCAR cells
    A, B = rep_a * a, rep_b * b
    centre = 0.5 * (A + B)
    height = abs(a[0] * b[1] - a[1] * b[0]) / np.linalg.norm(a)   # one POSCAR cell
    stm_height = abs(A[0] * B[1] - A[1] * B[0]) / np.linalg.norm(A)
    width = args.ortho or args.side * stm_height

    # periodic images covering the (rotated) square plus the biggest ball
    rmax = max(radius_for(s) for s in set(symbols))
    reach = width / math.sqrt(2) + rmax + 1.0
    na = int(math.ceil(reach / height)) + 2
    shifts = [i * a + j * b for i in range(-na, na + 1) for j in range(-na, na + 1)]
    xy, z, sy = [], [], []
    for sh in shifts:
        q = pos[:, :2] + sh
        keep = np.linalg.norm(q - centre, axis=1) < reach
        xy.append(q[keep]); z.append(pos[keep, 2]); sy += [s for s, k in zip(symbols, keep) if k]
    xy, z = np.vstack(xy), np.concatenate(z)

    # same transform as the STM images: mirror x, then rotate, about the centre
    d = xy - centre
    if not args.no_mirror:
        d[:, 0] = -d[:, 0]
    d = d @ R.T
    verts = np.column_stack([d, z - z.max()])              # top atom at z = 0

    objs = []
    for sym in sorted(set(sy)):
        sel = np.array([s == sym for s in sy])
        me = bpy.data.meshes.new(f'{gname}_{sym}_verts')
        me.from_pydata(verts[sel].tolist(), [], [])
        holder = bpy.data.objects.new(f'{gname}_{sym}', me)
        holder.instance_type = 'VERTS'
        holder.show_instancer_for_render = False
        ball = bpy.data.objects.new(f'{gname}_{sym}_ball', sphere_mesh)
        ball.parent = holder
        r = radius_for(sym)
        ball.scale = (r, r, r)
        scene.collection.objects.link(holder)
        scene.collection.objects.link(ball)
        # material on the object, so the shared sphere mesh stays neutral
        ball.material_slots[0].link = 'OBJECT'
        ball.material_slots[0].material = material_for(sym)
        objs += [holder, ball]

    cam.location = (0.0, 0.0, 50.0)
    cam.data.ortho_scale = width
    cam.data.clip_start, cam.data.clip_end = 0.1, 200.0

    out = gdir / args.outdir / args.outname
    out.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(out.resolve())
    bpy.ops.render.render(write_still=True)
    made.append(gname)
    print(f'{gname}: {len(sy)} atoms, view {width:.2f} A -> {out}')

    for o in objs:
        bpy.data.objects.remove(o, do_unlink=True)

print(f'Rendered {len(made)} geometries: {" ".join(made)}')

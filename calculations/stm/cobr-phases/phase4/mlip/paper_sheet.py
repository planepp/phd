#!/usr/bin/env python
"""
Paper-quality STM grid: one square, mirrored + rotated image per geometry,
no titles, no labels.

How it works
------------
Each input PNG shows one periodic surface cell (the dark parallelogram).
The script finds that parallelogram, reads out the cell content, and treats
it as periodic. The output is then sampled from the periodic surface:

    output = Rot(angle) . Mirror_x . (input)

which is exactly what you would get by applying mirror.py to the POSCAR
(mirror x through the COM, then ase rotate(angle, 'z', rotate_cell=True))
and recomputing the STM image. Because the surface is periodic, the
rotated image has no empty corners and can be cropped to any square.

No POSCAR is touched. With --write-poscars the transformed geometry is also
written next to the original under a NEW name (existing files are skipped,
never overwritten).

Geometry grid with the same letters/layout (renders from render_geoms.py):
  python paper_sheet.py --raw --subdir blender --pattern geom_top.png \
         --cell 400 --out geom_grid_structures.png

Usage (from the folder containing geom02, geom03, ...):
  python paper_sheet.py
  python paper_sheet.py --pattern '*z2.7*V0.3*0.5*.png' --cols 6
  python paper_sheet.py --no-mirror --angle 0        # just square crops
  python paper_sheet.py --side 1.5 --cell 400 --out fig.png
"""
import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

DEFAULT_NUMS = '2 3 4 5 6 7 8 9 10 13 14 15 19 20 21 22 23 24'

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, default=Path('.'))
p.add_argument('--nums', nargs='+', default=[DEFAULT_NUMS],
               help='Geometry numbers, e.g. "2 3 19" or "2,3,19".')
p.add_argument('--subdir', default='afm/images/stm')
p.add_argument('--pattern', default='*z2.7*V0.3*.png',
               help='Glob selecting the z=2.7, V=0.3, 50%% s image. '
                    'Adjust to your file naming.')
p.add_argument('--angle', type=float, default=-130,
               help='Rotation about z in degrees (ASE convention, + = CCW).')
p.add_argument('--no-mirror', action='store_true', help='Skip the x mirror.')
p.add_argument('--side', type=float, default=1.0,
               help='Square side in units of the cell height (1.0 ~ one cell).')
p.add_argument('--cell', type=int, default=0,
               help='Output tile size in px (0 = native resolution).')
p.add_argument('--cols', type=int, default=0, help='0 = roughly square grid.')
p.add_argument('--gap', type=int, default=4, help='Gap between tiles in px.')
p.add_argument('--bg', default='white', help='Gap/background colour.')
p.add_argument('--inset', type=float, default=3.0,
               help='Pixels to stay away from the anti-aliased cell border.')
p.add_argument('--no-letters', action='store_true',
               help='Do not draw the (a), (b), ... panel letters.')
p.add_argument('--letter-fmt', default='({})',
               help='Letter format, e.g. "({})" -> (a), "{}" -> a, "{})" -> a).')
p.add_argument('--letter-size', type=float, default=0.11,
               help='Letter height as a fraction of the tile size.')
p.add_argument('--tiles-dir', type=Path, default=None,
               help='Also save every processed tile here.')
p.add_argument('--raw', action='store_true',
               help='Use the images as they are (e.g. Blender renders): '
                    'centre-crop to square, no cell detection, mirror or rotation.')
p.add_argument('--out', default='geom_grid_z2.7_V0.3_s0.5.png',
               help='.png / .tif / .pdf / .jpg')
p.add_argument('--dpi', type=int, default=600, help='DPI tag written to the file.')
p.add_argument('--write-poscars', action='store_true',
               help='Also write the mirrored+rotated POSCAR for each geometry.')
p.add_argument('--poscar-in', default='POSCAR')
p.add_argument('--poscar-out', default=None,
               help='Default: POSCAR_mirror_rot<angle>')
args = p.parse_args()


# ---------------------------------------------------------------- geometry
def find_cell(rgb):
    """Locate the dark parallelogram. Returns origin, a, b in y-up pixel
    coords (x right, y up), origin = bottom-left corner."""
    gray = rgb.mean(axis=2)
    mask = gray < 245                                     # not white background
    mask = ndimage.binary_fill_holes(mask)                # bright spots inside
    mask = ndimage.binary_opening(mask, iterations=3)     # cut off title text
    lab, n = ndimage.label(mask)
    if n == 0:
        raise ValueError('no plot region found')
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    comp = lab == (np.argmax(sizes) + 1)                  # drop the title text

    rows = np.where(comp.any(axis=1))[0]
    left = np.array([np.argmax(comp[r]) for r in rows], float)
    right = np.array([comp.shape[1] - 1 - np.argmax(comp[r, ::-1]) for r in rows], float)

    def robust_line(y, x):
        k, c = np.polyfit(y, x, 1)
        for _ in range(3):                                # ignore notches
            ok = np.abs(x - (k * y + c)) < 2.0
            k, c = np.polyfit(y[ok], x[ok], 1)
        return k, c

    kl, cl = robust_line(rows, left)
    kr, cr = robust_line(rows, right + 1)
    k = 0.5 * (kl + kr)                                   # common edge slope
    width = np.median((right + 1) - left)
    y_top, y_bot = rows.min(), rows.max() + 1             # image rows (y down)

    x_bl = kl * y_bot + cl
    x_tl = kl * y_top + cl
    origin = np.array([x_bl, -y_bot])
    a = np.array([width, 0.0])
    b = np.array([x_tl - x_bl, y_bot - y_top])
    return origin, a, b


def cell_array(rgb, origin, a, b, inset):
    """Resample the parallelogram onto a regular (v, u) grid."""
    la, lb = np.linalg.norm(a), np.linalg.norm(b)
    nu, nv = int(round(la)), int(round(lb))
    eu, ev = inset / la, inset / lb
    u = eu + (np.arange(nu) + 0.5) / nu * (1 - 2 * eu)
    v = ev + (np.arange(nv) + 0.5) / nv * (1 - 2 * ev)
    U, V = np.meshgrid(u, v)
    X = origin[0] + U * a[0] + V * b[0]
    Y = origin[1] + U * a[1] + V * b[1]
    rows, cols = -Y - 0.5, X - 0.5                        # back to image index
    return np.stack([ndimage.map_coordinates(rgb[..., ch], [rows, cols],
                                             order=1, mode='nearest')
                     for ch in range(3)], axis=-1)


def transform(cell, a, b, angle, mirror, side_px):
    """Sample the periodic surface: out = R(angle) . M . in, square crop."""
    nv, nu = cell.shape[:2]
    S = int(round(side_px))
    j, i = np.meshgrid(np.arange(S) + 0.5 - S / 2, np.arange(S) + 0.5 - S / 2)
    q = np.stack([j, -i])                                 # y-up, centred
    t = np.deg2rad(-angle)                                # inverse rotation
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    pnt = np.tensordot(R, q, axes=1)
    if mirror:
        pnt[0] = -pnt[0]                                  # mirror is its own inverse
    pnt += ((a + b) / 2)[:, None, None]                   # centre of the cell
    uv = np.tensordot(np.linalg.inv(np.column_stack([a, b])), pnt, axes=1)
    iu = uv[0] * nu - 0.5
    iv = uv[1] * nv - 0.5
    out = np.stack([ndimage.map_coordinates(cell[..., ch], [iv, iu],
                                            order=1, mode='grid-wrap')
                    for ch in range(3)], axis=-1)
    return np.clip(out, 0, 255).astype(np.uint8)


def write_poscar(folder):
    from ase.io import read
    src = folder / args.poscar_in
    name = args.poscar_out or f'POSCAR_mirror_rot{args.angle:g}'
    dst = folder / name
    if not src.exists():
        print(f'{folder.name}: no {args.poscar_in}, POSCAR skipped')
        return
    if dst.exists():
        print(f'{folder.name}: {name} already exists, not overwritten')
        return
    mol = read(src)
    if not args.no_mirror:
        xc = mol.get_center_of_mass()[0]
        mol.positions[:, 0] = 2 * xc - mol.positions[:, 0]
        mol.wrap()
    mol.rotate(args.angle, 'z', rotate_cell=True)
    mol.write(dst, format='vasp', direct=True)
    print(f'{folder.name}: wrote {name}')


# ---------------------------------------------------------------- main
nums = [int(x) for s in args.nums for x in s.replace(',', ' ').split()]
if not nums:
    raise SystemExit('No geometry numbers given')
if args.tiles_dir:
    args.tiles_dir.mkdir(parents=True, exist_ok=True)

tiles = []
for n in nums:
    name = f'geom{n:02d}'
    if args.write_poscars:
        write_poscar(args.root / name)
    files = sorted((args.root / name / args.subdir).glob(args.pattern))
    if not files:
        print(f'{name}: no file matching {args.pattern}')
        tiles.append((name, None))
        continue
    if len(files) > 1:
        print(f'{name}: {len(files)} matches, using {files[0].name}')
    if args.raw:
        im = Image.open(files[0]).convert('RGBA')
        s0 = min(im.size)
        l0, t0 = (im.width - s0) // 2, (im.height - s0) // 2
        im = im.crop((l0, t0, l0 + s0, t0 + s0))
        tile = Image.new('RGB', im.size, args.bg)
        tile.paste(im, mask=im.split()[3])                # transparent -> background
        tiles.append((name, tile))
        if args.tiles_dir:
            tile.save(args.tiles_dir / f'{name}.png')
        continue
    rgb = np.asarray(Image.open(files[0]).convert('RGB'), float)
    try:
        origin, a, b = find_cell(rgb)
    except ValueError as e:
        print(f'{name}: {e}')
        tiles.append((name, None))
        continue
    cell = cell_array(rgb, origin, a, b, args.inset)
    side = args.side * b[1]                               # cell height in px
    tile = Image.fromarray(transform(cell, a, b, args.angle,
                                     not args.no_mirror, side))
    tiles.append((name, tile))
    if args.tiles_dir:
        tile.save(args.tiles_dir / f'{name}.png')

found = [t for _, t in tiles if t is not None]
if not found:
    raise SystemExit(f'No images matching {args.pattern}')

C = args.cell or min(t.width for t in found)
cols = args.cols or math.ceil(math.sqrt(len(tiles)))
rows = math.ceil(len(tiles) / cols)
G = args.gap
W, H = cols * C + (cols - 1) * G, rows * C + (rows - 1) * G
sheet = Image.new('RGB', (W, H), args.bg)


def load_font(size):
    for f in ('DejaVuSans-Bold.ttf', 'Arial Bold.ttf', 'arialbd.ttf',
              'LiberationSans-Bold.ttf', 'Helvetica-Bold.ttf'):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)          # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def letter(k):                                            # a..z, aa, ab, ...
    s = ''
    k += 1
    while k:
        k, rem = divmod(k - 1, 26)
        s = chr(97 + rem) + s
    return s


draw = ImageDraw.Draw(sheet)
fsize = max(10, int(round(args.letter_size * C)))
font = load_font(fsize)
pad = max(3, int(round(0.04 * C)))
stroke = max(1, fsize // 12)
mapping = []
for k, (name, tile) in enumerate(tiles):
    r, c = divmod(k, cols)
    x0, y0 = c * (C + G), r * (C + G)
    lab = args.letter_fmt.format(letter(k))
    mapping.append((lab, name, tile is not None))
    if tile is None:
        continue                                          # leave the slot empty
    if tile.width != C:
        tile = tile.resize((C, C), Image.LANCZOS)
    sheet.paste(tile, (x0, y0))
    if not args.no_letters:
        draw.text((x0 + pad, y0 + pad), lab, font=font, fill='white',
                  stroke_width=stroke, stroke_fill='black')

# letter -> geometry table, so a geometry grid can use the same order
map_file = Path(args.out).with_suffix('.letters.txt')
with open(map_file, 'w') as fh:
    for lab, name, ok in mapping:
        fh.write(f'{lab}\t{name}{"" if ok else chr(9) + "(missing)"}\n')
print('Panels: ' + ', '.join(f'{lab} {name}' for lab, name, _ in mapping))
print(f'Letter map written to {map_file}')

kw = {'dpi': (args.dpi, args.dpi)}
if args.out.lower().endswith(('.jpg', '.jpeg')):
    kw['quality'] = 95
sheet.save(args.out, **kw)
print(f'Wrote {args.out}: {W} x {H} px, {len(found)}/{len(tiles)} images, '
      f'{rows} x {cols} grid, tile {C} px')

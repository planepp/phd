#!/usr/bin/env python
"""
Put one STM image per geometry into a single JPEG grid (like a subplot).

For each selected geometry, looks in geomXX/afm/<subdir>/ for a PNG matching
--pattern and places it in the grid, labelled with the geometry name.
Missing images leave a labelled empty cell.

Usage (from the folder containing geom01, geom02, ...):
  python geom_sheet.py                                   # defaults below
  python geom_sheet.py --pattern '*z3.4*.png' --cols 6 --cell 250
  python geom_sheet.py --nums 2,3,4,19
"""
import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DEFAULT_NUMS = '2 3 4 5 6 7 8 9 10 13 14 15 19 20 21 22 23 24'

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, default=Path('.'),
               help='Folder containing the geomXX directories.')
p.add_argument('--nums', nargs='+', default=[DEFAULT_NUMS],
               help='Geometry numbers, e.g. "2 3 19" or "2,3,19".')
p.add_argument('--subdir', default='afm/images/stm',
               help='Image folder inside each geomXX.')
p.add_argument('--pattern', default='*z3.4*.png',
               help='Glob for the image to use in each folder.')
p.add_argument('--cols', type=int, default=0,
               help='Columns in the grid (0 = automatic, roughly square).')
p.add_argument('--cell', type=int, default=250, help='Thumbnail size in px.')
p.add_argument('--out', default='geom_sheet_ch_z3.4.jpg')
args = p.parse_args()

nums = [int(x) for s in args.nums for x in s.replace(',', ' ').split()]
if not nums:
    raise SystemExit('No geometry numbers given')

# Find one image per geometry
entries = []
for n in nums:
    name = f'geom{n:02d}'
    files = sorted((args.root / name / args.subdir).glob(args.pattern))
    if not files:
        print(f'{name}: no file matching {args.pattern}')
    elif len(files) > 1:
        print(f'{name}: {len(files)} matches, using {files[0].name}')
    entries.append((name, files[0] if files else None))

n_found = sum(f is not None for _, f in entries)
if n_found == 0:
    raise SystemExit(f'No images matching {args.pattern} in any geomXX/{args.subdir}')

cols = args.cols or math.ceil(math.sqrt(len(entries)))
rows = math.ceil(len(entries) / cols)
C, L, G = args.cell, 26, 6               # cell size, label height, gap
W = G + cols * (C + G)
H = G + rows * (C + L + G)

sheet = Image.new('RGB', (W, H), (60, 62, 66))
draw = ImageDraw.Draw(sheet)
try:
    font = ImageFont.truetype('DejaVuSans.ttf', 16)
except OSError:
    font = ImageFont.load_default()

for k, (name, f) in enumerate(entries):
    r, c = divmod(k, cols)
    x0 = G + c * (C + G)
    y0 = G + r * (C + L + G)
    draw.text((x0 + C // 2, y0 + L // 2), name, fill=(230, 230, 230), font=font, anchor='mm')
    box = [x0, y0 + L, x0 + C - 1, y0 + L + C - 1]

    im = None
    if f is not None:
        try:
            im = Image.open(f).convert('RGB')
        except OSError:
            print(f'{name}: could not read {f.name}')
    if im is None:
        draw.rectangle(box, outline=(110, 110, 110))
        draw.text((x0 + C // 2, y0 + L + C // 2), 'missing', fill=(150, 150, 150),
                  font=font, anchor='mm')
        continue

    im.thumbnail((C, C))
    sheet.paste(im, (x0 + (C - im.width) // 2, y0 + L + (C - im.height) // 2))

sheet.save(args.out, quality=90)
print(f'Wrote {args.out}: {W} x {H} px, {n_found}/{len(entries)} images, {rows} x {cols} grid')

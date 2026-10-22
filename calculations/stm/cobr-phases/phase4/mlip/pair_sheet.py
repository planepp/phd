#!/usr/bin/env python
"""
Combine STM tiles and structure tiles into one figure: for every geometry
the STM image and the structure sit next to each other, with ONE shared
panel letter.

Step 1 - save the individual (unlettered) tiles with the two existing runs:
  python paper_sheet.py --side 0.7 --tiles-dir tiles_stm
  python paper_sheet.py --raw --subdir blender --pattern geom_top.png --tiles-dir tiles_geo

Step 2 - build the paired figure:
  python pair_sheet.py                       # 4 pairs per row
  python pair_sheet.py --cols 2 --cell 500   # tall, full-page layout
  python pair_sheet.py --headers             # "STM" / "Model" above the top row
"""
import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DEFAULT_NUMS = '2 3 4 5 6 7 8 9 10 13 14 15 19 20 21 22 23 24'

p = argparse.ArgumentParser()
p.add_argument('--stm', type=Path, default=Path('tiles_stm'))
p.add_argument('--geo', type=Path, default=Path('tiles_geo'))
p.add_argument('--nums', nargs='+', default=[DEFAULT_NUMS],
               help='Geometries in panel order (same as for the other grids).')
p.add_argument('--geo-first', action='store_true', help='Structure left, STM right.')
p.add_argument('--cols', type=int, default=4, help='Pairs per row.')
p.add_argument('--cell', type=int, default=400, help='Tile size in px.')
p.add_argument('--gap', type=int, default=4, help='Gap inside a pair.')
p.add_argument('--pair-gap', type=int, default=24, help='Gap between pairs.')
p.add_argument('--bg', default='white')
p.add_argument('--frame', default='#b0b0b0',
               help='Thin frame around structure tiles ("none" to disable).')
p.add_argument('--letter-fmt', default='({})')
p.add_argument('--letter-size', type=float, default=0.11)
p.add_argument('--headers', action='store_true',
               help='Write "STM" and "Model" above the first row.')
p.add_argument('--header-text', nargs=2, default=['STM', 'Model'])
p.add_argument('--out', default='pairs_stm_structure.png')
p.add_argument('--dpi', type=int, default=600)
args = p.parse_args()


def load_font(size):
    for f in ('DejaVuSans-Bold.ttf', 'Arial Bold.ttf', 'arialbd.ttf',
              'LiberationSans-Bold.ttf', 'Helvetica-Bold.ttf'):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def letter(k):
    s, k = '', k + 1
    while k:
        k, rem = divmod(k - 1, 26)
        s = chr(97 + rem) + s
    return s


def load_tile(folder, name):
    f = folder / f'{name}.png'
    if not f.exists():
        print(f'{name}: no {f}')
        return None
    im = Image.open(f).convert('RGBA')
    flat = Image.new('RGB', im.size, args.bg)
    flat.paste(im, mask=im.split()[3])
    return flat.resize((args.cell, args.cell), Image.LANCZOS)


nums = [int(x) for s in args.nums for x in s.replace(',', ' ').split()]
names = [f'geom{n:02d}' for n in nums]

C, g, G = args.cell, args.gap, args.pair_gap
pair_w = 2 * C + g
cols = min(args.cols, len(names))
rows = math.ceil(len(names) / cols)
head_font = load_font(max(12, int(0.09 * C)))
head_h = int(0.16 * C) if args.headers else 0
W = cols * pair_w + (cols - 1) * G
H = head_h + rows * C + (rows - 1) * G
sheet = Image.new('RGB', (W, H), args.bg)
draw = ImageDraw.Draw(sheet)

font = load_font(max(10, int(round(args.letter_size * C))))
pad = max(3, int(round(0.04 * C)))
stroke = max(1, int(args.letter_size * C) // 12)

if args.headers:
    left, right = args.header_text if not args.geo_first else args.header_text[::-1]
    for c in range(cols):
        x0 = c * (pair_w + G)
        for txt, xc in ((left, x0 + C // 2), (right, x0 + C + g + C // 2)):
            draw.text((xc, head_h // 2), txt, fill='black', font=head_font, anchor='mm')

lines = []
for k, name in enumerate(names):
    r, c = divmod(k, cols)
    x0 = c * (pair_w + G)
    y0 = head_h + r * (C + G)
    stm, geo = load_tile(args.stm, name), load_tile(args.geo, name)
    first, second = (geo, stm) if args.geo_first else (stm, geo)
    for tile, x in ((first, x0), (second, x0 + C + g)):
        if tile is not None:
            sheet.paste(tile, (x, y0))
            if tile is geo and args.frame.lower() != 'none':
                draw.rectangle([x, y0, x + C - 1, y0 + C - 1], outline=args.frame)
    lab = args.letter_fmt.format(letter(k))
    lines.append(f'{lab}\t{name}')
    draw.text((x0 + pad, y0 + pad), lab, font=font, fill='white',
              stroke_width=stroke, stroke_fill='black')

Path(args.out).with_suffix('.letters.txt').write_text('\n'.join(lines) + '\n')
sheet.save(args.out, dpi=(args.dpi, args.dpi))
print('Panels: ' + ', '.join(l.replace('\t', ' ') for l in lines))
print(f'Wrote {args.out}: {W} x {H} px, {rows} x {cols} pairs')

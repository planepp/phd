#!/usr/bin/env python
"""
Put all bias-pair STM images into one lo x hi contact sheet (a single JPEG).

Rows = lo bias (top = most negative), columns = hi bias. Each cell is a
thumbnail of the PNG for that pair; empty cells mean no image.

Usage:
  python contact_sheet.py                         # defaults below
  python contact_sheet.py --pattern 'stm_current_iso1e-10_V*_s0.5.png' --cell 110
"""
import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

NAME_RE = re.compile(r'_V(-?\d+(?:\.\d+)?)-(-?\d+(?:\.\d+)?)_s')

p = argparse.ArgumentParser()
p.add_argument('--dir', type=Path, default=Path('images/stm'))
p.add_argument('--pattern', default='stm_current_iso1e-09_V*_s1.0.png')
p.add_argument('--cell', type=int, default=110, help='Thumbnail size in px.')
p.add_argument('--out', default='contact_sheet9.jpg')
args = p.parse_args()

items = {}
for f in args.dir.glob(args.pattern):
    m = NAME_RE.search(f.stem)
    if m:
        items[(float(m.group(1)), float(m.group(2)))] = f
if not items:
    raise SystemExit(f'No files matching {args.pattern} in {args.dir}')

los = sorted({lo for lo, _ in items})
his = sorted({hi for _, hi in items})
C, M = args.cell, 44                      # cell size, label margin
W, H = M + len(his) * C, M + len(los) * C
sheet = Image.new('RGB', (W, H), (60, 62, 66))
draw = ImageDraw.Draw(sheet)
try:
    font = ImageFont.truetype('DejaVuSans.ttf', 13)
except OSError:
    font = ImageFont.load_default()

for j, hi in enumerate(his):
    draw.text((M + j * C + C // 2, M // 2), f'{hi:+.1f}', fill=(230, 230, 230), font=font, anchor='mm')
for i, lo in enumerate(los):
    draw.text((M // 2, M + i * C + C // 2), f'{lo:+.1f}', fill=(230, 230, 230), font=font, anchor='mm')
draw.text((4, 4), 'lo \\ hi', fill=(170, 170, 170), font=font)

for k, ((lo, hi), f) in enumerate(sorted(items.items()), 1):
    try:
        im = Image.open(f).convert('RGB')
    except OSError:
        print(f'Could not read {f.name}, skipping')
        continue
    im.thumbnail((C - 4, C - 4))
    i, j = los.index(lo), his.index(hi)
    x = M + j * C + (C - im.width) // 2
    y = M + i * C + (C - im.height) // 2
    sheet.paste(im, (x, y))
    if lo * hi < 0:                      # mark sums with a thin outline
        draw.rectangle([M + j * C + 1, M + i * C + 1, M + (j + 1) * C - 2, M + (i + 1) * C - 2],
                       outline=(120, 170, 220))
    if k % 100 == 0:
        print(f'{k}/{len(items)}')

sheet.save(args.out, quality=88)
print(f'Wrote {args.out}: {W} x {H} px, {len(items)} images '
      f'(blue outline = sum, lo < 0 < hi)')

#!/usr/bin/env python
"""
Put all bias-pair STM / BRSTM images into one lo x hi contact sheet (JPEG).

Rows = lo bias (top = most negative), columns = hi bias. Each cell is a
thumbnail of the PNG for that pair; empty cells mean no image. Blue outline
= sum image (lo < 0 < hi).

Takes the same image options as brstm_img.py (--modes, --plot-type, --zs,
--iso-vals, --weights) and works out the file names from them, so it can be
called with exactly the arguments that made the images. Options it doesn't
know (e.g. --biases, --repeat) are ignored. One sheet is written per
mode x height/iso-value x weight.

Usage:
  python contact_sheet.py --modes stm --plot-type current --iso-vals 1e-9 --weights 1
  python contact_sheet.py --modes brstm --plot-type height --zs 3.2
  python contact_sheet.py --modes stm --plot-type current --iso-vals 1e-9 --min-range 0.2
  python contact_sheet.py --modes stm --plot-type current --iso-vals 1e-9 --bias-range -2.0 2.0 0.2

--bias-range and --min-range mean exactly the same as in make_images.sh, and
make_images.sh passes them on, so its sheet always shows what it made.
  python contact_sheet.py --dir images/stm --pattern 'stm_current_iso1e-10_V*_s0.5.png'
"""
import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

NAME_RE = re.compile(r'_V(-?\d+(?:\.\d+)?)-(-?\d+(?:\.\d+)?)_s')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Same meaning as in brstm_img.py
    p.add_argument('--wd', type=Path, default=Path.cwd())
    p.add_argument('--modes', nargs='+', default=['stm'])
    p.add_argument('--plot-type', choices=['height', 'current'], default='height')
    p.add_argument('--zs', nargs='+', type=float, default=[3.2])
    p.add_argument('--iso-vals', nargs='+', type=float, default=[1e-7])
    p.add_argument('--weights', nargs='+', type=float, default=[1.0])
    # Contact-sheet only
    p.add_argument('--dir', type=Path, default=None,
                   help='Image directory (only with --pattern; default <wd>/images/<mode>).')
    p.add_argument('--pattern', default=None,
                   help='Glob for the PNGs; overrides the name built from the options above.')
    p.add_argument('--out', default=None,
                   help='Output file (default <wd>/images/contact_<image name>.jpg).')
    p.add_argument('--cell', type=int, default=110, help='Thumbnail size in px.')
    p.add_argument('--bias-range', nargs=3, type=float, default=None, metavar=('LO', 'HI', 'STEP'),
                   help='Only include pairs on this bias grid, e.g. -2.0 2.0 0.2 '
                        '(default: every image found).')
    p.add_argument('--min-range', type=float, default=None,
                   help='Only include pairs with hi - lo >= this (e.g. 0.2 skips the 0.1 V ranges).')
    args, _ = p.parse_known_args()   # ignore brstm_img.py-only options
    return args


DEFAULT_RANGE = (-2.0, 2.0, 0.1)   # same default as make_images.sh


def _grid(bias_range):
    lo, hi, st = bias_range
    return [round(lo + k * st, 6) for k in range(round((hi - lo) / st) + 1)]


def _keep(lo, hi, args):
    if args.min_range and hi - lo < args.min_range - 1e-6:
        return False
    if args.bias_range:
        grid = _grid(args.bias_range)
        if not all(any(abs(v - g) < 1e-6 for g in grid) for v in (lo, hi)):
            return False
    return True


def _expected(args):
    """Number of pairs the filters allow, or None if there's no grid to count."""
    if not args.bias_range:
        return None
    g = _grid(args.bias_range)
    return sum(1 for i, a in enumerate(g) for b in g[i + 1:] if _keep(a, b, args))


def _suffix(args):
    """Tag added to the sheet name when filtering, so it doesn't overwrite the full sheet."""
    s = ''
    if args.bias_range and tuple(args.bias_range) != DEFAULT_RANGE:
        lo, hi, st = args.bias_range
        s += f'_V{lo:g}to{hi:g}_step{st:g}'
    if args.min_range:
        s += f'_min{args.min_range:g}'
    return s


def sheets(args):
    """Yield (image_dir, glob_pattern, out_path) for every sheet to make.

    File names must match brstm_img.py, which writes str(float) of each value
    (so --iso-vals 1e-9 -> 'iso1e-09', --weights 1 -> 's1.0').
    """
    if args.pattern:
        d = args.dir or args.wd / 'images' / 'stm'
        out = args.out or args.wd / 'images' / f"contact_{args.pattern.replace('*', '').replace('.png', '')}{_suffix(args)}.jpg"
        yield d, args.pattern, Path(out)
        return
    kind = 'height_z' if args.plot_type == 'height' else 'current_iso'
    vals = args.zs if args.plot_type == 'height' else args.iso_vals
    for mode in args.modes:
        if mode not in ('stm', 'brstm'):
            continue                      # AFM images aren't bias pairs
        for v in vals:
            for w in args.weights:
                stem = f'{mode}_{kind}{v}_V*_s{w}'
                out = args.out or args.wd / 'images' / f"contact_{stem.replace('_V*', '')}{_suffix(args)}.jpg"
                yield args.wd / 'images' / mode, stem + '.png', Path(out)


def make_sheet(img_dir, pattern, out, C, args):
    items = {}
    for f in img_dir.glob(pattern):
        m = NAME_RE.search(f.stem)
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
            if _keep(lo, hi, args):
                items[(lo, hi)] = f
    if not items:
        print(f'No files matching {pattern} in {img_dir}, no sheet written')
        return

    los = sorted({lo for lo, _ in items})
    his = sorted({hi for _, hi in items})
    M = 44                                # label margin
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

    n_ok = 0
    for k, ((lo, hi), f) in enumerate(sorted(items.items()), 1):
        try:
            im = Image.open(f).convert('RGB')
        except OSError:
            print(f'Could not read {f.name}, skipping')
            continue
        im.thumbnail((C - 4, C - 4))
        i, j = los.index(lo), his.index(hi)
        sheet.paste(im, (M + j * C + (C - im.width) // 2, M + i * C + (C - im.height) // 2))
        if lo * hi < 0:                   # mark sums with a thin outline
            draw.rectangle([M + j * C + 1, M + i * C + 1, M + (j + 1) * C - 2, M + (i + 1) * C - 2],
                           outline=(120, 170, 220))
        n_ok += 1
        if k % 100 == 0:
            print(f'  {k}/{len(items)}')

    expected = _expected(args)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)
    count = f'{n_ok} / {expected}' if expected else f'{n_ok}'
    print(f'Wrote {out}: {W} x {H} px, {count} images (blue outline = sum, lo < 0 < hi)')
    if expected and n_ok < expected:
        print(f'  {expected - n_ok} images missing -- resubmit to finish them')


def main():
    args = parse_args()
    for img_dir, pattern, out in sheets(args):
        make_sheet(img_dir, pattern, out, args.cell, args)


if __name__ == '__main__':
    main()

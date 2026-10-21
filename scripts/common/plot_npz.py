import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import sys

from ase.data.colors import jmol_colors
from ase.data import covalent_radii

from scipy.interpolate import RegularGridInterpolator


# --------------------
# args
# --------------------
parser = argparse.ArgumentParser()

parser.add_argument("npzfile", type=str, nargs="?", default=None)
parser.add_argument("mode", type=int, nargs="?", default=None)
parser.add_argument("--modes", type=int, nargs='+', default=None)
parser.add_argument("--molecule", action="store_true")
parser.add_argument("--interpolate", action="store_true")
parser.add_argument("--savepng", action="store_true")
parser.add_argument("--index_offset", type=int, default=0)
parser.add_argument(
    "--weights",
    type=float,
    nargs='+',
    default=None,
    help=(
        "Weighted sum of normalized images. Give one weight per mode, "
        "in the order: mode, then each of --modes. Each image is first "
        "normalized to max |intensity| = 1, then multiplied by its "
        "weight and summed. Example: 12 --modes 15 --weights 0.7 0.3"
    )
)
parser.add_argument(
    "--gauss_center",
    type=float,
    default=None,
    help=(
        "Gaussian-weighted sum: center frequency in cm^-1. Use together "
        "with --gauss_sigma. All modes within --gauss_cutoff sigmas of the "
        "center are selected automatically (no mode number needed), each "
        "image keeps its absolute intensity (no normalization) and is "
        "weighted by w_i = exp(-(f_i - center)^2 / (2 sigma^2)). "
        "Example: data.npz --gauss_center 1580 --gauss_sigma 10"
    )
)
parser.add_argument(
    "--gauss_sigma",
    type=float,
    default=None,
    help="Gaussian-weighted sum: width sigma in cm^-1 (use with --gauss_center)."
)
parser.add_argument(
    "--gauss_cutoff",
    type=float,
    default=3.0,
    help="Gaussian-weighted sum: include modes within this many sigmas of the center (default 3)."
)
parser.add_argument(
    "--subplot",
    action="store_true",
    help=(
        "Instead of summing --modes into a single combined image, "
        "plot mode and each of --modes side by side in one row of "
        "subplots, sharing a single color scale and colorbar "
        "(mirrors plotvibs.py's row layout)."
    )
)

args = parser.parse_args()

# --------------------
# npzfile: ask if missing
# --------------------
if args.npzfile is None:
    print("No npzfile provided. Usage: plot_npz.py <npzfile> <mode> [options]\n"
          "   or: plot_npz.py <npzfile> --gauss_center <cm-1> --gauss_sigma <cm-1>")
    sys.exit(1)

# --------------------
# load data
# --------------------
data = np.load(args.npzfile)
print("Model:", data['model'])
print("Mode indices available:", data['mode_indices'])

freqs = data["frequencies"]
mode_indices = data["mode_indices"]
positions = data["atom_pos"]
numbers = data["atomic_numbers"]

# --------------------
# gaussian mode: select modes automatically from center frequency
# --------------------
gaussian = args.gauss_center is not None or args.gauss_sigma is not None

if gaussian:
    if args.gauss_center is None or args.gauss_sigma is None:
        print("Gaussian weighting needs both --gauss_center and --gauss_sigma.")
        sys.exit(1)
    if args.gauss_sigma <= 0:
        print("--gauss_sigma must be positive.")
        sys.exit(1)
    if args.mode is not None or args.modes or args.weights is not None or args.subplot:
        print("With --gauss_center/--gauss_sigma, don't pass a mode, --modes, "
              "--weights or --subplot: modes are selected automatically.")
        sys.exit(1)

    dist = np.abs(freqs - args.gauss_center)
    selected = np.where(dist <= args.gauss_cutoff * args.gauss_sigma)[0]
    if selected.size == 0:
        print(f"No modes within {args.gauss_cutoff:g} sigma "
              f"({args.gauss_cutoff * args.gauss_sigma:g} cm-1) of "
              f"{args.gauss_center:g} cm-1. Try a larger --gauss_sigma.")
        sys.exit(1)

    # closest mode first: its grid is the reference grid for the sum
    selected = selected[np.argsort(dist[selected])]
    sel_modes = [int(mode_indices[i]) for i in selected]
    args.mode, args.modes = sel_modes[0], sel_modes[1:]

    args.weights = []
    print(f"Gaussian weights (center {args.gauss_center:g} cm-1, "
          f"sigma {args.gauss_sigma:g} cm-1, cutoff {args.gauss_cutoff:g} sigma):")
    for i, m in zip(selected, sel_modes):
        w = float(np.exp(-(freqs[i] - args.gauss_center) ** 2 / (2 * args.gauss_sigma ** 2)))
        args.weights.append(w)
        print(f"  mode {m + args.index_offset}: {freqs[i]:.1f} cm-1  ->  weight {w:.4f}")

# --------------------
# mode: check missing / invalid
# --------------------
if args.mode is None or args.mode not in mode_indices:
    print("No valid mode provided. Please choose a mode from the list above, "
          "or use --gauss_center/--gauss_sigma.")
    sys.exit(1)

all_modes = [args.mode] + (args.modes or [])

# --------------------
# weights: check count
# --------------------
if args.weights is not None and len(args.weights) != len(all_modes):
    print(
        f"--weights needs one value per mode ({len(all_modes)} modes: "
        f"{all_modes}), but got {len(args.weights)}."
    )
    sys.exit(1)


def normalize(z):
    """Scale an image so its maximum absolute intensity is 1."""
    zmax = np.abs(z).max()
    return z / zmax if zmax > 0 else z


def get_grid(m):
    return (
        data[f"x_pos_{m:03d}"],
        data[f"y_pos_{m:03d}"],
        data[f"spectrum_{m:03d}"]
    )


def freq_of(m):
    return freqs[np.where(mode_indices == m)[0][0]]


def prepare_image(m):
    """
    Load mode m's grid and apply the same transformations as the
    original single/sum plot: model-specific transpose, and padding
    for a singular axis so imshow never gets a zero-range extent.

    Returns z (ready to be transposed again by imshow's z.T call),
    plus the xy extent.
    """
    x, y, z = get_grid(m)
    z = z.copy()

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    if data['model'] == 'simple model':
        z = z.T  # rotate x,y axis with Mariana model (double transposing)

    if ymin == ymax:
        ymin, ymax = ymin - 0.5, ymax + 0.5
        z = np.tile(z[:, 0:1], (1, 2))
    elif xmin == xmax:
        xmin, xmax = xmin - 0.5, xmax + 0.5
        z = np.tile(z[0:1, :], (2, 1))

    return z, (xmin, xmax, ymin, ymax)


def overlay_molecule(ax):
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        c=[jmol_colors[n] for n in numbers],
        s=[covalent_radii[n] * 100 for n in numbers],
        edgecolors='k',
        linewidths=0.4,
        zorder=3
    )


offset = args.index_offset
mode_str = "+".join(str(m + offset) for m in all_modes)
freq_str = ", ".join(f"{freq_of(m):.1f}" for m in all_modes)

if args.weights is not None:
    if gaussian:
        weight_str = f"gauss_{args.gauss_center:g}_sigma{args.gauss_sigma:g}"
    else:
        weight_str = ":".join(f"{w:g}" for w in args.weights)
    if args.subplot:
        print("Note: --weights is ignored with --subplot (panels are plotted separately).")


# ============================================================================
# Row-of-subplots mode: one panel per mode, each with its own color scale
# ============================================================================

if args.subplot:

    n_modes = len(all_modes)

    images = [prepare_image(m) for m in all_modes]

    fig, axes = plt.subplots(
        1, n_modes,
        figsize=(5.0 * n_modes, 5.0),
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes.ravel()

    for ax, m, (z, (xmin, xmax, ymin, ymax)) in zip(axes, all_modes, images):
        ax.imshow(
            z.T,
            cmap="viridis",
            origin="lower",
            extent=[xmin, xmax, ymin, ymax],
            aspect="equal",
            interpolation="bilinear" if args.interpolate else "nearest",
            # no shared vmin/vmax: each panel auto-scales to its own data
        )

        if args.molecule:
            overlay_molecule(ax)

        ax.set_title(f"Mode {m + offset} ({freq_of(m):.1f} cm$^{{-1}}$)")
        ax.set_xlabel("x [Å]")

    axes[0].set_ylabel("y [Å]")

    if args.savepng:
        outdir = Path("images")
        outdir.mkdir(exist_ok=True, parents=True)

        outfile = outdir / f"TERS_modes_{mode_str}_row.png"
        # No bbox_inches="tight" here, to preserve the constructed
        # n x 1 row layout when saving (same reasoning as plotvibs.py)
        fig.savefig(outfile, dpi=300, pad_inches=0.2)
        print(f"Saved to {outfile}")

    plt.show()
    sys.exit(0)


# ============================================================================
# Default mode: sum --modes into a single combined image
# ============================================================================

x, y, z = get_grid(args.mode)
z = z.copy()

weighted = args.weights is not None
# manual --weights: normalize each image first
# gaussian: keep absolute intensities, weights applied on top
prep = (lambda a: a) if gaussian else normalize
if weighted:
    z = args.weights[0] * prep(z)

for i, m in enumerate(args.modes or [], start=1):
    xm, ym, zm = get_grid(m)

    if weighted:
        # normalize on the mode's own full grid (before interpolation),
        # so the peak used for scaling isn't affected by grid mismatch
        zm = args.weights[i] * prep(zm)

    interp = RegularGridInterpolator(
        (xm, ym),
        zm,
        method="linear",
        bounds_error=False,
        fill_value=0.0
    )

    X, Y = np.meshgrid(x, y, indexing="ij")
    z += interp((X, Y))


# --------------------
# extent
# --------------------
xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()


# --------------------
# plot
# --------------------
fig, ax = plt.subplots(figsize=(7, 7))
if data['model'] == 'simple model':
    z = z.T  # rotate x,y axis with Mariana model (double transposing)

# for plotting 1D arrays: pad the singular axis so imshow doesn't get a zero-range extent
if ymin == ymax:
    ymin, ymax = ymin - 0.5, ymax + 0.5
    z = np.tile(z[:, 0:1], (1, 2))
elif xmin == xmax:
    xmin, xmax = xmin - 0.5, xmax + 0.5
    z = np.tile(z[0:1, :], (2, 1))

im = ax.imshow(
    z.T,
    cmap="viridis",
    origin="lower",
    extent=[xmin, xmax, ymin, ymax],
    aspect="equal",
    interpolation="bilinear" if args.interpolate else "nearest"
)
cbar = fig.colorbar(im, ax=ax)
if gaussian:
    cbar.set_label("Gaussian-weighted intensity")
elif weighted:
    cbar.set_label("weighted normalized intensity")


# --------------------
# molecule overlay
# --------------------
if args.molecule:
    overlay_molecule(ax)


# --------------------
# labels
# --------------------
title = f"TERS image modes {mode_str} ({freq_str} cm⁻¹)"
if gaussian:
    title = (f"TERS image, Gaussian at {args.gauss_center:g} cm⁻¹, "
             f"σ = {args.gauss_sigma:g} cm⁻¹\n"
             f"{len(all_modes)} modes, absolute intensities")
elif weighted:
    title += f"\nweights {weight_str} (normalized)"
ax.set_title(title)
ax.set_xlabel("x [Å]")
ax.set_ylabel("y [Å]")


plt.tight_layout()


# --------------------
# save
# --------------------
if args.savepng:
    outdir = Path("images")
    outdir.mkdir(exist_ok=True, parents=True)

    if gaussian:
        outfile = outdir / f"TERS_{weight_str}.png"
    else:
        suffix = f"_w{weight_str.replace(':', '-')}" if weighted else ""
        outfile = outdir / f"TERS_modes_{mode_str}{suffix}.png"
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"Saved to {outfile}")


plt.show()

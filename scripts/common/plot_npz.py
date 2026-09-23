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
    print("No npzfile provided. Usage: plot_npz.py <npzfile> <mode> [options]")
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
# mode: check missing / invalid
# --------------------
if args.mode is None or args.mode not in mode_indices:
    print("No valid mode provided. Please choose a mode from the list above.")
    sys.exit(1)

all_modes = [args.mode] + (args.modes or [])


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

for m in (args.modes or []):
    xm, ym, zm = get_grid(m)

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
fig.colorbar(im, ax=ax)


# --------------------
# molecule overlay
# --------------------
if args.molecule:
    overlay_molecule(ax)


# --------------------
# labels
# --------------------
ax.set_title(f"TERS image modes {mode_str} ({freq_str} cm⁻¹)")
ax.set_xlabel("x [Å]")
ax.set_ylabel("y [Å]")


plt.tight_layout()


# --------------------
# save
# --------------------
if args.savepng:
    outdir = Path("images")
    outdir.mkdir(exist_ok=True, parents=True)

    outfile = outdir / f"TERS_modes_{mode_str}.png"
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"Saved to {outfile}")


plt.show()

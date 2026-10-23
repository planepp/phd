#!/usr/bin/env python3
"""
Plot out-of-plane vibrational modes as 2D maps.

Atoms are shown at their xy positions and colored by the out-of-plane (z)
component of their displacement vector for the chosen mode(s).

Usage:
    python viewoutofplane.py modes.xyz
        # list available modes

    python viewoutofplane.py modes.xyz 12
        # plot mode 12

    python viewoutofplane.py modes.xyz 12 15 18
        # plot modes 12, 15, and 18 in one figure with a shared colorbar

    python viewoutofplane.py modes.xyz 12 15 18 --atoms-size 3000
        # larger atoms

    python viewoutofplane.py modes.xyz 12 15 18 --zscale -0.3 0.3
        # fixed shared color scale across all modes

    python viewoutofplane.py modes.xyz 12 15 18 --savepng
        # save combined figure as PNG

Assumes each frame in the xyz file has a header line containing "frequency",
and atom lines formatted as:

    symbol  x  y  z  dx  dy  dz

(position followed by displacement vector).

If your file has a different column layout, adjust `parse_atom_line` below.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from ase.data import covalent_radii, atomic_numbers, chemical_symbols


# --- reading multi-mode xyz files -------------------------------------------

def read_blocks(path):
    """Split a multi-frame xyz file into (header, block_lines) tuples."""
    lines = Path(path).read_text().splitlines()

    blocks = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if line.isdigit():
            natoms = int(line)

            if i + 1 + natoms < len(lines):
                header = lines[i + 1]
                block = lines[i + 2:i + 2 + natoms]
                blocks.append((header, block))

            i += 2 + natoms

        else:
            i += 1

    return blocks


def parse_atom_line(line):
    """
    Parse:

        symbol x y z dx dy dz

    and return:

        (symbol, position, displacement)

    `symbol` may be a chemical symbol (e.g. "Fe") or an atomic number
    (e.g. "26").
    """
    parts = line.split()

    raw_symbol = parts[0]

    if raw_symbol.isdigit():
        symbol = chemical_symbols[int(raw_symbol)]
    else:
        symbol = raw_symbol

    x, y, z, dx, dy, dz = (
        float(v) for v in parts[1:7]
    )

    return (
        symbol,
        np.array([x, y, z]),
        np.array([dx, dy, dz]),
    )


def get_modes(xyz_file):
    """Read all vibrational modes from a multi-frame XYZ file."""
    modes = []

    for header, block in read_blocks(xyz_file):

        if "frequency" not in header.lower():
            continue

        # Extract the first floating-point number from the header.
        floats = re.findall(
            r"[-+]?\d*\.\d+|\d+",
            header,
        )

        if not floats:
            continue

        freq = float(floats[0])

        symbols = []
        positions = []
        displacements = []

        for line in block:
            sym, pos, disp = parse_atom_line(line)

            symbols.append(sym)
            positions.append(pos)
            displacements.append(disp)

        modes.append({
            "freq": freq,
            "symbols": symbols,
            "positions": np.array(positions),
            "displacements": np.array(displacements),
        })

    return modes


# --- plotting helpers -------------------------------------------------------

def get_plot_mask(positions, xlim=None, ylim=None):
    """
    Return a boolean mask selecting atoms inside the requested xy limits.
    """
    mask = np.ones(
        len(positions),
        dtype=bool,
    )

    if xlim is not None:
        mask &= (
            (positions[:, 0] > xlim[0])
            & (positions[:, 0] < xlim[1])
        )

    if ylim is not None:
        mask &= (
            (positions[:, 1] > ylim[0])
            & (positions[:, 1] < ylim[1])
        )

    return mask


def make_shared_norm(modes, zscale=None):
    """
    Create one color normalization shared by all selected modes.

    If zscale is supplied, use it directly.

    Otherwise use a symmetric range around zero based on the largest
    absolute dz found across ALL selected modes.
    """
    if zscale is not None:
        vmin, vmax = zscale

        if vmin >= vmax:
            raise ValueError(
                f"Invalid --zscale range: "
                f"vmin ({vmin}) must be smaller than vmax ({vmax})."
            )

        return Normalize(
            vmin=vmin,
            vmax=vmax,
        )

    max_abs = max(
        np.max(np.abs(mode["displacements"][:, 2]))
        for mode in modes
    )

    # Avoid Normalize(0, 0) for modes with zero z displacement.
    if max_abs == 0:
        max_abs = 1.0

    return Normalize(
        vmin=-max_abs,
        vmax=max_abs,
    )


def plot_vibration_mode(
    mode,
    norm,
    fig,
    ax,
    xlim=None,
    ylim=None,
    atoms_size=2000,
    fontsize=18,
    cmap_name="RdBu",
):
    """
    Plot one vibrational mode into an existing Axes.

    `norm` is shared between all modes so identical colors correspond
    to identical dz values.
    """
    positions = mode["positions"]

    dz = mode["displacements"][:, 2]

    numbers = np.array([
        atomic_numbers[symbol]
        for symbol in mode["symbols"]
    ])

    mask = get_plot_mask(
        positions,
        xlim=xlim,
        ylim=ylim,
    )

    cmap = plt.get_cmap(cmap_name)

    colors = cmap(
        norm(dz[mask])
    )

    ax.scatter(
        positions[mask, 0],
        positions[mask, 1],
        color=colors,
        s=atoms_size * covalent_radii[numbers[mask]],
        edgecolors="k",
        linewidths=0.5,
    )

    ax.set_aspect("equal")

    ax.set_xlabel(
        r"x ($\AA$)",
        fontsize=fontsize,
    )

    ax.set_ylabel(
        r"y ($\AA$)",
        fontsize=fontsize,
    )

    ax.tick_params(
        labelsize=fontsize,
    )

    ax.xaxis.set_major_locator(
        plt.MaxNLocator(5)
    )

    ax.yaxis.set_major_locator(
        plt.MaxNLocator(5)
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
    )

    ax.set_title(
        f"Mode {mode['index']}: "
        f"{mode['freq']:.2f} cm$^{{-1}}$",
        fontsize=fontsize,
    )

    return fig, ax


def plot_vibration_modes(
    modes,
    indices,
    zscale=None,
    xlim=None,
    ylim=None,
    atoms_size=2000,
    fontsize=18,
    cmap_name="RdBu",
):
    """
    Plot one or more vibrational modes in a single figure.

    All subplots use exactly the same Normalize object, so colors are
    directly comparable between modes.

    Parameters
    ----------
    modes : list
        Output of get_modes().

    indices : list of int
        Mode indices to plot.

    zscale : tuple/list or None
        Explicit (vmin, vmax) color scale. If None, determine one
        symmetric scale from the largest |dz| across all selected modes.

    Returns
    -------
    fig, axes
    """

    # ------------------------------------------------------------------
    # Select modes
    # ------------------------------------------------------------------

    selected_modes = []

    for idx in indices:
        mode = modes[idx].copy()

        # Store the original mode index for the subplot title.
        mode["index"] = idx

        selected_modes.append(mode)

    # ------------------------------------------------------------------
    # One shared normalization for ALL selected modes
    # ------------------------------------------------------------------

    norm = make_shared_norm(
        selected_modes,
        zscale=zscale,
    )

    n_modes = len(selected_modes)

    # ------------------------------------------------------------------
    # Determine subplot layout
    # ------------------------------------------------------------------

    if n_modes == 1:
        ncols = 1
    elif n_modes == 2:
        ncols = 2
    elif n_modes <= 4:
        ncols = n_modes
    else:
        # For many modes, use at most 3 columns.
        ncols = 3

    nrows = int(
        np.ceil(n_modes / ncols)
    )

    # ------------------------------------------------------------------
    # Create figure
    #
    # IMPORTANT:
    # Use constrained_layout=True rather than tight_layout().
    # tight_layout() is incompatible with the extra colorbar Axes.
    # ------------------------------------------------------------------

    # Always use one column: N rows × 1 column
    n_modes = len(selected_modes)
    nrows = 1
    ncols = n_modes

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.5, 6.0 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    axes = axes.ravel()

    # ------------------------------------------------------------------
    # Plot each selected mode
    # ------------------------------------------------------------------

    for ax, mode in zip(
        axes,
        selected_modes,
    ):
        plot_vibration_mode(
            mode,
            norm=norm,
            fig=fig,
            ax=ax,
            xlim=xlim,
            ylim=ylim,
            atoms_size=atoms_size,
            fontsize=fontsize,
            cmap_name=cmap_name,
        )

    # ------------------------------------------------------------------
    # Hide unused subplot axes
    # ------------------------------------------------------------------

    for ax in axes[n_modes:]:
        ax.set_visible(False)

    # ------------------------------------------------------------------
    # Shared colorbar
    # ------------------------------------------------------------------

    cmap = plt.get_cmap(cmap_name)

    sm = cm.ScalarMappable(
        norm=norm,
        cmap=cmap,
    )

    sm.set_array([])

    cbar = fig.colorbar(
    sm,
    ax=axes.tolist(),
    orientation="vertical",
    fraction=0.015,
    pad=0.03,
    shrink=0.4,
    aspect=30,
    )

    cbar.ax.set_ylabel(
        r"$\delta z$ ($\AA$)",
        fontsize=fontsize,
    )

    cbar.ax.tick_params(
        labelsize=fontsize,
    )

    cbar.set_ticks(
        np.linspace(
            norm.vmin,
            norm.vmax,
            5,
        )
    )

    cbar.ax.yaxis.set_major_formatter(
        "{x:.2f}"
    )

    return fig, axes


# --- CLI --------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Plot the out-of-plane displacement map of one or more "
            "vibrational modes. Multiple modes are rendered in one "
            "figure with a shared colorbar."
        )
    )

    parser.add_argument(
        "xyz",
        help="Input multi-mode XYZ file",
    )

    parser.add_argument(
        "idx",
        type=int,
        nargs="*",
        help=(
            "One or more target mode indices. "
            "If omitted, only the available modes are listed."
        ),
    )

    parser.add_argument(
        "--atoms-size",
        type=float,
        default=2000,
        help=(
            "Marker size scale factor "
            "(default: 2000)"
        ),
    )

    parser.add_argument(
        "--zscale",
        type=float,
        nargs=2,
        default=None,
        metavar=("VMIN", "VMAX"),
        help=(
            "Fixed shared colorbar range in Angstrom, "
            "e.g. --zscale -0.3 0.3. If omitted, "
            "auto-scales symmetrically using the largest "
            "|dz| across all selected modes."
        ),
    )

    parser.add_argument(
        "--savepng",
        action="store_true",
        help="Save the figure as a PNG",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Read modes
    # ------------------------------------------------------------------

    modes = get_modes(args.xyz)

    print(
        f"Found {len(modes)} vibrational modes "
        f"in {args.xyz}:"
    )

    for i, mode in enumerate(modes):
        print(
            f"  {i:3d}: "
            f"{mode['freq']:.2f} cm-1"
        )

    # ------------------------------------------------------------------
    # No indices -> list modes and exit
    # ------------------------------------------------------------------

    if not args.idx:
        print(
            "\nNo index specified — exiting "
            "after listing available modes."
        )
        sys.exit(0)

    # ------------------------------------------------------------------
    # Validate requested indices
    # ------------------------------------------------------------------

    invalid = [
        idx
        for idx in args.idx
        if not (0 <= idx < len(modes))
    ]

    if invalid:
        print(
            f"Invalid mode index/indices: {invalid}. "
            f"Must be between 0 and {len(modes) - 1}."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Remove duplicates while preserving order
    # ------------------------------------------------------------------

    indices = list(
        dict.fromkeys(args.idx)
    )

    print("\nSelected modes:")

    for idx in indices:
        print(
            f"  {idx:3d}: "
            f"{modes[idx]['freq']:.3f} cm-1"
        )

    # ------------------------------------------------------------------
    # Report shared color scale
    # ------------------------------------------------------------------

    if args.zscale is not None:

        print(
            f"\nUsing fixed shared z-scale: "
            f"[{args.zscale[0]:.3f}, "
            f"{args.zscale[1]:.3f}] Å"
        )

    else:

        selected_max = max(
            np.max(
                np.abs(
                    modes[idx]["displacements"][:, 2]
                )
            )
            for idx in indices
        )

        if selected_max == 0:
            selected_max = 1.0

        print(
            f"\nUsing shared auto z-scale: "
            f"[-{selected_max:.3f}, "
            f"{selected_max:.3f}] Å"
        )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    fig, axes = plot_vibration_modes(
        modes,
        indices,
        zscale=args.zscale,
        atoms_size=args.atoms_size,
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    if args.savepng:

        if len(indices) == 1:
            out_png = (
                f"{Path(args.xyz).stem}"
                f"_mode{indices[0]}.png"
            )
        else:
            mode_string = "-".join(
                map(str, indices)
            )

            out_png = (
                f"{Path(args.xyz).stem}"
                f"_mode{mode_string}.png"
            )

        fig.savefig(
            out_png,
            dpi=300,
            pad_inches=0.2,
        )

        print(f"Saved {out_png}")

    # ------------------------------------------------------------------
    # Show
    # ------------------------------------------------------------------

    plt.show()

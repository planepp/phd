#!/usr/bin/env python3
"""
Plot out-of-plane vibrational modes as 2D xy maps.

Atoms are plotted at their xy positions and colored by the out-of-plane
(z) component of their displacement vector.

Multiple modes can be plotted together in one figure with:
    - one vertical column of subplots
    - one shared colorbar
    - one shared color scale across all selected modes

Ag atoms are ignored completely.

Usage
-----
List available modes:
    python plot_vibration.py modes.xyz

Plot one mode:
    python plot_vibration.py modes.xyz 12

Plot several modes:
    python plot_vibration.py modes.xyz 12 15 18

Save as PNG:
    python plot_vibration.py modes.xyz 12 15 18 --savepng

Use a fixed color scale:
    python plot_vibration.py modes.xyz 12 15 18 --zscale -0.3 0.3

Change atom size:
    python plot_vibration.py modes.xyz 12 15 --atoms-size 3000


Input XYZ format
----------------
The file is assumed to contain multiple XYZ frames.

Each frame has a header containing "frequency", for example:
    123.45 frequency ...

Each atom line is assumed to contain:
    symbol  x  y  z  dx  dy  dz

For example:
    C   -2.2782  -0.0563  -0.0113  -0.0160   0.0339  -0.0019

The first three numbers after the element are the atomic coordinates,
and the last three are the displacement vector.
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


# ============================================================================
# Reading multi-mode XYZ files
# ============================================================================

def read_blocks(path):
    """
    Split a multi-frame XYZ file into (header, block_lines) tuples.
    """
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

        symbol, position, displacement

    The symbol can be either a chemical symbol such as C, H, Ag,
    or an atomic number such as 6, 1, 47.
    """

    parts = line.split()

    raw_symbol = parts[0]

    # Convert atomic number to chemical symbol if necessary
    if raw_symbol.isdigit():
        symbol = chemical_symbols[int(raw_symbol)]
    else:
        symbol = raw_symbol

    x, y, z, dx, dy, dz = (
        float(v) for v in parts[1:7]
    )

    position = np.array([x, y, z])
    displacement = np.array([dx, dy, dz])

    return symbol, position, displacement


def get_modes(xyz_file):
    """
    Read all vibrational modes from the XYZ file.

    Returns a list of dictionaries containing:
        freq
        symbols
        positions
        displacements
    """

    modes = []

    for header, block in read_blocks(xyz_file):

        # Only consider frames whose header contains "frequency"
        if "frequency" not in header.lower():
            continue

        # Extract numbers from the header
        floats = re.findall(
            r"[-+]?\d*\.\d+|\d+",
            header
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


# ============================================================================
# Shared color normalization
# ============================================================================

def make_shared_norm(modes, zscale=None):
    """
    Create one color normalization shared by all selected modes.

    Ag atoms are ignored.

    If zscale is supplied:
        --zscale VMIN VMAX

    then those values are used directly.

    Otherwise the color scale is symmetric around zero and is determined
    from the largest absolute dz among all non-Ag atoms in all selected modes.
    """

    # User-specified scale
    if zscale is not None:
        return Normalize(
            vmin=zscale[0],
            vmax=zscale[1]
        )

    max_abs = 0.0

    for mode in modes:

        # Ignore Ag atoms
        not_ag = np.array([
            symbol != "Ag"
            for symbol in mode["symbols"]
        ])

        dz = mode["displacements"][not_ag, 2]

        if len(dz) > 0:
            max_abs = max(
                max_abs,
                np.abs(dz).max()
            )

    # Avoid an invalid Normalize if all dz happen to be zero
    if max_abs == 0:
        max_abs = 1.0

    return Normalize(
        vmin=-max_abs,
        vmax=max_abs
    )


# ============================================================================
# Plot one mode
# ============================================================================

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
    Plot one vibrational mode into an existing matplotlib axis.

    Ag atoms are completely ignored.
    """

    positions = mode["positions"]
    dz = mode["displacements"][:, 2]

    # ------------------------------------------------------------------------
    # Ignore Ag atoms
    # ------------------------------------------------------------------------

    not_ag = np.array([
        symbol != "Ag"
        for symbol in mode["symbols"]
    ])

    # ------------------------------------------------------------------------
    # x/y limits
    # ------------------------------------------------------------------------

    if xlim is not None:
        idx = (
            (positions[:, 0] > xlim[0]) &
            (positions[:, 0] < xlim[1])
        )
    else:
        idx = np.ones(
            len(positions),
            dtype=bool
        )

    if ylim is not None:
        idy = (
            (positions[:, 1] > ylim[0]) &
            (positions[:, 1] < ylim[1])
        )
    else:
        idy = np.ones(
            len(positions),
            dtype=bool
        )

    # Final selection:
    #   non-Ag AND inside x limits AND inside y limits
    selection = not_ag & idx & idy

    # ------------------------------------------------------------------------
    # Colormap
    # ------------------------------------------------------------------------

    cmap = plt.get_cmap(cmap_name)

    colors = cmap(
        norm(dz[selection])
    )

    # Atomic numbers for marker sizes
    numbers = np.array([
        atomic_numbers[symbol]
        for symbol in mode["symbols"]
    ])

    # ------------------------------------------------------------------------
    # Scatter plot
    # ------------------------------------------------------------------------

    ax.scatter(
        positions[selection, 0],
        positions[selection, 1],
        color=colors,
        s=atoms_size * covalent_radii[numbers[selection]],
        edgecolors="k",
        linewidths=0.5,
    )

    # ------------------------------------------------------------------------
    # Axis formatting
    # ------------------------------------------------------------------------

    ax.set_aspect("equal")

    ax.set_xlabel(
        r"x ($\AA$)",
        fontsize=fontsize
    )

    ax.set_ylabel(
        r"y ($\AA$)",
        fontsize=fontsize
    )

    ax.tick_params(
        labelsize=fontsize
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
        ha="right"
    )

    ax.set_title(
        f"{mode['freq']:.2f} cm$^{{-1}}$",
        fontsize=fontsize
    )

    # Apply limits if supplied
    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    return ax


# ============================================================================
# Plot multiple modes
# ============================================================================

def plot_vibration_modes(
    modes,
    indices,
    zscale=None,
    atoms_size=2000,
    fontsize=18,
    cmap_name="RdBu",
    xlim=None,
    ylim=None,
):
    """
    Plot several vibrational modes in one vertical figure.

    Layout:
        mode 1
        mode 2
        mode 3
        mode 4
        ...

    All modes share one colorbar and one color normalization.
    """

    selected_modes = [
        modes[i]
        for i in indices
    ]

    n_modes = len(selected_modes)

    # ------------------------------------------------------------------------
    # Shared color normalization
    # ------------------------------------------------------------------------

    norm = make_shared_norm(
        selected_modes,
        zscale=zscale
    )

    # ------------------------------------------------------------------------
    # One vertical column
    # ------------------------------------------------------------------------

    ncols = n_modes
    nrows = 1

    # Height is scaled with number of modes.
    #
    # This is intentionally a tall figure because the requested layout
    # is always n x 1.
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(8.0, 5.5 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    axes = axes.ravel()

    # ------------------------------------------------------------------------
    # Plot each mode
    # ------------------------------------------------------------------------

    for ax, mode in zip(axes, selected_modes):

        plot_vibration_mode(
            mode=mode,
            norm=norm,
            fig=fig,
            ax=ax,
            xlim=xlim,
            ylim=ylim,
            atoms_size=atoms_size,
            fontsize=fontsize,
            cmap_name=cmap_name,
        )

    # ------------------------------------------------------------------------
    # One shared colorbar
    # ------------------------------------------------------------------------

    cmap = plt.get_cmap(cmap_name)

    sm = cm.ScalarMappable(
        norm=norm,
        cmap=cmap
    )

    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=axes.tolist(),
        orientation="vertical",

        # Small colorbar
        fraction=0.012,
        pad=0.03,
        shrink=0.60,
        aspect=35,
    )

    cbar.ax.set_ylabel(
        r"$\delta z$ ($\AA$)",
        fontsize=fontsize
    )

    cbar.ax.tick_params(
        labelsize=fontsize
    )

    cbar.set_ticks(
        np.linspace(
            norm.vmin,
            norm.vmax,
            5
        )
    )

    cbar.ax.yaxis.set_major_formatter(
        "{x:.2f}"
    )

    return fig, axes


# ============================================================================
# Command-line interface
# ============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Plot out-of-plane displacement maps for one or more "
            "vibrational modes."
        )
    )

    # ------------------------------------------------------------------------
    # Input XYZ file
    # ------------------------------------------------------------------------

    parser.add_argument(
        "xyz",
        help="Input multi-mode XYZ file"
    )

    # ------------------------------------------------------------------------
    # Mode indices
    #
    # nargs="*" allows:
    #
    #   modes.xyz
    #   modes.xyz 12
    #   modes.xyz 12 15 18
    # ------------------------------------------------------------------------

    parser.add_argument(
        "idx",
        type=int,
        nargs="*",
        help="One or more mode indices"
    )

    # ------------------------------------------------------------------------
    # Atom size
    # ------------------------------------------------------------------------

    parser.add_argument(
        "--atoms-size",
        type=float,
        default=2000,
        help=(
            "Marker size scale factor "
            "(default: 2000)"
        )
    )

    # ------------------------------------------------------------------------
    # Color scale
    # ------------------------------------------------------------------------

    parser.add_argument(
        "--zscale",
        type=float,
        nargs=2,
        default=None,
        metavar=("VMIN", "VMAX"),
        help=(
            "Fixed colorbar range in Angstrom, "
            "e.g. --zscale -0.3 0.3. "
            "If omitted, the scale is symmetric around zero "
            "and determined from the largest |dz| across all "
            "selected non-Ag atoms."
        )
    )

    # ------------------------------------------------------------------------
    # Save PNG
    # ------------------------------------------------------------------------

    parser.add_argument(
        "--savepng",
        action="store_true",
        help="Save the figure as a PNG"
    )

    args = parser.parse_args()

    # =========================================================================
    # Read modes
    # =========================================================================

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

    # =========================================================================
    # No mode specified -> only list modes
    # =========================================================================

    if not args.idx:

        print(
            "\nNo mode index specified — "
            "exiting after listing available modes."
        )

        sys.exit(0)

    # =========================================================================
    # Validate mode indices
    # =========================================================================

    for idx in args.idx:

        if not (0 <= idx < len(modes)):

            print(
                f"Invalid mode index {idx}. "
                f"Must be between 0 and {len(modes) - 1}."
            )

            sys.exit(1)

    # =========================================================================
    # Remove duplicate indices while preserving order
    # =========================================================================

    selected_indices = list(
        dict.fromkeys(args.idx)
    )

    # =========================================================================
    # Print selected modes
    # =========================================================================

    print("\nSelected modes:")

    for idx in selected_indices:

        print(
            f"  {idx:3d}: "
            f"{modes[idx]['freq']:.3f} cm-1"
        )

    # =========================================================================
    # Create figure
    # =========================================================================

    fig, axes = plot_vibration_modes(
        modes=modes,
        indices=selected_indices,
        zscale=args.zscale,
        atoms_size=args.atoms_size,
    )

    # =========================================================================
    # Save
    # =========================================================================

    if args.savepng:

        mode_string = "-".join(
            str(i)
            for i in selected_indices
        )

        out_png = (
            f"{Path(args.xyz).stem}"
            f"_mode{mode_string}.png"
        )

        # Do NOT use bbox_inches="tight" here.
        #
        # Keeping the original figure bounding box preserves the
        # carefully constructed n x 1 layout when saving.
        fig.savefig(
            out_png,
            dpi=300,
            pad_inches=0.2,
        )

        print(
            f"\nSaved {out_png}"
        )

    # =========================================================================
    # Display
    # =========================================================================

    plt.show()


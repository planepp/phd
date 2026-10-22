#!/usr/bin/env python3
"""
Plot the projected DOS of a subset of atoms (e.g. an adsorbed molecule on a
substrate) from a VASP DOSCAR file computed with ISPIN=2 and LORBIT=11.

Usage examples
--------------
# Select molecule atoms by 1-based index range (as in POSCAR/VASP numbering)
python plot_molecule_dos.py --doscar DOSCAR --poscar POSCAR --atoms 37-42

# Select molecule atoms by element symbol (all atoms of these species)
python plot_molecule_dos.py --doscar DOSCAR --poscar POSCAR --elements C,O,H

# Just list species/index ranges found in POSCAR (helps you pick --atoms)
python plot_molecule_dos.py --poscar POSCAR --list

Notes
-----
- LORBIT=11 site-projected DOS columns (per spin channel doubled by ISPIN=2):
    s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2
  This script sums these into total s / p / d / total(s+p+d) per spin.
- Energies are shifted so E_fermi = 0 (E_fermi is read from DOSCAR header).
- Spin-down DOS is plotted as negative values (standard convention).
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def read_poscar_species(poscar_path):
    """Return list of (element, count) and cumulative 1-based index ranges."""
    with open(poscar_path) as f:
        lines = f.readlines()

    # Line 6 (index 5) may be element symbols (VASP5+) or counts directly (VASP4)
    tokens6 = lines[5].split()
    if all(t.replace('.', '', 1).isdigit() for t in tokens6):
        # VASP4 style: no element line, counts on line 6
        elements = [f"species{i+1}" for i in range(len(tokens6))]
        counts = [int(t) for t in tokens6]
    else:
        elements = tokens6
        counts = [int(t) for t in lines[6].split()]

    ranges = []
    start = 1
    for el, c in zip(elements, counts):
        ranges.append((el, start, start + c - 1))
        start += c
    return ranges


def parse_atom_selector(sel, poscar_ranges):
    """Parse '--atoms 12-20,25,30' style string into a set of 1-based indices."""
    indices = set()
    for part in sel.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-')
            indices.update(range(int(a), int(b) + 1))
        else:
            indices.add(int(part))
    return indices


def parse_element_selector(sel, poscar_ranges):
    wanted = {e.strip() for e in sel.split(',')}
    indices = set()
    for el, start, end in poscar_ranges:
        if el in wanted:
            indices.update(range(start, end + 1))
    return indices


def parse_doscar(doscar_path):
    with open(doscar_path) as f:
        lines = f.readlines()

    header0 = lines[0].split()
    nions = int(header0[0])

    # Line index 5 (6th line): Emax Emin NEDOS Efermi weight
    l6 = lines[5].split()
    nedos = int(l6[2])
    efermi = float(l6[3])

    pos = 6
    # --- total DOS block ---
    total_block = lines[pos:pos + nedos]
    pos += nedos
    total_data = np.array([list(map(float, l.split())) for l in total_block])
    # columns: E, up, down, int_up, int_down  (ISPIN=2, no projections)

    # --- per-ion projected DOS blocks ---
    ion_data = []
    for i in range(nions):
        pos += 1  # skip the per-ion header line (E Emin NEDOS Efermi 1)
        block = lines[pos:pos + nedos]
        pos += nedos
        arr = np.array([list(map(float, l.split())) for l in block])
        ion_data.append(arr)

    return {
        "nions": nions,
        "nedos": nedos,
        "efermi": efermi,
        "energy": total_data[:, 0] - efermi,
        "total_up": total_data[:, 1],
        "total_down": total_data[:, 2],
        "ion_data": ion_data,  # list of arrays, columns as in DOSCAR (LORBIT=11)
    }


def mark_top_peaks(ax, energy, y_pos, n, sign=1, color="black", fontsize=7):
    """Find the n tallest local maxima in y_pos (a non-negative DOS array)
    and mark them on ax at (energy, sign*y_pos), annotated with their
    energy value. sign=+1 for spin-up, -1 for spin-down (since spin-down
    is plotted as negative)."""
    if n <= 0 or len(y_pos) == 0:
        return
    peaks, _ = find_peaks(y_pos)
    if len(peaks) == 0:
        return
    heights = y_pos[peaks]
    order = np.argsort(heights)[::-1][:n]
    top = peaks[order]
    for idx in top:
        e = energy[idx]
        val = sign * y_pos[idx]
        marker = "^" if sign > 0 else "v"
        ax.plot(e, val, marker=marker, color=color, ms=5, zorder=5,
                 markeredgecolor="black", markeredgewidth=0.5)
        ax.annotate(f"{e:.2f}", (e, val), textcoords="offset points",
                     xytext=(0, 6 * sign), ha="center", va="bottom" if sign > 0 else "top",
                     fontsize=fontsize, color=color)


def gaussian_smooth(y, energy, sigma_ev):
    """Smooth y(energy) with a Gaussian of given sigma (in eV).

    Converts sigma from eV to grid points using the (uniform) energy
    spacing, then applies scipy's gaussian_filter1d.
    """
    if sigma_ev <= 0:
        return y
    de = abs(energy[1] - energy[0])
    sigma_pts = sigma_ev / de
    return gaussian_filter1d(y, sigma=sigma_pts)


def sum_molecule_pdos(dos, atom_indices):
    """
    Sum LORBIT=11 columns (s,py,pz,px,dxy,dyz,dz2,dxz,dx2-y2), each with
    up/down, over the selected 1-based atom indices.
    Returns dict with s/p/d/total for up and down.
    """
    ncols = dos["ion_data"][0].shape[1]
    # column 0 is energy; remaining should be 18 columns (9 orbitals x 2 spins)
    if ncols != 19:
        print(f"Warning: expected 19 columns for ISPIN2/LORBIT11 per-ion block, "
              f"found {ncols}. Check your DOSCAR settings.", file=sys.stderr)

    n = dos["nedos"]
    s_up = np.zeros(n); s_dn = np.zeros(n)
    p_up = np.zeros(n); p_dn = np.zeros(n)
    d_up = np.zeros(n); d_dn = np.zeros(n)

    for idx in atom_indices:
        arr = dos["ion_data"][idx - 1]  # convert to 0-based
        # columns (1-based counting after energy col=0):
        # 1:s_up 2:s_dn 3:py_up 4:py_dn 5:pz_up 6:pz_dn 7:px_up 8:px_dn
        # 9:dxy_up 10:dxy_dn 11:dyz_up 12:dyz_dn 13:dz2_up 14:dz2_dn
        # 15:dxz_up 16:dxz_dn 17:dx2_up 18:dx2_dn
        s_up += arr[:, 1]; s_dn += arr[:, 2]
        p_up += arr[:, 3] + arr[:, 5] + arr[:, 7]
        p_dn += arr[:, 4] + arr[:, 6] + arr[:, 8]
        d_up += arr[:, 9] + arr[:, 11] + arr[:, 13] + arr[:, 15] + arr[:, 17]
        d_dn += arr[:, 10] + arr[:, 12] + arr[:, 14] + arr[:, 16] + arr[:, 18]

    return {
        "s_up": s_up, "s_dn": s_dn,
        "p_up": p_up, "p_dn": p_dn,
        "d_up": d_up, "d_dn": d_dn,
        "tot_up": s_up + p_up + d_up,
        "tot_dn": s_dn + p_dn + d_dn,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doscar", default="DOSCAR")
    ap.add_argument("--poscar", default="POSCAR")
    ap.add_argument("--atoms", help="1-based index range(s), e.g. '37-42' or '37-40,42'")
    ap.add_argument("--elements", help="comma-separated element symbols, e.g. 'C,O,H'")
    ap.add_argument("--elim", nargs=2, type=float, default=[-6, 6],
                     help="energy window relative to Efermi, default -6 6")
    ap.add_argument("--orbitals", action="store_true",
                     help="plot s/p/d resolved instead of just total")
    ap.add_argument("--list", action="store_true",
                     help="just print species index ranges from POSCAR and exit")
    ap.add_argument("--out", default="molecule_dos.png")
    ap.add_argument("--smooth", type=float, default=0.0,
                     help="Gaussian smoothing sigma in eV (e.g. 0.05). "
                          "0 (default) = no smoothing.")
    ap.add_argument("--peaks", type=int, default=0,
                     help="Mark the N tallest peaks of the molecule DOS for "
                          "each spin channel (up and down), labeled with "
                          "their energy in eV. 0 (default) = off.")
    args = ap.parse_args()

    ranges = read_poscar_species(args.poscar)

    if args.list:
        print("Species and 1-based index ranges in POSCAR:")
        for el, s, e in ranges:
            print(f"  {el:>4s}: {s}-{e}")
        return

    if args.atoms:
        atom_indices = parse_atom_selector(args.atoms, ranges)
    elif args.elements:
        atom_indices = parse_element_selector(args.elements, ranges)
    else:
        ap.error("Provide --atoms or --elements (or use --list to inspect POSCAR first)")

    print(f"Selected {len(atom_indices)} atoms for molecule DOS: "
          f"{sorted(atom_indices)}")

    dos = parse_doscar(args.doscar)

    print(f"DOSCAR reports NIONS = {dos['nions']}; "
          f"POSCAR total atoms = {sum(c[2] - c[1] + 1 for c in ranges)}")
    if dos["nions"] != sum(c[2] - c[1] + 1 for c in ranges):
        print("WARNING: NIONS in DOSCAR does not match POSCAR atom count. "
              "These files are likely from different runs/geometries — "
              "atom indexing will be wrong.", file=sys.stderr)

    ncols = dos["ion_data"][0].shape[1]
    print(f"Detected {ncols} columns per ion block (expected 19 for ISPIN=2, LORBIT=11).")

    # Sanity check independent of atom selection: does summing *every* ion's
    # PDOS stay <= the reported total DOS? If not, the problem is in the
    # DOSCAR/parsing itself, not in which atoms you picked for the molecule.
    all_ions = sum_molecule_pdos(dos, range(1, dos["nions"] + 1))
    diff_up = all_ions["tot_up"] - dos["total_up"]
    diff_dn = all_ions["tot_dn"] - dos["total_down"]
    print(f"Sum of ALL ions vs reported total DOS: "
          f"max excess up = {diff_up.max():.4f}, max excess down = {diff_dn.max():.4f} "
          f"(should be <= 0, i.e. sum-of-ions <= total; small negative values "
          f"are normal, from interstitial charge not assigned to any atom)")
    if diff_up.max() > 1e-3 or diff_dn.max() > 1e-3:
        i = int(np.argmax(diff_up)) if diff_up.max() > diff_dn.max() else int(np.argmax(diff_dn))
        print(f"WARNING: sum over ALL ions exceeds the reported total DOS at "
              f"E = {dos['energy'][i]:+.3f} eV. This means the discrepancy is "
              f"NOT about which atoms you selected as the molecule - something "
              f"is off in how this DOSCAR is being parsed (e.g. an element with "
              f"f-orbital projectors shifting column order, or a non-standard "
              f"LORBIT/ISPIN combination). Share a few lines of one ion block "
              f"plus the total-DOS header line and I can check the exact format.",
              file=sys.stderr)

    mol = sum_molecule_pdos(dos, atom_indices)

    if np.any(mol["tot_up"] > dos["total_up"] + 1e-6) or \
       np.any(mol["tot_dn"] > dos["total_down"] + 1e-6):
        print("WARNING: molecule DOS exceeds total system DOS at some energies. "
              "This is not physical — double-check --atoms/--elements selection "
              "and that DOSCAR/POSCAR correspond to the same structure.",
              file=sys.stderr)

    energy = dos["energy"]
    sm = lambda y: gaussian_smooth(y, energy, args.smooth)  # noqa: E731

    fig, ax = plt.subplots(figsize=(7, 5))

    # reference: total system DOS, lightly shaded
    #ax.fill_between(energy, sm(dos["total_up"]), 0, color="grey", alpha=0.15,
    #                 label="Total DOS (system)")
    #ax.fill_between(energy, -sm(dos["total_down"]), 0, color="grey", alpha=0.15)

    if args.orbitals:
        for key, color, label in [("s", "tab:blue", "s"),
                                   ("p", "tab:orange", "p"),
                                   ("d", "tab:green", "d")]:
            ax.plot(energy, sm(mol[f"{key}_up"]), color=color, label=f"molecule {label}")
            ax.plot(energy, -sm(mol[f"{key}_dn"]), color=color)
    else:
        ax.plot(energy, sm(mol["tot_up"]), color="blue", lw=1.8, label="molecule DOS")
        ax.plot(energy, -sm(mol["tot_dn"]), color="blue", lw=1.8)

    if args.peaks > 0:
        mask = (energy >= args.elim[0]) & (energy <= args.elim[1])
        e_win = energy[mask]
        up_win = sm(mol["tot_up"])[mask]
        dn_win = sm(mol["tot_dn"])[mask]

        mark_top_peaks(ax, e_win, up_win, args.peaks, sign=1, color="blue")
        mark_top_peaks(ax, e_win, dn_win, args.peaks, sign=-1, color="blue")

        def report_peaks(label, y_pos):
            peaks, _ = find_peaks(y_pos)
            if len(peaks) == 0:
                print(f"  {label}: no peaks found")
                return
            heights = y_pos[peaks]
            order = np.argsort(heights)[::-1][:args.peaks]
            top = sorted(peaks[order], key=lambda i: e_win[i])
            for i in top:
                print(f"  {label}: E = {e_win[i]:+.3f} eV, DOS = {y_pos[i]:.3f}")

        print(f"\nTop {args.peaks} molecule DOS peaks per spin channel "
              f"(within elim = {args.elim}):")
        report_peaks("spin-up  ", up_win)
        report_peaks("spin-down", dn_win)

    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8, ls="--", label="$E_F$")
    ax.set_xlim(args.elim)
    ax.set_xlabel("E - E$_F$ (eV)")
    ax.set_ylabel("DOS (states/eV)")
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()
    fig.savefig(args.out, dpi=200)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()

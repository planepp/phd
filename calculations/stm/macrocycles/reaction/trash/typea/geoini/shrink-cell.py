#!/usr/bin/env python3
"""
Shrink the in-plane cell of an adsorbate/Au(111) slab POSCAR and delete the
Au atoms outside it. New cell vectors are whole numbers of Au spacings, so the
slab stays periodic and the adsorbates keep their site relative to the gold.

Give ALL endpoints (initial, final, ...) in one call: they get the identical
cell, origin and Au atoms, so the atom order still matches for NEB.

Usage:
    python shrink_cell.py POSCAR_intial POSCAR_final_NEB_SiBr            # auto size, 10 A gap
    python shrink_cell.py POSCAR_intial POSCAR_final_NEB_SiBr --gap 8 --c 24
    python shrink_cell.py POSCAR_intial POSCAR_final_NEB_SiBr --na 14 --nb 8

Writes <name>_small next to each input. Cartesian POSCARs only.
"""
import argparse
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("files", nargs="+")
p.add_argument("--na", type=int, help="Au spacings along a (default: smallest that satisfies --gap)")
p.add_argument("--nb", type=int, help="Au spacings along b (default: smallest that satisfies --gap)")
p.add_argument("--gap", type=float, default=10.0,
               help="min distance (A) between an adsorbate and its periodic images")
p.add_argument("--c", type=float, help="new c length (vacuum), A; default keep")
p.add_argument("--slab", default="Au", help="slab element (default Au)")
args = p.parse_args()


def read(fname):
    L = open(fname).read().splitlines()
    scale = float(L[1])
    lat = scale * np.array([[float(x) for x in l.split()] for l in L[2:5]])
    species, counts = L[5].split(), list(map(int, L[6].split()))
    sd = L[7].strip().lower().startswith("s")
    ctype = L[8 if sd else 7].strip().lower()
    if not ctype.startswith(("c", "k")):
        raise SystemExit(f"{fname}: only Cartesian coordinates are supported")
    start = 9 if sd else 8
    body = L[start:start + sum(counts)]
    xyz = scale * np.array([[float(x) for x in l.split()[:3]] for l in body])
    rest = [" ".join(l.split()[3:]) for l in body]
    labels = [s for s, n in zip(species, counts) for _ in range(n)]
    block = [b for b, n in enumerate(counts) for _ in range(n)]
    return dict(L=L, lat=lat, species=species, counts=counts, sd=sd,
                xyz=xyz, rest=rest, labels=labels, block=block)


S = [read(f) for f in args.files]
ref = S[0]
lat = ref["lat"]
for s, f in zip(S, args.files):
    if not np.allclose(s["lat"], lat) or s["species"] != ref["species"] or s["counts"] != ref["counts"]:
        raise SystemExit(f"{f}: lattice/species/counts differ from {args.files[0]}")
slab = np.array([l == args.slab for l in ref["labels"]])
for s, f in zip(S[1:], args.files[1:]):
    if not np.allclose(s["xyz"][slab], ref["xyz"][slab], atol=1e-6):
        raise SystemExit(f"{f}: {args.slab} positions differ from {args.files[0]}; cannot crop identically")

# Au spacing and number of spacings in the original cell
au = ref["xyz"][slab]
toplayer = au[au[:, 2] > au[:, 2].max() - 0.8]
d = np.linalg.norm(toplayer[:, None, :2] - toplayer[None, :, :2], axis=-1)
dnn = np.median(np.sort(d, axis=1)[:, 1])
Na0, Nb0 = (int(round(np.linalg.norm(lat[i]) / dnn)) for i in (0, 1))
ua, ub = lat[0] / Na0, lat[1] / Nb0          # one Au spacing along a and b
print(f"Au spacing {dnn:.4f} A; original cell = {Na0} x {Nb0} spacings")

ads = [s["xyz"][~slab] for s in S]


def image_gap(na, nb):
    A, B = na * ua, nb * ub
    g = np.inf
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            if i == j == 0:
                continue
            t = i * A + j * B
            for X in ads:
                g = min(g, np.linalg.norm(X[:, None] - (X[None] + t), axis=-1).min())
    return g


if args.na and args.nb:
    na, nb = args.na, args.nb
else:
    best = None
    for na in range(args.na or 1, (args.na or Na0) + 1):
        for nb in range(args.nb or 1, (args.nb or Nb0) + 1):
            if image_gap(na, nb) >= args.gap and (best is None or na * nb < best[0] * best[1]):
                best = (na, nb)
    if best is None:
        raise SystemExit("no smaller cell satisfies --gap; lower it or give --na/--nb")
    na, nb = best
A, B = na * ua, nb * ub
newlat = np.array([A, B, lat[2] if args.c is None else lat[2] / np.linalg.norm(lat[2]) * args.c])
print(f"new cell = {na} x {nb} spacings  (|a|={np.linalg.norm(A):.3f}, |b|={np.linalg.norm(B):.3f} A), "
      f"adsorbate-image gap {image_gap(na, nb):.2f} A")

# origin: whole Au spacings, chosen to centre all adsorbates of all files
inv_old = np.linalg.inv(lat[:2, :2])
allads = np.vstack(ads)
fc = (0.5 * (allads[:, :2].min(0) + allads[:, :2].max(0))) @ inv_old   # centre in old frac
ka = int(round(fc[0] * Na0 - na / 2))
kb = int(round(fc[1] * Nb0 - nb / 2))
origin = ka * ua + kb * ub
inv_new = np.linalg.inv(newlat)

# keep slab atoms in [-1/2, n-1/2) spacings: boundary sits between Au rows for every stacking layer.
# Each Au may be used through any periodic image of the ORIGINAL cell (window may cross its edge).
au_new = au.copy()
keep_au = np.zeros(len(au), bool)
for i in (-1, 0, 1):
    for j in (-1, 0, 1):
        img = au + i * lat[0] + j * lat[1]
        fs_sp = ((img - origin) @ inv_new)[:, :2] * [na, nb]
        inside = np.all((fs_sp >= -0.5) & (fs_sp < np.array([na, nb]) - 0.5), axis=1) & ~keep_au
        au_new[inside] = img[inside]
        keep_au |= inside
slab_idx = np.where(slab)[0]
keep = ~slab.copy()
keep[slab_idx[keep_au]] = True

layers = np.round(au[keep_au, 2], 0)
for z in np.unique(layers):
    n = (layers == z).sum()
    flag = "" if n == na * nb else f"   <-- expected {na*nb}, check!"
    print(f"  {args.slab} layer z~{z:4.0f}: {n} atoms{flag}")

newcounts = [int(sum(keep[i] for i in range(len(keep)) if ref["block"][i] == b))
             for b in range(len(ref["counts"]))]

for s, f in zip(S, args.files):
    xyz = s["xyz"].copy()
    xyz[slab] = au_new
    xyz -= origin
    fr = xyz @ inv_new
    fa = fr[~slab]
    if (fa[:, :2] < 0).any() or (fa[:, :2] >= 1).any() or (fa[:, 2] >= 1).any():
        raise SystemExit(f"{f}: adsorbate atoms fall outside the new cell; use larger --na/--nb or --c")
    # wrap kept Au into [0,1) so every atom is inside the cell
    xyz_out = xyz.copy()
    fr_au = fr[slab]
    fr_au[:, :2] -= np.floor(fr_au[:, :2])
    xyz_out[slab] = fr_au @ newlat
    L = s["L"]
    hdr = [L[0], " 1.0"] + ["  %22.16f %22.16f %22.16f" % tuple(v) for v in newlat] + \
          [L[5], " " + " ".join(f"{n:4d}" for n in newcounts)] + (["Selective dynamics"] if s["sd"] else []) + ["Cartesian"]
    body = [" %20.16f %20.16f %20.16f   %s" % (*xyz_out[i], s["rest"][i]) for i in range(len(xyz_out)) if keep[i]]
    out = f + "_small"
    open(out, "w").write("\n".join(hdr + body) + "\n")
    print(f"{f}: {len(xyz)} -> {len(body)} atoms  ->  {out}")

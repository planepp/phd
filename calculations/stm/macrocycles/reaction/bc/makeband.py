#!/usr/bin/env python3
"""
Starting band for the type b -> type c NEB: one molecule turns over (twists
~180 deg) about its C-Si bond.

Usage:
    python make_twist_band.py CONTCAR_b CONTCAR_c NIMAGES [outdir] [--min-height 2.2] [--si-follow 1.0]

How the images are built (t = k/(NIMAGES+1)):
  * the twisting molecule (the one bonded to the Si that moves most) is turned
    rigidly about the axis that best maps its initial pose onto its final pose;
    the axis passes through its carbon bonded to the Si. Of the two senses of
    turning, the one that needs less lifting is used. A small linear correction
    makes the last image coincide exactly with the final state;
  * a rigid turn about the C-Si bond pushes the first ring of the molecule into
    the gold, so where any atom would come closer than --min-height (A) to the top
    Au layer, the molecule is lifted rigidly just enough. The Si and the Br on it
    follow by the fraction --si-follow of that lift (1.0 = Si lifts with the
    molecule and keeps its C-Si bond, stretching Si-Au; 0 = Si stays on the gold
    and the C-Si bond stretches instead);
  * everything else (Si, Br on Si, the other molecule, the Au) is interpolated
    linearly.
Molecules are unfolded as whole units, so nothing is dragged across the cell.
Writes outdir/00 ... outdir/<N+1>/POSCAR (Cartesian, selective-dynamics flags kept).
"""
import os
import sys
import numpy as np

args = [a for a in sys.argv[1:] if not a.startswith("--")]
ini_f, fin_f, nimg = args[0], args[1], int(args[2])
outdir = args[3] if len(args) > 3 else "."
hmin, follow = 2.2, 1.0
if "--min-height" in sys.argv:
    hmin = float(sys.argv[sys.argv.index("--min-height") + 1])
if "--si-follow" in sys.argv:
    follow = float(sys.argv[sys.argv.index("--si-follow") + 1])


def read(fname):
    L = open(fname).read().splitlines()
    s = float(L[1])
    lat = s * np.array([[float(x) for x in l.split()] for l in L[2:5]])
    species, counts = L[5].split(), list(map(int, L[6].split()))
    sd = L[7].strip().lower().startswith("s")
    st = 9 if sd else 8
    body = L[st:st + sum(counts)]
    xyz = np.array([[float(x) for x in l.split()[:3]] for l in body])
    xyz = xyz * s if L[st - 1].strip().lower().startswith(("c", "k")) else xyz @ lat
    flags = [" ".join(l.split()[3:6]) for l in body] if sd else [""] * len(body)
    lab = [a for a, n in zip(species, counts) for _ in range(n)]
    return lat, xyz, lab, flags, species, counts, sd


lat, Pi, lab, flags, species, counts, sd = read(ini_f)
lat2, Pf, lab2, *_ = read(fin_f)
if lab != lab2 or not np.allclose(lat, lat2):
    raise SystemExit("endpoints differ in atoms, order or cell")
n = len(lab)
inv = np.linalg.inv(lat)


def mic(v):
    f = v @ inv
    f[..., :2] -= np.round(f[..., :2])
    return f @ lat


def bonded(p, a, b):
    d = np.linalg.norm(mic(p[b] - p[a]))
    pair = {lab[a], lab[b]}
    if pair == {"H"} or "Au" in pair:
        return False
    if "Si" in pair or "Br" in pair:
        return d < 2.4
    return d < 1.75


ads = [i for i in range(n) if lab[i] != "Au"]
au = [i for i in range(n) if lab[i] == "Au"]
si = lab.index("Si")


def unwrap(p, start_pos):
    pos, st = {si: start_pos}, [si]
    while st:
        x = st.pop()
        for y in ads:
            if y not in pos and bonded(p, x, y):
                pos[y] = pos[x] + mic(p[y] - p[x])
                st.append(y)
    for y in ads:
        if y not in pos:
            pos[y] = start_pos + mic(p[y] - p[si])
    out = p.copy()
    for k, v in pos.items():
        out[k] = v
    return out


Pi = unwrap(Pi, Pi[si])
Pf = unwrap(Pf, Pi[si] + mic(Pf[si] - Pi[si]))
Pf[au] = Pi[au] + mic(Pf[au] - Pi[au])
top = Pi[au, 2].max()

nbr = [j for j in ads if j != si and bonded(Pi, si, j)]
carbons = [j for j in nbr if lab[j] == "C"]
br_si = [j for j in nbr if lab[j] == "Br"][0]


def molecule(c0):
    seen, st = {c0}, [c0]
    while st:
        x = st.pop()
        for y in ads:
            if y not in seen and y not in (si, br_si) and bonded(Pi, x, y):
                seen.add(y)
                st.append(y)
    return sorted(seen)


mols = [molecule(c) for c in carbons]
moves = [np.linalg.norm(Pf[m] - Pi[m], axis=1).mean() for m in mols]
k_tw = int(np.argmax(moves))
tw, c_tw = mols[k_tw], carbons[k_tw]
print(f"Si = atom {si+1}; twisting molecule bonded through C{c_tw+1} "
      f"({len(tw)} atoms, mean move {moves[k_tw]:.2f} A; other molecule {moves[1-k_tw]:.2f} A)")

# rotation that maps the twisting molecule onto its final pose
A, B = Pi[tw], Pf[tw]
ac, bc = A.mean(0), B.mean(0)
H = (A - ac).T @ (B - bc)
U, _, Vt = np.linalg.svd(H)
R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
w, v = np.linalg.eig(R)
axis = np.real(v[:, np.argmin(abs(w - 1))])
axis /= np.linalg.norm(axis)
bond = Pi[si] - Pi[c_tw]
print(f"twist {np.degrees(theta):.1f} deg; axis is {np.degrees(np.arccos(abs(axis @ bond) / np.linalg.norm(bond))):.0f} deg "
      f"from the C{c_tw+1}-Si bond, {np.degrees(np.arcsin(abs(axis[2]))):.0f} deg out of the surface plane")
origin = Pi[c_tw]


def rot(X, ang, u=axis, o=origin):
    X = X - o
    return o + X * np.cos(ang) + np.cross(u, X) * np.sin(ang) + np.outer(X @ u, u) * (1 - np.cos(ang))


# two ways to turn over: by +theta or by -(2*pi - theta) about the same axis
# (opposite edge of the molecule goes up first); keep the one that needs less lifting
best = None
for ang in (theta, theta - 2 * np.pi):
    resid = B - rot(A, ang)
    need = max(max(0.0, hmin - ((rot(A, t * ang) + t * resid)[:, 2].min() - top)) for t in np.linspace(0, 1, 41))
    if best is None or need < best[0]:
        best = (need, ang, resid)
need, theta, resid = best
print(f"turning sense chosen: {np.degrees(theta):.0f} deg, max lift needed {need:.2f} A "
      f"(keeps every atom >= {hmin} A above the top Au layer)")

images, lifts = [], []
for k in range(nimg + 2):
    t = k / (nimg + 1)
    P = Pi + t * (Pf - Pi)
    X = rot(A, t * theta) + t * resid
    lift = max(0.0, hmin - (X[:, 2].min() - top)) if 0 < k < nimg + 1 else 0.0
    X[:, 2] += lift
    P[tw] = X
    P[si, 2] += follow * lift
    P[br_si, 2] += follow * lift
    images.append(P)
    lifts.append(lift)

# report
nonb = np.ones((n, n), bool)
for P in (Pi, Pf):
    D = np.linalg.norm(mic(P[:, None] - P[None]), axis=-1)
    nonb &= D > 2.6
np.fill_diagonal(nonb, False)
print(f"\n{'img':>3} {'twist':>6} {'lift':>5} {'Si-C':>5} {'Si-Au':>6} {'lowest atom':>11} {'highest':>8} {'closest non-bonded pair':>25} {'dist':>5}")
for k, P in enumerate(images):
    D = np.where(nonb, np.linalg.norm(mic(P[:, None] - P[None]), axis=-1), 99)
    a, b = np.unravel_index(D.argmin(), D.shape)
    print(f"{k:3d} {np.degrees(k/(nimg+1)*theta):6.0f} {lifts[k]:5.2f} {np.linalg.norm(P[c_tw]-P[si]):5.2f} {np.linalg.norm(mic(P[au]-P[si]),axis=1).min():6.2f} "
          f"{P[tw,2].min()-top:11.2f} {P[tw,2].max()-top:8.2f} {lab[a]+str(a+1)+'-'+lab[b]+str(b+1):>25} {D.min():5.2f}")

for k, P in enumerate(images):
    d = os.path.join(outdir, f"{k:02d}")
    os.makedirs(d, exist_ok=True)
    hdr = [" ".join(species), "   1.0"] + ["  %20.16f %20.16f %20.16f" % tuple(r) for r in lat] + \
          ["  " + "  ".join(species), "  " + "  ".join(str(c) for c in counts)] + \
          (["Selective dynamics"] if sd else []) + ["Cartesian"]
    body = [" %20.16f %20.16f %20.16f   %s" % (*P[i], flags[i]) for i in range(n)]
    open(os.path.join(d, "POSCAR"), "w").write("\n".join(hdr + body) + "\n")
print(f"\nwritten {outdir}/00 ... {outdir}/{nimg+1:02d}")

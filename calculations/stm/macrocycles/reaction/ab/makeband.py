#!/usr/bin/env python3
"""
Starting band for the type a -> type b rotation NEB.

Usage:
    python make_rotation_band.py CONTCAR_initial CONTCAR_final NIMAGES [outdir]

How the images are built (t = k/(NIMAGES+1)):
  * the molecule bonded to the Si that swings (the one that moves most) is turned
    rigidly in-plane about the vertical axis that best maps its initial pose onto
    its final pose (the pivot sits at the Si); a small linear correction makes the
    last image coincide exactly with the final state;
  * the Br on the Si keeps its Si-Br length and moves over the TOP of the Si:
    initial direction -> straight up (halfway) -> final direction;
  * everything else (Si, the other molecule, its Br, the Au) is interpolated linearly.
Atoms are unfolded consistently (molecules as whole units), so nothing is dragged
across the cell. Writes outdir/00 ... outdir/<N+1>/POSCAR (Cartesian, flags kept).
"""
import os
import sys
import numpy as np

ini_f, fin_f, nimg = sys.argv[1], sys.argv[2], int(sys.argv[3])
outdir = sys.argv[4] if len(sys.argv) > 4 else "."


def read(fname):
    L = open(fname).read().splitlines()
    s = float(L[1])
    lat = s * np.array([[float(x) for x in l.split()] for l in L[2:5]])
    species, counts = L[5].split(), list(map(int, L[6].split()))
    sd = L[7].strip().lower().startswith("s")
    st = 9 if sd else 8
    body = L[st:st + sum(counts)]
    xyz = np.array([[float(x) for x in l.split()[:3]] for l in body])
    if not L[st - 1].strip().lower().startswith(("c", "k")):
        xyz = xyz @ lat
    else:
        xyz *= s
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


# --- unwrap the whole adsorbate cluster by walking bonds (initial geometry) ---
ads = [i for i in range(n) if lab[i] != "Au"]
si = lab.index("Si")


def unwrap(p, start_pos):
    pos = {si: start_pos}
    st = [si]
    while st:
        x = st.pop()
        for y in ads:
            if y not in pos and bonded(p, x, y):
                pos[y] = pos[x] + mic(p[y] - p[x])
                st.append(y)
    missing = [y for y in ads if y not in pos]
    for y in missing:                       # free atoms, if any: nearest image to Si
        pos[y] = start_pos + mic(p[y] - p[si])
    out = p.copy()
    for k, v in pos.items():
        out[k] = v
    return out


Pi = unwrap(Pi, Pi[si])
Pf = unwrap(Pf, Pi[si] + mic(Pf[si] - Pi[si]))
au = [i for i in range(n) if lab[i] == "Au"]
Pf[au] = Pi[au] + mic(Pf[au] - Pi[au])

# --- identify the two molecules bonded to the Si, and the Br on the Si ---
nbr = [j for j in ads if j != si and bonded(Pi, si, j)]
carbons = [j for j in nbr if lab[j] == "C"]
br_si = [j for j in nbr if lab[j] == "Br"][0]


def molecule(c0):
    seen, st = {c0}, [c0]
    while st:
        x = st.pop()
        for y in ads:
            if y not in seen and y != si and y != br_si and bonded(Pi, x, y):
                seen.add(y)
                st.append(y)
    return sorted(seen)


mols = [molecule(c) for c in carbons]
moves = [np.linalg.norm(Pf[m] - Pi[m], axis=1).mean() for m in mols]
rot_mol = mols[int(np.argmax(moves))]
print(f"Si = atom {si+1}, Br on Si = atom {br_si+1}")
print(f"swinging molecule: {len(rot_mol)} atoms (incl. its outer Br), mean move {max(moves):.1f} A; "
      f"other molecule mean move {min(moves):.1f} A")

# --- rigid in-plane rotation that maps the swinging molecule onto its final pose ---
A, B = Pi[rot_mol], Pf[rot_mol]
ac, bc = A.mean(0), B.mean(0)
H = (A - ac).T @ (B - bc)
U, _, Vt = np.linalg.svd(H)
R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
theta = np.arctan2(R[1, 0], R[0, 0])
R2 = R[:2, :2]
pivot = np.linalg.solve(np.eye(2) - R2, bc[:2] - R2 @ ac[:2])


def rz(X, t, o):
    X = X.copy()
    xy = X[:, :2] - o
    c, s = np.cos(t), np.sin(t)
    X[:, 0] = o[0] + c * xy[:, 0] - s * xy[:, 1]
    X[:, 1] = o[1] + s * xy[:, 0] + c * xy[:, 1]
    return X


resid = B - rz(A, theta, pivot)
resid[:, 2] = B[:, 2] - A[:, 2]
print(f"rotation {np.degrees(theta):.1f} deg about vertical axis at ({pivot[0]:.2f}, {pivot[1]:.2f}); "
      f"pivot is {np.linalg.norm(pivot - Pi[si][:2]):.2f} A from the Si; "
      f"rigid-fit residual {np.sqrt((resid[:, :2]**2).sum(1).mean()):.2f} A")

# --- Br on Si: over the top ---
rbr_i = Pi[br_si] - Pi[si]
rbr_f = Pf[br_si] - Pf[si]
up = np.array([0.0, 0.0, 1.0])


def slerp(a, b, t):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    om = np.arccos(np.clip(a @ b, -1, 1))
    if om < 1e-6:
        return a
    return (np.sin((1 - t) * om) * a + np.sin(t * om) * b) / np.sin(om)


def br_dir(t):
    return slerp(rbr_i, up, 2 * t) if t <= 0.5 else slerp(up, rbr_f, 2 * t - 1)


# --- build images ---
images = []
for k in range(nimg + 2):
    t = k / (nimg + 1)
    P = Pi + t * (Pf - Pi)                                   # linear for everything ...
    P[rot_mol] = rz(A, t * theta, pivot) + t * resid         # ... except the swinging molecule
    L = (1 - t) * np.linalg.norm(rbr_i) + t * np.linalg.norm(rbr_f)
    P[br_si] = P[si] + L * br_dir(t)                          # ... and the Br over the top
    images.append(P)

# --- report ---
nonb = np.ones((n, n), bool)
for P in (Pi, Pf):
    D = np.linalg.norm(mic(P[:, None] - P[None]), axis=-1)
    nonb &= D > 2.6
np.fill_diagonal(nonb, False)
c_rot = [c for c in carbons if c in rot_mol][0]
top = Pi[au, 2].max()
print(f"\n{'img':>3} {'Si-C(swing)':>11} {'Si-Br':>6} {'Br height':>9} {'closest non-bonded pair':>26} {'dist':>5}")
for k, P in enumerate(images):
    D = np.where(nonb, np.linalg.norm(mic(P[:, None] - P[None]), axis=-1), 99)
    a, b = np.unravel_index(D.argmin(), D.shape)
    print(f"{k:3d} {np.linalg.norm(P[c_rot]-P[si]):11.2f} {np.linalg.norm(P[br_si]-P[si]):6.2f} "
          f"{P[br_si,2]-top:9.2f} {lab[a]+str(a+1)+'-'+lab[b]+str(b+1):>26} {D.min():5.2f}")

# --- write ---
for k, P in enumerate(images):
    d = os.path.join(outdir, f"{k:02d}")
    os.makedirs(d, exist_ok=True)
    hdr = [" ".join(species), "   1.0"] + ["  %20.16f %20.16f %20.16f" % tuple(v) for v in lat] + \
          ["  " + "  ".join(species), "  " + "  ".join(str(c) for c in counts)] + \
          (["Selective dynamics"] if sd else []) + ["Cartesian"]
    body = [" %20.16f %20.16f %20.16f   %s" % (*P[i], flags[i]) for i in range(n)]
    open(os.path.join(d, "POSCAR"), "w").write("\n".join(hdr + body) + "\n")
print(f"\nwritten {outdir}/00 ... {outdir}/{nimg+1:02d}")

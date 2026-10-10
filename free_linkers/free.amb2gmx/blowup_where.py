#!/usr/bin/env python3
"""
blowup_where.py -- where did the run blow up?

When SETTLE or LINCS fails, mdrun writes the whole system at the step before
(step<N>b_n0.pdb) and at the failing step (step<N>c_n0.pdb). In one 4 fs step a
normal atom moves about 0.02 A, so atoms that moved an angstrom or more mark the
spot. This lists them and says what surrounds the worst one.

    python3 blowup_where.py step5620352b_n0.pdb step5620352c_n0.pdb

Needs only numpy.
"""
import sys
import numpy as np

WATER = {"SOL", "HOH", "WAT", "OPC"}
IONS = {"NA", "CL", "NA+", "CL-", "K", "K+", "Na+", "Cl-"}
NEAR = 6.0          # A: shell reported around the worst atom


def read(path):
    box, rows, xyz = None, [], []
    for l in open(path):
        if l.startswith("CRYST1"):
            box = [float(l[6:15]), float(l[15:24]), float(l[24:33]),
                   float(l[33:40]), float(l[40:47]), float(l[47:54])]
        elif l.startswith(("ATOM", "HETATM")):
            rows.append((l[12:16].strip(), l[17:21].strip(), l[21].strip(), l[22:26].strip()))
            xyz.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
    return box, rows, np.array(xyz)


def lattice(a, b, c, al, be, ga):
    al, be, ga = np.radians([al, be, ga])
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    return np.array([[a, 0, 0],
                     [b * np.cos(ga), b * np.sin(ga), 0],
                     [cx, cy, np.sqrt(max(c * c - cx * cx - cy * cy, 0.0))]])


def kind(resn):
    return "water" if resn in WATER else "ion" if resn in IONS else "solute"


if len(sys.argv) != 3:
    sys.exit(__doc__)
box, rows, xb = read(sys.argv[1])
_, rows_c, xc = read(sys.argv[2])
if len(xb) != len(xc):
    sys.exit(f"atom counts differ: {len(xb)} vs {len(xc)}")
V = lattice(*box) if box else None
Vinv = np.linalg.inv(V) if V is not None else None


def min_image(d):
    if V is None:
        return d
    f = d @ Vinv
    return (f - np.round(f)) @ V


disp = np.linalg.norm(min_image(xc - xb), axis=1)
lost = ~np.isfinite(disp)          # NaN at the failing step: what SETTLE/LINCS gave up on
order = np.argsort(np.where(lost, np.inf, disp))[::-1]

print(f"{len(xb)} atoms; box {box[0]/10:.2f} x {box[1]/10:.2f} x {box[2]/10:.2f} nm" if box else f"{len(xb)} atoms")
print(f"median move in this step {np.nanmedian(disp):.3f} A (normal is ~0.02 A at 4 fs)")
if lost.any():
    print(f"{int(lost.sum())} atoms have NaN coordinates at the failing step -- "
          "these are what could not be settled or constrained")
print()

print(f"{'atom':>7}  {'moved/A':>8}  residue")
print("-" * 44)
for i in order[:12]:
    name, resn, ch, resi = rows[i]
    tag = f"{resn}{resi}" + (f" chain {ch}" if ch and kind(resn) == "solute" else "")
    moved = "NaN" if lost[i] else f"{disp[i]:.2f}"
    print(f"{i+1:>7}  {moved:>8}  {tag} {name}  ({kind(resn)})")

big = np.nan_to_num(disp, nan=0.0) > 1.0
kinds = {}
for i in np.where(big)[0]:
    kinds[kind(rows[i][1])] = kinds.get(kind(rows[i][1]), 0) + 1
print(f"\natoms that moved more than 1 A: {int(big.sum())}  " +
      ", ".join(f"{k} {n}" for k, n in sorted(kinds.items())))

w = order[0]
d = np.linalg.norm(min_image(xb - xb[w]), axis=1)
solute = np.array([kind(r[1]) == "solute" for r in rows])
near = np.where((d < NEAR) & solute)[0]
print(f"\naround the worst atom ({w+1}, {rows[w][1]}{rows[w][3]} {rows[w][0]}) in the step before:")
if solute.any():
    j = np.where(solute)[0][np.argmin(d[solute])]
    print(f"  nearest solute atom {d[j]:.2f} A away: {rows[j][1]}{rows[j][3]} {rows[j][0]}"
          + (f" chain {rows[j][2]}" if rows[j][2] else ""))
res = sorted({(rows[i][2], rows[i][1], rows[i][3]) for i in near},
             key=lambda r: (r[0], int(r[2]) if r[2].lstrip("-").isdigit() else 0))
if res:
    print(f"  solute residues within {NEAR:.0f} A: " +
          ", ".join(f"{rn}{ri}" + (f"/{ch}" if ch else "") for ch, rn, ri in res))
else:
    print(f"  no solute within {NEAR:.0f} A -- it started in bulk water")
nw = len({rows[i][3] for i in np.where((d < NEAR) & np.array([kind(r[1]) == 'water' for r in rows]))[0]})
ni = int(((d < NEAR) & np.array([kind(r[1]) == "ion" for r in rows])).sum())
print(f"  waters within {NEAR:.0f} A: {nw}; ions: {ni}")

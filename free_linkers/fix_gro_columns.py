#!/usr/bin/env python3
"""
fix_gro_columns.py -- repair a .gro whose residue or atom numbers ran past 99,999.

    python3 fix_gro_columns.py free_GMX.gro        (keeps free_GMX.gro.bak)

.gro is fixed-column: residue number in columns 1-5, residue name 6-10, atom name
11-15, atom number 16-20, then x y z in 8-character fields. Numbers above 99,999
must wrap round to 0. acpype writes them in full, which pushes the rest of the
line one column to the right, and GROMACS stops with "Something is wrong in the
coordinate formatting". This rewrites every atom line with both numbers wrapped,
keeping residue names, atom names and coordinates exactly as they were.
"""
import shutil
import sys

if len(sys.argv) != 2:
    sys.exit(__doc__)
path = sys.argv[1]
lines = open(path).read().split("\n")
title = lines[0]
natoms = int(lines[1].strip())
atoms = lines[2:2 + natoms]
box = lines[2 + natoms]
if len(atoms) != natoms:
    sys.exit(f"{path} says {natoms} atoms but has {len(atoms)} atom lines")

out, changed, first = [], 0, None
for i, line in enumerate(atoms, start=1):
    l = line.rstrip()
    nfloat = 0
    for tok in reversed(l.split()):
        if "." in tok:
            nfloat += 1
        else:
            break
    if nfloat >= 6:
        ntail = 48                 # x y z and velocities, 8 characters each
    elif nfloat >= 3:
        ntail = 24                 # x y z
    else:
        sys.exit(f"line {i + 2}: no coordinates found: {line!r}")
    tail, head = l[-ntail:], l[:-ntail]

    # a 6-digit residue number puts a digit where the residue name should start
    k = 6 if len(head) > 5 and head[5].isdigit() else 5
    resnr = int(head[:k]) % 100000
    resname = head[k:k + 5].strip()
    atomname = head[k + 5:k + 10].strip()
    new = f"{resnr:5d}{resname:<5s}{atomname:>5s}{i % 100000:5d}{tail}"
    if new != l:
        changed += 1
        if first is None:
            first = (i + 2, l, new)
    out.append(new)

# every line must now have its decimal points where GROMACS looks for them
for n, l in enumerate(out, start=3):
    if l[24] != "." or l[32] != "." or l[40] != ".":
        sys.exit(f"line {n} still misaligned after repair: {l!r}")

if changed == 0:
    print(f"{path}: all {natoms} atom lines already in the right columns; nothing changed")
    sys.exit(0)
shutil.copy(path, path + ".bak")
open(path, "w").write("\n".join([title, lines[1]] + out + [box] + lines[3 + natoms:]))
ln, before, after = first
print(f"{path}: re-aligned {changed} of {natoms} atom lines (backup: {path}.bak)")
print(f"  first repaired line {ln}:")
print(f"    before: {before}")
print(f"    after:  {after}")

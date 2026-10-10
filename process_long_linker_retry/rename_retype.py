#!/usr/bin/env python3
"""
Step 2, simple version: rename and retype the Cys backbone and the two caps
in CYL_capped.ac by atom number, and (optionally) set ff19SB charges on them.

    python3 rename_retype.py CYL_capped.ac CYL_fixed.ac

Atom numbers come from the SMILES order: heavy atoms 1-77 in the order they are
written, then hydrogens 78-164 in the order of the heavy atom they sit on.
Before changing anything the script checks the bonds between these atoms, so a
file numbered differently stops with an error instead of being mislabelled.
Linker atoms (everything after SG) keep their names, GAFF2 types and charges.
"""
import sys

SET_CHARGES = True   # False = keep AM1-BCC charges everywhere (prepgen then rescales)

# number: (new name, ff19SB type, ff19SB charge or None = keep AM1-BCC)
# charges from amino19.lib (CYX), aminont12.lib (ACE), aminoct12.lib (NME)
ATOMS = {
    # ACE cap -- deleted by prepgen
    1:   ("CAY", "CT", -0.3662),
    78:  ("HY1", "HC",  0.1123),
    79:  ("HY2", "HC",  0.1123),
    80:  ("HY3", "HC",  0.1123),
    2:   ("CY",  "C",   0.5972),
    3:   ("OY",  "O",  -0.5679),
    # Cys backbone and side chain up to the sulfur
    4:   ("N",   "N",  -0.4157),
    81:  ("H",   "H",   0.2719),
    5:   ("CA",  "XC",  0.0429),
    82:  ("HA",  "H1",  0.0766),
    6:   ("CB",  "2C",  None),
    83:  ("HB2", "H1",  None),
    84:  ("HB3", "H1",  None),
    7:   ("SG",  "S",   None),
    74:  ("C",   "C",   0.5973),
    75:  ("O",   "O",  -0.5679),
    # NME cap -- deleted by prepgen
    76:  ("NT",  "N",  -0.4157),
    161: ("HNT", "H",   0.2719),
    77:  ("CAT", "CT", -0.1490),
    162: ("HT1", "H1",  0.0976),
    163: ("HT2", "H1",  0.0976),
    164: ("HT3", "H1",  0.0976),
}
CAPS = {1, 78, 79, 80, 2, 3, 76, 161, 77, 162, 163, 164}

# bonds that must exist if the numbering above is right (7-8 = SG to the linker)
BONDS = [(1, 2), (2, 3), (2, 4), (1, 78), (1, 79), (1, 80),
         (4, 81), (4, 5), (5, 82), (5, 6), (6, 83), (6, 84), (6, 7), (7, 8),
         (5, 74), (74, 75), (74, 76), (76, 161), (76, 77),
         (77, 162), (77, 163), (77, 164)]

if len(sys.argv) != 3:
    sys.exit(__doc__)
lines = open(sys.argv[1]).read().splitlines()

# ---- read atoms (antechamber .ac columns: name 13-16, charge 54-63, type 64-73)
atoms = {}                                   # number -> [line index, name, charge, type]
bonds = set()
for k, line in enumerate(lines):
    if line.startswith("ATOM"):
        n = int(line[4:11])
        name, q, t = line[13:17].strip(), float(line[54:64]), line[64:].strip()
        if (q, t) != (float(line.split()[-2]), line.split()[-1]):
            sys.exit("ERROR: unexpected .ac column layout on line %d" % (k + 1))
        atoms[n] = [k, name, q, t]
    elif line.startswith("BOND"):
        f = line.split()
        bonds.add(frozenset((int(f[2]), int(f[3]))))

# ---- check the numbering before touching anything
if len(atoms) != 164:
    sys.exit("ERROR: %d atoms, expected 164 (H87 C49 N7 O19 S2)" % len(atoms))
bad = [b for b in BONDS if frozenset(b) not in bonds]
wrong_el = [n for n, (new, _, _) in ATOMS.items() if atoms[n][3][0].lower() != new[0].lower()]
if bad or wrong_el:
    sys.exit("ERROR: this file is numbered differently.\n  missing bonds: %s\n"
             "  wrong element: %s\nFix the numbers in ATOMS and BONDS." % (bad, wrong_el))

# ---- rename, retype, recharge
print(" no.  old name type    ->  new name type   charge")
for n in sorted(ATOMS, key=lambda n: (n in CAPS, n)):
    new, t, q = ATOMS[n]
    k, old, q_old, t_old = atoms[n]
    if SET_CHARGES and q is not None:
        atoms[n][2] = q
    print("%4d  %-5s %-4s   ->  %-5s    %-4s  %8.4f%s" % (n, old, t_old, new, t, atoms[n][2],
          "   (cap, removed by prepgen)" if n in CAPS else ""))
    atoms[n][1], atoms[n][3] = new, t

if SET_CHARGES:
    # residue must sum to exactly 0: spread the leftover over the side chain
    fixed = {n for n, (_, _, q) in ATOMS.items() if q is not None}
    side = [n for n in atoms if n not in fixed]
    leftover = -sum(atoms[n][2] for n in atoms if n not in CAPS)
    for n in side:
        atoms[n][2] = round(atoms[n][2] + leftover / len(side), 6)
    atoms[6][2] = round(atoms[6][2] - sum(atoms[n][2] for n in atoms if n not in CAPS), 6)
    print("\nleftover %+.4f e spread over %d side-chain atoms (%+.5f e each)"
          % (leftover, len(side), leftover / len(side)))

names = [a[1] for a in atoms.values()]
dup = sorted({x for x in names if names.count(x) > 1})
if dup:
    sys.exit("ERROR: duplicate atom names after renaming: %s" % dup)

# ---- write
for n, (k, name, q, t) in atoms.items():
    l = lines[k]
    lines[k] = l[:13] + "%-4s" % name + l[17:54] + "%10.6f" % q + "%10s" % t
for k, line in enumerate(lines):
    if line.startswith("BOND"):
        f = line.split()
        i, j = int(f[2]), int(f[3])
        lines[k] = "BOND%5d%5d%5d%5s  %5s%5s" % (int(f[1]), i, j, f[4], atoms[i][1], atoms[j][1])
open(sys.argv[2], "w").write("\n".join(lines) + "\n")

q_res = sum(atoms[n][2] for n in atoms if n not in CAPS)
q_cap = sum(atoms[n][2] for n in CAPS)
print("residue charge %+.6f   caps %+.6f   ->  wrote %s" % (q_res, q_cap, sys.argv[2]))

#!/usr/bin/env python3
"""
split_linkers.py -- take the free linkers out of gmx insert-molecules' output and
write them as a PDB tleap reads correctly.

    python3 split_linkers.py fibril_box.pdb start.pdb linkers_noh.pdb

insert-molecules appends the new molecules after the fibril and leaves the
fibril's atoms untouched, so everything past the fibril's atom count is linker.
The fibril itself is NOT taken from start.pdb: it goes through
fix_names_amber.py on its own, because that script renames NME -> MET (GROMACS
cuts NMET to NME) and would turn every linker's N-methyl cap into a methionine.

What this writes:
  - heavy atoms only; tleap rebuilds the hydrogens from CYL.prepin and the
    ACE/NME templates, as Step 10 does for the fibril
  - residues renumbered 1..3N, atoms renumbered 1..
  - one TER after each NME and nowhere else: every ACE-CYL-NME is its own
    molecule, and no TER falls between ACE and CYL -- tleap reads every TER as a
    chain break, and CYL would lose its peptide bond to ACE
"""
import sys

SEQ = ("ACE", "CYL", "NME")
HEAVY = {"ACE": 3, "CYL": 72, "NME": 2}        # 77 heavy atoms per linker


def atom_lines(path):
    return [l for l in open(path) if l.startswith(("ATOM", "HETATM"))]


def is_hydrogen(line):
    # Amber names every hydrogen H..., sometimes behind a digit (1HH3).
    # Decided from the name, not the element column, which GROMACS guesses.
    return line[12:16].strip().lstrip("0123456789").upper().startswith("H")


def residues(lines):
    """group consecutive atom lines into residues by (chain, resSeq, iCode, resName)"""
    out, key = [], None
    for l in lines:
        k = (l[21], l[22:26], l[26], l[17:21].strip())
        if k != key:
            out.append([])
            key = k
        out[-1].append(l)
    return out


if len(sys.argv) != 4:
    sys.exit(__doc__)
fib_pdb, ins_pdb, out_pdb = sys.argv[1:]

n_fib = len(atom_lines(fib_pdb))
allat = atom_lines(ins_pdb)
if len(allat) <= n_fib:
    sys.exit(f"{ins_pdb} has {len(allat)} atoms, the fibril alone {n_fib}: nothing was inserted")

head = allat[n_fib - 1][17:21].strip()
if head in SEQ:
    sys.exit(f"atom {n_fib} of {ins_pdb} is in residue {head}, not the fibril -- "
             "is fibril_box.pdb the file insert-molecules was given with -f?")

res = residues(allat[n_fib:])
if len(res) % 3:
    sys.exit(f"{len(res)} linker residues is not a multiple of 3")
names = [r[0][17:21].strip() for r in res]
for i in range(0, len(names), 3):
    if tuple(names[i:i + 3]) != SEQ:
        sys.exit(f"linker {i // 3 + 1} reads {names[i:i + 3]}, expected {list(SEQ)}")

n_link = len(res) // 3
out, serial = [], 0
for j, r in enumerate(res):
    rn = names[j]
    heavy = [l for l in r if not is_hydrogen(l)]
    if len(heavy) != HEAVY[rn]:
        sys.exit(f"linker {j // 3 + 1} {rn}: {len(heavy)} heavy atoms, expected {HEAVY[rn]}")
    for l in heavy:
        serial += 1
        l = l.rstrip("\n").ljust(54)
        out.append(f"{l[:6]}{serial % 100000:5d}{l[11:22]}{(j + 1) % 10000:4d}{l[26:]}\n")
    if rn == "NME":
        out.append("TER\n")
out.append("END\n")
open(out_pdb, "w").writelines(out)

h = sum(1 for l in allat[n_fib:] if is_hydrogen(l))
print(f"fibril: {n_fib} atoms skipped (taken from fib_noh.pdb instead)")
print(f"linkers: {n_link} x ACE-CYL-NME, {serial} heavy atoms "
      f"({serial // n_link} each), {h} hydrogens dropped for tleap to rebuild")
print(f"wrote {out_pdb}: {n_link} TER records, one after each NME")

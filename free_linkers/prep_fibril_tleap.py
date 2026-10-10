#!/usr/bin/env python3
"""
prep_fibril_tleap.py -- turn a GROMACS-written fibril PDB into one tleap reads.

    python3 prep_fibril_tleap.py fibril_box.pdb fib_noh.pdb

Does what fix_names_amber.py (Step 10) does, without PyMOL, plus two things the
free-linker system needs:
  - deletes Met217 (the cloning Met: its light S-CH3 crashed a 4 fs run)
  - writes a TER at every chain break, found from the chain ID OR a drop in the
    residue number, so it still works if the chain IDs were lost on the way

and as before:
  - strips every hydrogen (tleap rebuilds them from its templates)
  - GROMACS 4-letter terminal names -> 3-letter (CASN -> ASN, NGLY -> GLY,
    CCYS -> CYS, NLYS -> LYS ...); tleap makes each chain's first and last
    residue terminal by itself
  - Ile CD -> CD1, C-terminal OC1/OC2 -> O/OXT
"""
import sys
from collections import OrderedDict

AMINO = {"ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "CYM", "GLN", "GLU", "GLY", "HIS",
         "HID", "HIE", "HIP", "ILE", "LEU", "LYS", "LYN", "MET", "PHE", "PRO", "SER",
         "THR", "TRP", "TYR", "VAL", "ASH", "GLH"}
SKIP = {"SOL", "HOH", "WAT", "NA", "CL", "NA+", "CL-", "K", "K+"}
DROP_RESI, DROP_RESN = 217, "MET"


def base(rn):
    rn = rn.upper()
    return rn[1:] if len(rn) == 4 and rn[0] in "NC" and rn[1:] in AMINO else rn


def is_h(name):
    return name.lstrip("0123456789").upper().startswith("H")


def fmt_name(name):
    return name if len(name) >= 4 else " " + name.ljust(3)


if len(sys.argv) != 3:
    sys.exit(__doc__)
src, dst = sys.argv[1:]

out, serial = [], 0
prev_chain, prev_resi = None, None
chains = OrderedDict()            # chain number -> list of (resn, resi)
dropped, cur = 0, 0
last_key = None
for line in open(src):
    if line.startswith("CRYST1"):
        out.append(line)
        continue
    if not line.startswith(("ATOM", "HETATM")):
        continue
    resn = line[17:21].strip()
    if resn.upper() in SKIP:
        continue
    name = line[12:16].strip()
    ch = line[21]
    resi = int(line[22:26])
    rn = base(resn)

    new_chain = prev_resi is not None and (ch != prev_chain or resi < prev_resi)
    if new_chain:
        out.append("TER\n")
        cur += 1
    prev_chain, prev_resi = ch, resi

    key = (cur, resi)
    if rn == DROP_RESN and resi == DROP_RESI:
        if key != last_key:
            dropped += 1
        last_key = key
        continue
    last_key = key
    if is_h(name):
        continue
    if rn == "ILE" and name == "CD":
        name = "CD1"
    elif name == "OC1":
        name = "O"
    elif name == "OC2":
        name = "OXT"

    lst = chains.setdefault(cur, [])
    if not lst or lst[-1][1] != resi:
        lst.append((rn, resi))
    serial += 1
    line = line.rstrip("\n").ljust(54)
    out.append(f"{line[:6]}{serial % 100000:5d} {fmt_name(name)}{line[16]}{rn:<3} {line[21]}{line[22:]}\n")

# TER only after real atoms: drop a leading/duplicate TER left by a skipped residue
clean = []
for l in out:
    if l == "TER\n" and (not clean or clean[-1] == "TER\n" or clean[-1].startswith("CRYST1")):
        continue
    clean.append(l)
clean += ["TER\n", "END\n"]
open(dst, "w").writelines(clean)

nres = sum(len(v) for v in chains.values())
print(f"wrote {dst}: {len(chains)} chains, {nres} residues, {serial} heavy atoms, "
      f"{clean.count('TER' + chr(10))} TER records")
print(f"deleted Met{DROP_RESI} from {dropped} chains")
for k, v in chains.items():
    print(f"  chain {k + 1:2d}: {len(v):3d} residues  {v[0][0]}{v[0][1]} ... {v[-1][0]}{v[-1][1]}")
odd = sorted({r for v in chains.values() for r, _ in v if r not in AMINO})
if odd:
    print("residue names tleap may not know:", ", ".join(odd))

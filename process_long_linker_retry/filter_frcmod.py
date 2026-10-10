#!/usr/bin/env python3
"""
Keep only the frcmod lines tleap needs for CYL:
  all-upper-case types (Cys part / caps)  -> already in ff19SB      -> drop
  any lower-case (GAFF2) type             -> missing from gaff2.dat -> keep

    python3 filter_frcmod.py frcmod.CYL.gaff2 frcmod.CYL
"""
import sys

NTYPES = {"BOND": 2, "ANGLE": 3, "DIHE": 4, "IMPROPER": 4}
lines = open(sys.argv[1]).readlines()
out, section, kept, dropped = [lines[0]], None, 0, 0
for line in lines[1:]:
    word = line.strip()
    if word in ("MASS", "BOND", "ANGLE", "DIHE", "IMPROPER", "NONBON") or not word:
        section = word or section
        out.append(line)
        continue
    n = NTYPES.get(section)
    types = line[:3 * n - 1].split("-") if n else [line.split()[0]]
    if any(t.strip()[:1].islower() for t in types):
        out.append(line)
        kept += 1
    else:
        dropped += 1
open(sys.argv[2], "w").writelines(out)

attn = sum("ATTN" in l for l in out)
print("kept %d lines, dropped %d already in ff19SB, %d marked ATTN" % (kept, dropped, attn))
if attn:
    sys.exit("ERROR: ATTN lines left -- parmchk2 found no estimate for them")

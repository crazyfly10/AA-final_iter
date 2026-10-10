#!/usr/bin/env python3
"""
Make the ion entries in [ molecules ] of an acpype .top follow the order of the ions
in its .gro. tleap's addIonsRand places Na+ and Cl- alternately; acpype lists them as
two blocks, so grompp reports non-matching atom names.

    python3 fix_ion_order.py system_GMX.top system_GMX.gro     (edits the .top, keeps .top.bak)
"""
import shutil
import sys

top_file, gro_file = sys.argv[1], sys.argv[2]
lines = open(top_file).read().splitlines(keepends=True)

# single-atom moleculetypes (the ions), from the topology itself
natoms, section, current = {}, None, None
for line in lines:
    s = line.split(";")[0].strip()
    if s.startswith("["):
        section = s.strip("[] ").lower()
        continue
    if not s:
        continue
    if section == "moleculetype":
        current = s.split()[0]
        natoms[current] = 0
    elif section == "atoms" and current:
        natoms[current] += 1
ions = {name for name, n in natoms.items() if n == 1}

# order of the ions in the coordinates
order = []
for line in open(gro_file).read().splitlines()[2:-1]:
    name = line[10:15].strip()
    if name in ions:
        if order and order[-1][0] == name:
            order[-1][1] += 1
        else:
            order.append([name, 1])

# rewrite [ molecules ]: ion lines replaced, in place, by the coordinate order
out, section, done, before = [], None, False, {}
for line in lines:
    s = line.split(";")[0].strip()
    if s.startswith("["):
        section = s.strip("[] ").lower()
    elif section == "molecules" and s and s.split()[0] in ions:
        before[s.split()[0]] = before.get(s.split()[0], 0) + int(s.split()[1])
        if not done:
            out += [" %-16s %d\n" % (name, n) for name, n in order]
            done = True
        continue
    out.append(line)

after = {}
for name, n in order:
    after[name] = after.get(name, 0) + n
if before != after:
    sys.exit("ERROR: ion counts differ between .top %s and .gro %s; nothing changed" % (before, after))

shutil.copy(top_file, top_file + ".bak")
open(top_file, "w").writelines(out)
print("ions %s: [ molecules ] now lists %d ion entries in coordinate order (backup: %s.bak)"
      % (after, len(order), top_file))

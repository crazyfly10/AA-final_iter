#!/usr/bin/env python3
"""
check_ff19sb_prmtop.py  --  sanity checks for a non-standard residue in an ff19SB prmtop.

  python3 check_ff19sb_prmtop.py test.parm7 --res CYL
  python3 check_ff19sb_prmtop.py test.parm7 --res CYL --ref ref.parm7 --ref-res CYX

With --ref, build the reference from the same sequence with the parent residue
(e.g. ACE ALA CYX ALA NME vs ACE ALA CYL ALA NME) and the script also checks that
  * the modified residue gets the same CMAP and backbone/chi1 torsions as the parent
  * nothing in the rest of the peptide changed (your frcmod did not leak into ff19SB)
Needs ParmEd (ships with AmberTools).
"""
import argparse
import sys
from collections import defaultdict

import parmed as pmd

BACKBONE = ("N", "H", "CA", "HA", "CB", "C", "O")


def resid_of(parm, name):
    hits = [r for r in parm.residues if r.name == name]
    if not hits:
        sys.exit("ERROR: residue %s not found in %s" % (name, parm))
    return hits[0]


def label(atom, center):
    """Atom name, with -/+ for atoms in the previous/next residue."""
    d = atom.residue.idx - center.idx
    return {0: "", -1: "-", 1: "+"}.get(d, "?") + atom.name


def terms_by_names(parm, center, only_res=None):
    """{(kind, atom-label tuple): sorted parameters} for terms touching `center`
    (or, with only_res, for terms entirely inside the residues in only_res)."""
    out = defaultdict(list)

    def wanted(atoms):
        if only_res is not None:
            return all(a.residue.idx in only_res for a in atoms)
        return any(a.residue is center for a in atoms)

    def key(kind, atoms):
        names = tuple(label(a, center) if only_res is None
                      else "%d:%s" % (a.residue.idx, a.name) for a in atoms)
        if kind == "impr":
            names = (names[2],) + tuple(sorted(names[:2] + names[3:]))
        else:
            names = min(names, names[::-1])
        return (kind, names)

    for b in parm.bonds:
        if wanted((b.atom1, b.atom2)):
            out[key("bond", (b.atom1, b.atom2))].append((round(b.type.k, 3), round(b.type.req, 4)))
    for a in parm.angles:
        at = (a.atom1, a.atom2, a.atom3)
        if wanted(at):
            out[key("angle", at)].append((round(a.type.k, 3), round(a.type.theteq, 2)))
    for d in parm.dihedrals:
        at = (d.atom1, d.atom2, d.atom3, d.atom4)
        if wanted(at):
            kind = "impr" if d.improper else "dihe"
            out[key(kind, at)].append((int(d.type.per), round(d.type.phi_k, 4),
                                       round(d.type.phase % 360.0, 1)))
    return {k: sorted(v) for k, v in out.items()}


def cmap_of(parm, res):
    got = [c for c in parm.cmaps if c.atom3.residue is res]
    return got[0] if got else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("prmtop")
    ap.add_argument("--res", required=True, help="modified residue name, e.g. CYL")
    ap.add_argument("--ref", help="reference prmtop with the parent residue")
    ap.add_argument("--ref-res", default="CYX", help="parent residue name (default CYX)")
    a = ap.parse_args()

    parm = pmd.load_file(a.prmtop)
    res = resid_of(parm, a.res)
    bad = 0

    q = sum(at.charge for at in res.atoms)
    print("Residue %s %d: %d atoms, net charge %+.6f" % (res.name, res.idx + 1, len(res.atoms), q))
    if abs(q - round(q)) > 1e-4:
        print("  !! residue charge is not an integer"); bad += 1
    types = {at.name: at.type for at in res.atoms}
    print("  types: " + "  ".join("%s=%s" % (n, types.get(n, "-")) for n in BACKBONE + ("SG",)))
    if types.get("CA") != "XC":
        print("  !! CA is not type XC: ff19SB backbone torsions/CMAP will be inconsistent"); bad += 1

    cm = cmap_of(parm, res)
    if cm is None:
        print("  !! no CMAP term on %s (frcmod.ff19SB_%s not loaded, or wrong atom names)"
              % (res.name, res.name)); bad += 1
    else:
        same = sorted({c.atom3.residue.name for c in parm.cmaps
                       if c.type is not cm.type and list(c.type.grid) == list(cm.type.grid)})
        print("  CMAP present (%dx%d grid)%s" % (cm.type.resolution, cm.type.resolution,
              ("; same grid as " + ", ".join(same)) if same else ""))

    mine = terms_by_names(parm, res)
    print("\nBackbone torsions about N-CA and CA-C (ff19SB zeroes these; the CMAP does the work):")
    for (kind, names), vals in sorted(mine.items()):
        if kind != "dihe" or tuple(sorted(names[1:3])) not in (("CA", "N"), ("C", "CA")):
            continue
        ksum = sum(abs(v[1]) for v in vals)
        print("  %-24s sum|k| = %.4f %s" % ("-".join(names), ksum, "" if ksum < 1e-6 else "<- NONZERO"))
        if ksum > 1e-6 and ("-C" in names or "+N" in names or "HA" in names):
            bad += 1

    if a.ref:
        ref = pmd.load_file(a.ref)
        rres = resid_of(ref, a.ref_res)
        theirs = terms_by_names(ref, rres)
        print("\nCompared with %s %d in %s (terms present in both residues):" % (rres.name, rres.idx + 1, a.ref))
        shared = sorted(set(mine) & set(theirs))
        diffs = [k for k in shared if mine[k] != theirs[k]]
        for k in diffs:
            print("  differs: %-5s %-26s %s vs %s" % (k[0], "-".join(k[1]), mine[k], theirs[k]))
        print("  %d shared terms, %d differ" % (len(shared), len(diffs)))
        rcm = cmap_of(ref, rres)
        if cm is not None and rcm is not None:
            same = list(cm.type.grid) == list(rcm.type.grid)
            print("  CMAP grid identical to %s: %s" % (rres.name, same))
            bad += 0 if same else 1
        sg = [at for at in res.atoms if at.name == "SG"]
        rsg = [at for at in rres.atoms if at.name == "SG"]
        if sg and rsg:
            print("  SG Lennard-Jones: %s rmin=%.4f eps=%.4f | %s rmin=%.4f eps=%.4f"
                  % (res.name, sg[0].rmin, sg[0].epsilon, rres.name, rsg[0].rmin, rsg[0].epsilon))

        # everything outside the modified residue must be untouched
        others = [r.idx for r in parm.residues if r is not res]
        if [parm.residues[i].name for i in others] == [ref.residues[i].name for i in others
                                                       if i < len(ref.residues)]:
            t1 = terms_by_names(parm, res, only_res=set(others))
            t2 = terms_by_names(ref, rres, only_res=set(others))
            nd = sum(1 for k in set(t1) | set(t2) if t1.get(k) != t2.get(k))
            lj = sum(1 for i in others for x, y in zip(parm.residues[i].atoms, ref.residues[i].atoms)
                     if (x.type, round(x.rmin, 4), round(x.epsilon, 4), round(x.charge, 4))
                     != (y.type, round(y.rmin, 4), round(y.epsilon, 4), round(y.charge, 4)))
            ncm = sum(1 for c1, c2 in zip([c for c in parm.cmaps if c.atom3.residue is not res],
                                          [c for c in ref.cmaps if c.atom3.residue is not rres])
                      if list(c1.type.grid) != list(c2.type.grid))
            print("\nRest of the system vs reference: %d bonded-term differences, "
                  "%d atom (type/LJ/charge) differences, %d CMAP differences" % (nd, lj, ncm))
            bad += nd + lj + ncm
        bad += len([k for k in diffs if k[0] in ("dihe", "impr") and
                    all(n.lstrip("+-") in BACKBONE + ("SG", "HB2", "HB3") for n in k[1])])

    print("\n%s" % ("ALL CHECKS PASSED" if bad == 0 else "%d PROBLEM(S) FOUND" % bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

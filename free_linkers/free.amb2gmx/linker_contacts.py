#!/usr/bin/env python3
"""
linker_contacts.py -- which parts of the linkers hold them together, and which
parts touch the fibril?

    python3 linker_contacts.py solute.gro prod.xtc [from_ns] [every_ps]

defaults: from 50 ns, one frame every 500 ps (contacts change slowly).
Same inputs as free_eq_check.py.

Each linker's heavy atoms are split into four parts, using the bonds found in
solute.gro:
  cys+caps   ACE, NME and the Cys backbone up to SG (by atom name)
  spacer     the thiosuccinimide ring and the linkage to the PEG: atoms within
             7 bonds of SG
  biotin     the biotin rings, valeryl chain and its amide: within 7 bonds of the
             biotin sulfur
  PEG        everything in between
For Cys-thiosuccinimide-CH2CH2C(O)NH-PEG12-NH-biotin that gives 11 / 12 / 16 / 38
heavy atoms; the script prints what it found, so check those numbers first.

It then counts heavy-atom contacts (within 4.5 A, across the periodic boundary)
between different linkers, and between linkers and the fibril, part by part.
"expected" is the share a pair of parts would get if contacts fell on atoms at
random; a ratio well above 1 marks the pair that holds the linkers together.

Needs MDAnalysis.
"""
import sys
from collections import deque
import numpy as np

CONTACT = 4.5              # A, as in free_eq_check.py
BOND = 1.95                # A: heavy-atom pairs closer than this are bonded (C-S is 1.82)
REACH = 7                  # bonds from SG / biotin S that count as spacer / biotin
CYS_NAMES = {"N", "CA", "CB", "SG", "C", "O"}
PARTS = ["cys+caps", "spacer", "biotin", "PEG"]


def element(name):
    return name.lstrip("0123456789")[:1].upper()


def guess_bonds(x, box):
    """bonded pairs among heavy atoms, by distance, minimum image"""
    L = np.asarray(box[:3], float)
    d = x[:, None, :] - x[None, :, :]
    d -= L * np.round(d / L)
    r = np.sqrt((d ** 2).sum(-1))
    i, j = np.where((r > 0.1) & (r < BOND))
    adj = [[] for _ in range(len(x))]
    for a, b in zip(i, j):
        adj[a].append(b)
    return adj


def bfs(adj, start):
    dist = np.full(len(adj), np.inf)
    dist[start] = 0
    q = deque([start])
    while q:
        a = q.popleft()
        for b in adj[a]:
            if dist[b] == np.inf:
                dist[b] = dist[a] + 1
                q.append(b)
    return dist


def classify(names, resnames, adj):
    """part index (0-3) of each heavy atom of one ACE-CYL-NME linker"""
    n = len(names)
    part = np.full(n, -1)
    for k in range(n):
        if resnames[k] in ("ACE", "NME") or (resnames[k] == "CYL" and names[k] in CYS_NAMES):
            part[k] = 0
    sg = [k for k in range(n) if resnames[k] == "CYL" and names[k] == "SG"]
    sb = [k for k in range(n) if resnames[k] == "CYL" and element(names[k]) == "S" and names[k] != "SG"]
    if len(sg) != 1 or len(sb) != 1:
        sys.exit(f"expected one SG and one other sulfur in CYL, found {len(sg)} and {len(sb)}")
    d_sg, d_sb = bfs(adj, sg[0]), bfs(adj, sb[0])
    if np.isinf(d_sg).any():
        lost = [names[k] for k in np.where(np.isinf(d_sg))[0]]
        sys.exit(f"bond guessing left {len(lost)} heavy atoms unconnected to SG: {' '.join(lost[:10])}")
    for k in range(n):
        if part[k] < 0:
            part[k] = 1 if d_sg[k] <= REACH else 2 if d_sb[k] <= REACH else 3
    return part


def expected_pair_share(frac):
    e = np.zeros((4, 4))
    for a in range(4):
        for b in range(a, 4):
            e[a, b] = frac[a] ** 2 if a == b else 2 * frac[a] * frac[b]
    return e


def report(M, F, part1, n_frames, out=print):
    size = np.bincount(part1, minlength=4)
    frac = size / size.sum()
    tot = M.sum()
    out(f"\nlinker-linker contacts: {tot / max(n_frames, 1):.0f} heavy-atom pairs per frame on average")
    out(f"   {'parts':<22}{'observed':>10}{'expected':>10}{'ratio':>8}")
    E = expected_pair_share(frac)
    rows = [(a, b) for a in range(4) for b in range(a, 4)]
    for a, b in sorted(rows, key=lambda ab: -(M[ab] / tot / E[ab] if tot else 0)):
        obs = M[a, b] / tot if tot else 0.0
        out(f"   {PARTS[a] + ' - ' + PARTS[b]:<22}{obs:>10.1%}{E[a, b]:>10.1%}{obs / E[a, b]:>8.2f}")
    # how often each part is involved, against its share of atoms
    inv = np.array([M[a, :].sum() + M[:, a].sum() for a in range(4)], float)
    inv /= 2 * tot if tot else 1
    out(f"\n   {'part':<12}{'atoms':>7}{'share of atoms':>16}{'share of contacts':>19}{'ratio':>8}")
    for a in range(4):
        out(f"   {PARTS[a]:<12}{size[a]:>7d}{frac[a]:>16.1%}{inv[a]:>19.1%}{inv[a] / frac[a]:>8.2f}")

    ft = F.sum()
    out(f"\nlinker-fibril contacts: {ft / max(n_frames, 1):.0f} heavy-atom pairs per frame on average")
    out(f"   {'part':<12}{'share of atoms':>16}{'share of contacts':>19}{'ratio':>8}")
    for a in range(4):
        s = F[a] / ft if ft else 0.0
        out(f"   {PARTS[a]:<12}{frac[a]:>16.1%}{s:>19.1%}{s / frac[a]:>8.2f}")


def count_frame(fib, lnk, lnk_id, part_all, box, finder, M, F):
    pairs = finder(lnk, lnk, max_cutoff=CONTACT, box=box, return_distances=False)
    if len(pairs):
        a, b = pairs[:, 0], pairs[:, 1]
        keep = (a < b) & (lnk_id[a] != lnk_id[b])
        pa, pb = part_all[a[keep]], part_all[b[keep]]
        np.add.at(M, (np.minimum(pa, pb), np.maximum(pa, pb)), 1)
    pairs = finder(lnk, fib, max_cutoff=CONTACT, box=box, return_distances=False)
    if len(pairs):
        np.add.at(F, part_all[pairs[:, 0]], 1)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import capped_distance

    start = float(sys.argv[3]) * 1000.0 if len(sys.argv) > 3 else 50000.0
    every = float(sys.argv[4]) if len(sys.argv) > 4 else 500.0
    u = mda.Universe(sys.argv[1], sys.argv[2])

    rn = [r.resname for r in u.residues]
    if "ACE" not in rn:
        sys.exit("no ACE residue found -- is this the fibril + linker system?")
    first = rn.index("ACE")
    lres = u.residues[first:]
    if len(lres) % 3 or any(tuple(rn[first + i:first + i + 3]) != ("ACE", "CYL", "NME")
                            for i in range(0, len(lres), 3)):
        sys.exit("residues after the fibril are not a run of ACE-CYL-NME")
    n_link = len(lres) // 3
    fib = u.residues[:first].atoms.select_atoms("not name H*")
    lnk = lres.atoms.select_atoms("not name H*")
    per = lnk.n_atoms // n_link
    if per * n_link != lnk.n_atoms:
        sys.exit("linkers do not all have the same number of heavy atoms")
    lnk_id = np.repeat(np.arange(n_link), per)

    # parts, from the first linker's bonds in the starting structure; second linker as a check
    u.trajectory[0]
    parts = []
    for k in (0, 1):
        g = lnk[k * per:(k + 1) * per]
        parts.append(classify(list(g.names), list(g.resnames), guess_bonds(g.positions, u.dimensions)))
    if not np.array_equal(parts[0], parts[1]):
        print("WARNING: linkers 1 and 2 split differently -- bond guessing may have failed; using linker 1")
    part1 = parts[0]
    part_all = np.tile(part1, n_link)
    size = np.bincount(part1, minlength=4)
    print(f"{n_link} linkers, {per} heavy atoms each; parts: "
          + ", ".join(f"{PARTS[a]} {size[a]}" for a in range(4)))
    print("   (expected for Cys-thiosuccinimide-CH2CH2C(O)NH-PEG12-NH-biotin: 11, 12, 16, 38)")

    M = np.zeros((4, 4))
    F = np.zeros(4)
    n_frames, last = 0, -1e30
    for ts in u.trajectory:
        if ts.time < start - 1e-6 or ts.time - last < every - 1e-6:
            continue
        last = ts.time
        count_frame(fib.positions, lnk.positions, lnk_id, part_all, ts.dimensions, capped_distance, M, F)
        n_frames += 1
    print(f"{n_frames} frames from {start / 1000:.0f} ns, one every {every:.0f} ps")
    report(M, F, part1, n_frames)


if __name__ == "__main__":
    main()

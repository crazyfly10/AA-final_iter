#!/usr/bin/env python3
"""
image_contacts.py -- which part of the fibril comes close to its own periodic image?

    python3 image_contacts.py prot_so_far.gro nojump_prot.xtc [cutoff_A] [every_ps]

defaults: cutoff 20 A, one frame every 1000 ps. Use the protein-only files that
stop_check.sh writes. The trajectory must be no-jump, so the fibril is in one
piece and any short distance across the box edge is the fibril meeting its copy.

For each sampled frame: the closest heavy-atom pair that is close only across the
periodic boundary (far apart inside the fibril), and which residues they are.
At the end: the residues most often involved, so you can tell a rotating rod
(core residues of the end chains) from a floppy tail reaching across (cap tether,
His tag, N-terminal tails).

Needs MDAnalysis.
"""
import sys
from collections import Counter
import numpy as np

try:
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import capped_distance
except ImportError:
    sys.exit("pip install MDAnalysis")

CAP_FIRST_OLD = 8        # the cap keeps its own numbering from residue 8 after Step F6
N_HETS = 11
ACROSS = 5.0             # A: image distance this much shorter than the direct one = across the box


def label_residues(u):
    """chain letter and author-style number for every residue"""
    met = u.residues[0].resname.upper() in ("MET", "NMET")
    first, length = (217, 73) if met else (218, 72)
    labels = []
    for k, r in enumerate(u.residues):
        c = k // length
        if c < N_HETS:
            labels.append(f"{'ABCDEFGHIJK'[c]}:{r.resname}{first + k - c * length}")
        else:
            old = CAP_FIRST_OLD + k - N_HETS * length
            part = "core" if 10 <= old <= 66 else "tether" if old >= 74 else "rung"
            labels.append(f"L:{r.resname}{old} (cap {part})")
    return labels, first, length


def image_pairs(pos, box, cutoff, finder=capped_distance):
    """pairs (i<j) whose minimum-image distance is short only because of the
    periodic boundary; returns (i, j, d_image) sorted by d_image"""
    pairs, d_img = finder(pos, pos, max_cutoff=cutoff, box=box, return_distances=True)
    keep = pairs[:, 0] < pairs[:, 1]
    pairs, d_img = pairs[keep], d_img[keep]
    d_dir = np.linalg.norm(pos[pairs[:, 0]] - pos[pairs[:, 1]], axis=1)
    across = d_dir - d_img > ACROSS
    pairs, d_img = pairs[across], d_img[across]
    order = np.argsort(d_img)
    return pairs[order], d_img[order]


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    top, traj = sys.argv[1], sys.argv[2]
    cutoff = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
    every = float(sys.argv[4]) if len(sys.argv) > 4 else 1000.0

    u = mda.Universe(top, traj)
    heavy = u.select_atoms("not name H*")
    labels, first, length = label_residues(u)
    print(f"{len(u.trajectory)} frames, {heavy.n_atoms} heavy atoms; HET-s chains "
          f"{first}-{first + length - 1}; image contacts under {cutoff:.0f} A\n")
    print(f"{'t/ns':>7}  {'closest/A':>9}  pair")
    print("-" * 72)

    involved = Counter()
    last_t = -1e30
    worst = (np.inf, None, None)
    for ts in u.trajectory:
        if ts.time - last_t < every - 1e-6:
            continue
        last_t = ts.time
        pos = heavy.positions
        pairs, d = image_pairs(pos, ts.dimensions, cutoff)
        if len(d) == 0:
            print(f"{ts.time/1000:7.1f}  {'> ' + str(int(cutoff)):>9}")
            continue
        i, j = pairs[0]
        ri, rj = heavy[i].residue.ix, heavy[j].residue.ix
        print(f"{ts.time/1000:7.1f}  {d[0]:9.2f}  {labels[ri]}  <->  image of {labels[rj]}")
        for a, b in pairs[d < 12.0]:
            involved[labels[heavy[a].residue.ix]] += 1
            involved[labels[heavy[b].residue.ix]] += 1
        if d[0] < worst[0]:
            worst = (d[0], ts.time, (labels[ri], labels[rj]))

    print()
    if worst[1] is not None:
        print(f"closest overall: {worst[0]:.2f} A at {worst[1]/1000:.1f} ns, "
              f"{worst[2][0]} <-> image of {worst[2][1]}")
    if involved:
        print("\nresidues most often within 12 A of the fibril's image (atom-pair counts):")
        for lab, n in involved.most_common(15):
            print(f"  {n:6d}  {lab}")
    else:
        print("no part of the fibril came within 12 A of its own image")


if __name__ == "__main__":
    main()

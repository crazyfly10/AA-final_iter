#!/usr/bin/env python3
"""
linked_eq_check.py -- from when is the linked run usable for the bonded targets?

    python3 linked_eq_check.py solute.gro prot_cyl.xtc [every_ps]

solute.gro / prot_cyl.xtc: fibril + attached linkers only (Prot_cyl), written with
trjconv; ions in the group are ignored. Default: one frame every 100 ps.

The bonded distributions come from the attached CYL linkers, so what has to have
stopped changing is the linkers' conformation. Three slow descriptors per linker
stand in for it (the bond, angle and dihedral distributions of the coarse-grained
beads settle at least as fast as these):
  Rg        radius of gyration of the linker's heavy atoms
  reach     SG to biotin S distance: how far the linker stretches
  on        PEG + biotin heavy atoms within 4.5 A of the fibril: lying on the
            surface or out in the water

Where to start: the number of statistically independent samples left after a
start time (Chodera, JCTC 12, 1799 (2016)) is low for a start still inside the
initial relaxation, because the drift makes the data look strongly correlated,
and falls again for late starts, which throw good data away. For each descriptor
the script takes the earliest start that keeps at least 90% of the best number;
the latest of these over all descriptors and linkers is the one to use. It
then checks that the first and second half of the kept stretch agree within
their statistical uncertainty, and says how many independent samples each
descriptor has: a few dozen or more is comfortable.

It also reports how often each linker went on and off the fibril: a linker that
never left the surface gives bonded distributions for the lying-down state only.

Needs MDAnalysis.
"""
import sys
from collections import deque
import numpy as np

CONTACT = 4.5          # A
BOND = 1.95            # A: heavy-atom pairs closer than this are bonded
REACH = 7              # bonds from SG / biotin S that count as spacer / biotin
BACKBONE = {"N", "CA", "CB", "SG", "C", "O"}
SKIP_RES = {"CYL", "WAT", "SOL", "HOH", "Na+", "Cl-", "NA", "CL", "K+", "K"}
START_STEP_NS = 5.0    # candidate start times
Z_MAX = 3.0            # halves may differ by this many standard errors
BLOCK_NS = 20.0
ON_MIN = 5             # PEG+biotin atoms on the fibril to count the linker as lying on it


def element(name):
    return name.lstrip("0123456789")[:1].upper()


def guess_bonds(x, box):
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
    parent = np.full(len(adj), -1)
    order = [start]
    dist[start] = 0
    q = deque([start])
    while q:
        a = q.popleft()
        for b in adj[a]:
            if dist[b] == np.inf:
                dist[b] = dist[a] + 1
                parent[b] = a
                order.append(b)
                q.append(b)
    return dist, parent, order


class Linker:
    """one attached CYL: which heavy atoms are SG, biotin S, PEG + biotin, and a
    bond tree for making it whole across the periodic boundary"""

    def __init__(self, names, x0, box):
        n = len(names)
        adj = guess_bonds(x0, box)
        sg = [k for k in range(n) if names[k] == "SG"]
        sb = [k for k in range(n) if element(names[k]) == "S" and names[k] != "SG"]
        if len(sg) != 1 or len(sb) != 1:
            sys.exit(f"CYL: expected one SG and one other sulfur, found {len(sg)} and {len(sb)}")
        self.sg, self.sb = sg[0], sb[0]
        d_sg, self.parent, self.order = bfs(adj, self.sg)
        if np.isinf(d_sg).any():
            lost = [names[k] for k in np.where(np.isinf(d_sg))[0]]
            sys.exit(f"CYL: bond guessing left atoms unconnected to SG: {' '.join(lost[:10])}")
        d_sb = bfs(adj, self.sb)[0]
        part = np.array([0 if names[k] in BACKBONE else 1 if d_sg[k] <= REACH
                         else 2 if d_sb[k] <= REACH else 3 for k in range(n)])
        self.sizes = np.bincount(part, minlength=4)
        self.distal = np.where(part >= 2)[0]          # biotin + PEG
        self.biotin = np.where(part == 2)[0]

    def whole(self, x, box):
        L = np.asarray(box[:3], float)
        y = x.copy()
        for a in self.order[1:]:
            p = self.parent[a]
            d = y[a] - y[p]
            y[a] = y[p] + d - L * np.round(d / L)
        return y


def frame_values(link, x, fib, box, finder):
    """Rg, reach (SG-biotin S), PEG+biotin atoms on the fibril, biotin on the fibril"""
    y = link.whole(x, box)
    rg = float(np.sqrt(((y - y.mean(0)) ** 2).sum(1).mean()))
    reach = float(np.linalg.norm(y[link.sb] - y[link.sg]))
    pairs = finder(x[link.distal], fib, max_cutoff=CONTACT, box=box, return_distances=False)
    touching = np.unique(pairs[:, 0]) if len(pairs) else np.array([], int)
    on = len(touching)
    biotin_on = bool(np.isin(link.distal[touching], link.biotin).any()) if on else False
    return rg, reach, on, biotin_on


def stat_ineff(y, mintime=3):
    """statistical inefficiency g (in frames): the data hold len(y)/g independent samples"""
    y = np.asarray(y, float)
    n = len(y)
    dy = y - y.mean()
    var = dy.var()
    if n < 3 or var == 0:
        return 1.0
    g = 1.0
    for lag in range(1, n - 1):
        c = np.dot(dy[:n - lag], dy[lag:]) / ((n - lag) * var)
        if c <= 0 and lag > mintime:
            break
        g += 2.0 * c * (1.0 - lag / n)
    return max(1.0, g)


def detect_start(y, step, keep=0.9):
    """earliest start frame (multiple of step, within the first half) whose number
    of independent samples left is within `keep` of the best start's. The best
    start alone is noisy once the data are stationary; the earliest near-best one
    marks where the initial relaxation ends."""
    cand = []
    for i0 in range(0, len(y) // 2 + 1, step):
        g = stat_ineff(y[i0:])
        cand.append((i0, g, (len(y) - i0) / g))
    top = max(c[2] for c in cand)
    return next(c for c in cand if c[2] >= keep * top)


def halves(y):
    """difference of the two halves' means in standard errors, the means, and the
    number of independent samples in the whole stretch"""
    h = len(y) // 2
    a, b = y[:h], y[h:]
    ga, gb = stat_ineff(a), stat_ineff(b)
    se = np.sqrt(a.var() * ga / len(a) + b.var() * gb / len(b))
    z = abs(a.mean() - b.mean()) / se if se > 0 else 0.0
    return z, a.mean(), b.mean(), len(y) / stat_ineff(y)


def runs(s):
    d = np.diff(np.r_[0, np.asarray(s, int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def switches(on, g):
    """number of lie-down / lift-off events, ignoring stretches shorter than g frames"""
    s = np.asarray(on, bool).copy()
    for a, b in runs(~s):
        if a > 0 and b < len(s) and b - a < g:
            s[a:b] = True
    for a, b in runs(s):
        if b - a < g:
            s[a:b] = False
    return max(0, len(np.flatnonzero(np.diff(s.astype(int)))))


def analyse(t, data, labels, out=print):
    """t: times (ps); data[k]: array (frames, 4) for linker k"""
    dt = float(np.median(np.diff(t)))
    names = ("Rg", "reach", "on")
    out(f"{len(t)} frames, {t[0]/1000:.0f}-{t[-1]/1000:.0f} ns, one every {dt:.0f} ps\n")

    for k, lab in enumerate(labels):
        d = data[k]
        on = d[:, 2] >= ON_MIN
        out(f"{lab}: {d[:, 2].mean():.1f} PEG+biotin atoms on the fibril on average; lying on it "
            f"{on.mean():.0%} of the time, biotin on it {d[:, 3].mean():.0%}; "
            f"{switches(on, max(1, int(round(1000 / dt))))} lie-down/lift-off events")
        out(f"   {'block/ns':>10}{'Rg/A':>8}{'reach/A':>9}{'on':>7}{'biotin on':>11}")
        nb = int(np.ceil((t[-1] - t[0] + dt) / (BLOCK_NS * 1000)))
        for b in range(nb):
            t0 = t[0] + b * BLOCK_NS * 1000
            s = (t >= t0) & (t < t0 + BLOCK_NS * 1000)
            if s.sum() < 2:
                continue
            out(f"   {t0/1000:>4.0f}-{(t0 + BLOCK_NS*1000)/1000:<5.0f}{d[s, 0].mean():>8.2f}"
                f"{d[s, 1].mean():>9.2f}{d[s, 2].mean():>7.1f}{d[s, 3].mean():>11.0%}")
        out("")

    labs = [f"{lab} {n}" for lab in labels for n in names]
    series = [data[k][:, j] for k in range(len(labels)) for j in range(3)]
    step = max(1, int(round(START_STEP_NS * 1000 / dt)))
    out("earliest start keeping at least 90% of the most independent samples, per descriptor")
    out(f"   {'descriptor':<16}{'start/ns':>9}{'corr. time/ns':>15}{'independent samples':>21}")
    starts = []
    for lab, y in zip(labs, series):
        i0, g, neff = detect_start(y, step)
        starts.append(i0)
        out(f"   {lab:<16}{t[i0]/1000 - t[0]/1000:>9.0f}{g * dt / 1000:>15.2f}{neff:>21.0f}")
    i0 = max(starts)
    best = (t[i0] - t[0]) / 1000
    out(f"\nlatest of these: {best:.0f} ns -> keep {(t[-1] - t[i0]) / 1000:.0f} ns")

    out(f"\nfirst half vs second half of the kept stretch (difference in standard errors; under {Z_MAX:g} = agree)")
    out(f"   {'descriptor':<16}{'1st half':>10}{'2nd half':>10}{'difference':>12}{'independent samples':>21}")
    worst = 0.0
    for lab, y in zip(labs, series):
        z, m1, m2, neff = halves(y[i0:])
        worst = max(worst, z)
        out(f"   {lab:<16}{m1:>10.2f}{m2:>10.2f}{z:>12.1f}{neff:>21.0f}" + ("   DIFFER" if z >= Z_MAX else ""))
    out("")
    if worst < Z_MAX:
        out(f"use {best:.0f} ns to the end for the bonded distributions")
    else:
        out(f"the halves still differ after {best:.0f} ns: the linker conformations have not settled, "
            "or switch too rarely for this run to average over")
    return best


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import capped_distance

    every = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0
    u = mda.Universe(sys.argv[1], sys.argv[2])
    cyl = [r for r in u.residues if r.resname == "CYL"]
    if not cyl:
        sys.exit("no CYL residue found -- is this the linked system?")
    fib = u.atoms[[a.index for r in u.residues if r.resname not in SKIP_RES for a in r.atoms
                   if not a.name.startswith("H")]]
    heavy = [r.atoms[np.array([not a.name.startswith("H") for a in r.atoms], bool)] for r in cyl]

    u.trajectory[0]
    links = [Linker(list(h.names), h.positions.copy(), u.dimensions) for h in heavy]
    labels = [f"CYL{r.resid}" for r in cyl]
    print(f"fibril: {fib.n_atoms} heavy atoms; {len(cyl)} attached linkers: "
          + ", ".join(f"{lab} ({lk.sizes[2] + lk.sizes[3]} PEG+biotin atoms)" for lab, lk in zip(labels, links)))

    t, rows, last = [], [[] for _ in links], -1e30
    for ts in u.trajectory:
        if ts.time - last < every - 1e-6:
            continue
        last = ts.time
        t.append(ts.time)
        fpos = fib.positions
        for k, (h, lk) in enumerate(zip(heavy, links)):
            rows[k].append(frame_values(lk, h.positions, fpos, ts.dimensions, capped_distance))
    analyse(np.array(t), [np.array(r, float) for r in rows], labels)


if __name__ == "__main__":
    main()

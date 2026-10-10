#!/usr/bin/env python3
"""
linker_exchange.py -- do the free linkers really come and go, and where is the big cluster?

    python3 linker_exchange.py solute.gro prod.xtc [from_ns] [every_ps]

defaults: from 50 ns, one frame every 100 ps. Same inputs as free_eq_check.py.

free_eq_check.py counts linkers; this follows each one (numbered L1-L40 in the
order they sit in the topology).

  1. Adsorption, per linker: how long it spent on the fibril and how many separate
     visits. A visit has to last 1 ns, and gaps shorter than 1 ns are bridged, so
     a linker brushing the surface or lifting off for a moment does not count.
     The count of adsorbed linkers going up and down does not by itself show
     exchange: the same few could stay on throughout while others flicker.
  2. The largest linker cluster: which linkers are in it, whether it keeps the
     same members, and how many of them are on the fibril. A patch of linkers
     stuck on the fibril and an aggregate floating in solution mean different
     things for the RDFs.
  3. The largest cluster among linkers NOT touching the fibril: aggregation in
     solution.

Contacts: any heavy atom within 4.5 A, measured across the periodic boundary,
as in free_eq_check.py. Needs MDAnalysis.
"""
import sys
import numpy as np

CONTACT = 4.5          # A: heavy-atom contact, as in free_eq_check.py
MIN_VISIT_PS = 1000.0  # a visit must last this long; shorter gaps are bridged
MOST = 0.9             # "throughout" = more than 90% of the window
BLOCK_PS = 10000.0     # block length for the cluster table


# ----------------------------------------------------------------- per frame
def components(n, links):
    """connected-component label of each of n nodes"""
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in links:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return np.array([find(i) for i in range(n)])


def frame_state(fib, lnk, lnk_id, n_link, box, finder):
    """adsorbed (bool per linker), member of the largest cluster (bool per
    linker), size of the largest cluster among linkers off the fibril"""
    ads = np.zeros(n_link, bool)
    pairs = finder(lnk, fib, max_cutoff=CONTACT, box=box, return_distances=False)
    if len(pairs):
        ads[np.unique(lnk_id[pairs[:, 0]])] = True

    pairs = finder(lnk, lnk, max_cutoff=CONTACT, box=box, return_distances=False)
    links = set()
    if len(pairs):
        a, b = lnk_id[pairs[:, 0]], lnk_id[pairs[:, 1]]
        keep = a < b
        links = set(zip(a[keep].tolist(), b[keep].tolist()))

    root = components(n_link, links)
    lab, cnt = np.unique(root, return_counts=True)
    big = root == lab[np.argmax(cnt)]

    off = [(i, j) for i, j in links if not ads[i] and not ads[j]]
    root2 = components(n_link, off)
    free_max = int(np.unique(root2[~ads], return_counts=True)[1].max()) if (~ads).any() else 0
    return ads, big, free_max


# ----------------------------------------------------------------- analysis
def runs(s):
    """(start, end) of each stretch of True, end exclusive"""
    d = np.diff(np.r_[0, np.asarray(s, int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def visits(s, g):
    """stretches on the fibril lasting at least g frames, after bridging gaps
    shorter than g frames"""
    s = np.asarray(s, bool).copy()
    for a, b in runs(~s):
        if a > 0 and b < len(s) and b - a < g:
            s[a:b] = True
    for a, b in runs(s):
        if b - a < g:
            s[a:b] = False
    return runs(s)


def names(idx):
    return " ".join(f"L{i + 1}" for i in idx) if len(idx) else "none"


def analyse(times, ads, big, free_max, out=print):
    T, n_link = ads.shape
    dt = float(np.median(np.diff(times))) if T > 1 else 1.0
    g = max(1, int(round(MIN_VISIT_PS / dt)))
    span = (times[-1] - times[0] + dt) / 1000.0
    out(f"{T} frames, {times[0]/1000:.0f}-{times[-1]/1000:.0f} ns, one every {dt:.0f} ps; "
        f"visits must last {MIN_VISIT_PS/1000:.0f} ns\n")

    # ---- 1. adsorption per linker
    frac = ads.mean(0)
    rows, arrivals, departures, lengths = [], 0, 0, []
    for k in range(n_link):
        v = visits(ads[:, k], g)
        arr = sum(1 for a, b in v if a > 0)
        dep = sum(1 for a, b in v if b < T)
        arrivals += arr
        departures += dep
        lengths += [(b - a) * dt / 1000.0 for a, b in v if a > 0 and b < T]
        longest = max(((b - a) * dt / 1000.0 for a, b in v), default=0.0)
        rows.append((k, frac[k], len(v), arr, dep, longest))

    visited = [r[0] for r in rows if r[2] > 0]
    stuck = [r[0] for r in rows if r[1] > MOST]
    never = [r[0] for r in rows if r[2] == 0]
    out("1. adsorption, linker by linker")
    lab = lambda s: f"   {s:<32}"
    out(lab("on the fibril on average") + f"{ads.sum(1).mean():.2f} linkers")
    out(lab("visited the fibril") + f"{len(visited)} of {n_link}")
    out(lab(f"on for >{MOST:.0%} of the window") + f"{len(stuck)}  ({names(stuck)})")
    out(lab("never visited") + f"{len(never)}")
    out(lab("arrivals / departures") + f"{arrivals} / {departures} in {span:.0f} ns")
    if lengths:
        out(lab("completed visits") + f"{len(lengths)}, median {np.median(lengths):.1f} ns, "
            f"longest {max(lengths):.1f} ns")
    else:
        out(lab("completed visits") + "none (no linker both arrived and left inside the window)")
    out("\n   linkers on the fibril for at least 10% of the window:")
    out(f"   {'linker':>7}{'on':>7}{'visits':>8}{'arrive':>8}{'leave':>7}{'longest/ns':>12}")
    for k, f, nv, arr, dep, lo in sorted(rows, key=lambda r: -r[1]):
        if f >= 0.10:
            out(f"   {'L' + str(k + 1):>7}{f:>7.0%}{nv:>8d}{arr:>8d}{dep:>7d}{lo:>12.1f}")

    # ---- 2. largest cluster
    size = big.sum(1)
    on = (big & ads).sum(1)
    core = np.where(big.mean(0) > MOST)[0]
    out("\n2. largest linker cluster")
    out(f"   mean size {size.mean():.2f}; of its members, on average {on.mean():.2f} "
        f"({(on / np.maximum(size, 1)).mean():.0%}) are on the fibril")
    out(f"   in it for >{MOST:.0%} of the window: {len(core)}  ({names(core)})")
    if len(core):
        both = np.intersect1d(core, stuck)
        out(f"   of those, also on the fibril for >{MOST:.0%}: {len(both)}  ({names(both)})")
    out(f"\n   {'block/ns':>10}{'size':>7}{'on fibril':>11}{'same members':>14}{'off-fibril max':>16}")
    prev = None
    nb = int(np.ceil((times[-1] - times[0] + dt) / BLOCK_PS))
    for b in range(nb):
        t0 = times[0] + b * BLOCK_PS
        sel = (times >= t0) & (times < t0 + BLOCK_PS)
        if sel.sum() < 2:
            continue
        i0 = np.where(sel)[0][0]
        members = set(np.where(big[i0])[0])
        same = f"{len(members & prev)}/{len(members)}" if prev is not None else "-"
        prev = members
        out(f"   {t0/1000:>4.0f}-{(t0 + BLOCK_PS)/1000:<5.0f}{size[sel].mean():>7.2f}{on[sel].mean():>11.2f}"
            f"{same:>14}{free_max[sel].mean():>16.2f}")
    out("   (same members: how many of the cluster's members at the start of the block were")
    out("    also in it at the start of the previous block)")

    # ---- 3. aggregation off the fibril
    out("\n3. largest cluster among linkers off the fibril")
    out(f"   mean {free_max.mean():.2f}, largest seen {free_max.max()}")
    return dict(visited=len(visited), stuck=stuck, arrivals=arrivals, departures=departures,
                core=core, on_frac=(on / np.maximum(size, 1)).mean(), free_mean=free_max.mean())


# ----------------------------------------------------------------- main
def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import capped_distance

    start = float(sys.argv[3]) * 1000.0 if len(sys.argv) > 3 else 50000.0
    every = float(sys.argv[4]) if len(sys.argv) > 4 else 100.0
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
    lnk_id = np.repeat(np.arange(n_link),
                       [len(lres[3 * k:3 * k + 3].atoms.select_atoms("not name H*")) for k in range(n_link)])
    print(f"fibril: {fib.n_atoms} heavy atoms; {n_link} linkers, {lnk.n_atoms // n_link} heavy atoms each")

    times, A, B, F, last = [], [], [], [], -1e30
    for ts in u.trajectory:
        if ts.time < start - 1e-6 or ts.time - last < every - 1e-6:
            continue
        last = ts.time
        a, b, f = frame_state(fib.positions, lnk.positions, lnk_id, n_link, ts.dimensions, capped_distance)
        times.append(ts.time)
        A.append(a)
        B.append(b)
        F.append(f)
    if len(times) < 10:
        sys.exit(f"only {len(times)} frames after {start/1000:.0f} ns")
    analyse(np.array(times), np.array(A), np.array(B), np.array(F))


if __name__ == "__main__":
    main()

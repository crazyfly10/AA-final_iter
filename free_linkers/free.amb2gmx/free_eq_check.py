#!/usr/bin/env python3
"""
free_eq_check.py -- have the free linkers equilibrated with the fibril?

    python3 free_eq_check.py solute.gro prod.xtc [every_ps]

solute.gro / prod.xtc: the fibril + linkers only (production writes only
Prot_cyl to the .xtc). No no-jump trajectory needed: every distance here is
measured across the periodic boundary where that is shorter.

Before taking any RDF from this run, three things must have stopped changing:
  adsorbed   linkers with a heavy atom within 4.5 A of the fibril -- the
             adsorption/desorption balance
  Rg         mean radius of gyration of the linkers -- they start from
             conformations at most 1 ns old (Step 8 ran 1 ns)
  cluster    size of the largest linker-linker cluster -- if this keeps
             growing, the linkers are aggregating, and a linker-linker RDF
             from this run describes an aggregate, not a solution

Each is judged with the same drift test as check_capped_stability_3.py, over
the last half of the run. Writes free_eq.dat and free_eq.png.
Needs MDAnalysis and matplotlib.
"""
import sys
import numpy as np

CONTACT = 4.5          # A: heavy-atom contact
EQ_FRACTION = 0.5
TINY = 1e-12


def settled(t, y):
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if len(y) < 4 or t.std() == 0:
        return True, 0.0, max(float(y.std()), TINY), 0.0, float("inf")
    a, b = np.polyfit(t, y, 1)
    res = y - (a * t + b)
    r1 = np.corrcoef(res[:-1], res[1:])[0, 1] if res.std() > 0 else 0.0
    r1 = min(max(r1 if np.isfinite(r1) else 0.0, 0.0), 0.99)
    neff = max(3.0, len(y) * (1 - r1) / (1 + r1))
    sigma = max(float(res.std()), TINY)
    se = max(sigma / np.sqrt(neff) / t.std(), TINY)
    drift = a * (t[-1] - t[0])
    ok = abs(drift) < sigma or abs(a) < 3 * se
    return ok, float(drift), sigma, a * 1000.0, se * 1000.0


def unwrap_sequential(x, box):
    """make one molecule whole: place each atom at the minimum image of the
    atom before it (consecutive atoms in an Amber residue are 1-2 bonds apart)"""
    L = np.asarray(box[:3], float)
    y = x.copy()
    for k in range(1, len(y)):
        d = y[k] - y[k - 1]
        y[k] = y[k - 1] + d - L * np.round(d / L)
    return y


def radius_of_gyration(x):
    return float(np.sqrt(((x - x.mean(0)) ** 2).sum(1).mean()))


def largest_cluster(n, links):
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
    sizes = {}
    for i in range(n):
        r = find(i)
        sizes[r] = sizes.get(r, 0) + 1
    return max(sizes.values()) if sizes else 0


def frame_stats(fib, lnk, lnk_id, n_link, box, finder):
    """fib, lnk: heavy-atom coordinates; lnk_id: linker index of each lnk atom"""
    pairs = finder(lnk, fib, max_cutoff=CONTACT, box=box, return_distances=False)
    adsorbed = len(set(lnk_id[pairs[:, 0]])) if len(pairs) else 0

    pairs = finder(lnk, lnk, max_cutoff=CONTACT, box=box, return_distances=False)
    if len(pairs):
        a, b = lnk_id[pairs[:, 0]], lnk_id[pairs[:, 1]]
        keep = a != b
        links = set(zip(a[keep].tolist(), b[keep].tolist()))
    else:
        links = set()
    cluster = largest_cluster(n_link, links)

    rg = np.mean([radius_of_gyration(unwrap_sequential(lnk[lnk_id == k], box))
                  for k in range(n_link)])
    return adsorbed, rg, cluster


def main():
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import capped_distance
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(sys.argv) < 3:
        sys.exit(__doc__)
    every = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0
    u = mda.Universe(sys.argv[1], sys.argv[2])

    names = [r.resname for r in u.residues]
    if "ACE" not in names:
        sys.exit("no ACE residue found -- is this the fibril + linker system?")
    first = names.index("ACE")
    lres = u.residues[first:]
    if len(lres) % 3 or any(tuple(names[first + i:first + i + 3]) != ("ACE", "CYL", "NME")
                            for i in range(0, len(lres), 3)):
        sys.exit("residues after the fibril are not a run of ACE-CYL-NME")
    n_link = len(lres) // 3
    fib = u.residues[:first].atoms.select_atoms("not name H*")
    lnk = lres.atoms.select_atoms("not name H*")
    lnk_id = np.repeat(np.arange(n_link), [len(lres[3 * k:3 * k + 3].atoms.select_atoms("not name H*"))
                                           for k in range(n_link)])
    print(f"fibril: {first} residues, {fib.n_atoms} heavy atoms; linkers: {n_link} x ACE-CYL-NME, "
          f"{lnk.n_atoms // n_link} heavy atoms each; {len(u.trajectory)} frames\n")

    rows, last = [], -1e30
    for ts in u.trajectory:
        if ts.time - last < every - 1e-6:
            continue
        last = ts.time
        rows.append((ts.time,) + frame_stats(fib.positions, lnk.positions, lnk_id, n_link,
                                             ts.dimensions, capped_distance))
    d = np.array(rows)
    np.savetxt("free_eq.dat", d, fmt=["%.1f", "%d", "%.3f", "%d"],
               header="time_ps adsorbed mean_Rg_A largest_cluster")

    cut = int(len(d) * (1 - EQ_FRACTION))
    t = d[cut:, 0]
    print(f"{'quantity':<16}{'mean':>9}{'drift':>10}{'sigma':>8}  last half {t[0]/1000:.0f}-{t[-1]/1000:.0f} ns")
    print("-" * 60)
    verdicts = []
    for j, (lab, unit) in enumerate((("adsorbed", ""), ("mean Rg", " A"), ("largest cluster", "")), start=1):
        ok, drift, sigma, slope, se = settled(t, d[cut:, j])
        print(f"{lab:<16}{d[cut:, j].mean():>9.2f}{drift:>+10.2f}{sigma:>8.2f}  "
              f"{'settled' if ok else 'NOT settled'}")
        verdicts.append((lab, ok))
    agg = d[cut:, 3].mean() > n_link / 2
    print(f"\nlargest cluster averages {d[cut:, 3].mean():.1f} of {n_link} linkers"
          + (" -- MORE THAN HALF: the linkers are aggregating" if agg else ""))
    print("\nverdict")
    for lab, ok in verdicts:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {lab} has settled")
    print(f"  [{'FAIL' if agg else 'PASS'}]  linkers not aggregated into one cluster")
    print(f"\ndiscard everything before {t[0]/1000:.0f} ns at least; take RDFs only from the "
          "stretch where all three are flat (see free_eq.png)")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, j, lab in zip(ax, (1, 2, 3), ("linkers on the fibril", "mean linker Rg (A)", "largest linker cluster")):
        a.plot(d[:, 0] / 1000, d[:, j], lw=0.8)
        a.axvline(d[cut, 0] / 1000, ls=":", color="grey")
        a.set_xlabel("time (ns)")
        a.set_title(lab)
        a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("free_eq.png", dpi=150)
    print("wrote free_eq.dat and free_eq.png")


if __name__ == "__main__":
    main()

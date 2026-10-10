#/usr/bin/env python3
"""
check_capped_stability.py — is the capped-fibril MD run stable and equilibrated?

Answers four separate questions, because "is it stable" bundles several
things that can fail independently:

  1. Is the SIMULATION well-behaved?         (energy, temperature, pressure)
  2. Is the FIBRIL BODY stable?               (interior backbone RMSD)
  3. Is the CAP STILL DOCKED?                 (K-L core-COM distance over time)
  4. Is the CAP INTERFACE REAL, not just nearby?  (H-bond count vs an
     interior interface, as a reference)

A cap that satisfies (3) but fails (4) is sitting next to the fibril without
actually templating onto it — geometrically "docked" but not a real stacking
interface. That distinction matters and neither RMSD nor distance alone can
see it.

Requires: pip install MDAnalysis matplotlib

Usage:
    python3 check_capped_stability.py prod.tpr nojump.xtc prod.edr
"""
import os
import sys
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import rms
except ImportError:
    sys.exit("pip install MDAnalysis")

# ---------------------------------------------------------------- config
TPR, XTC = sys.argv[1], sys.argv[2]
EDR = sys.argv[3] if len(sys.argv) > 3 else None

CORE_HETS = [(226, 234), (237, 245), (262, 270), (273, 282)]

# EDIT to match your actual residue numbering once confirmed. If your chains
# are contiguous blocks (typical after tleap/pdb2gmx), fill these in as
# (first_resid, n_residues) per chain in axial order.
CHAIN_BLOCKS = [(1,73), (74,73), (147,73), (220,73), (293,73), (366,73),
                (439,73), (512,73), (585,73), (658,73), (731,73), (804,123)]

INTERIOR_CHAINS = (5, 6, 7, 8)   # 0-indexed positions in axial order, e.g. F,G,H,I
CAP_CHAIN_IDX   = -1             # last chain = the cap
LAST_FIBRIL_IDX = -2             # chain K = second-to-last

STRIDE = 5


def core_sel_from_block(first, n, res_offset=0):
    """
    CORE_HETS is in HET-s author numbering (starts at 217/218). Each chain's
    own numbering starts at 'first' = 1 locally. Map HET-s resid X to local
    position (X - 217), then to the chain's actual resid (first + that - 1).
    """
    HETS_START = 217
    parts = []
    for lo, hi in CORE_HETS:
        local_lo = lo - HETS_START
        local_hi = hi - HETS_START
        parts.append(f"resid {first+local_lo+res_offset}:{first+local_hi+res_offset}")
    return " or ".join(parts)


def core_sel_cap(cap_offset=796):
    """
    The cap had residues 1-7 deleted and was renumbered sequentially, so old
    cap-numbering n now sits at new resid (n + 796) — confirmed empirically:
    old cap residue 8 (Gly) -> new resid 804.

    Core in OLD cap numbering was 10-18, 21-29, 46-54, 57-66 (HET-s 226-234,
    237-245, 262-270, 273-282 via the +216 offset). Shift by cap_offset.
    """
    core_old = [(10,18), (21,29), (46,54), (57,66)]
    parts = [f"resid {lo+cap_offset}:{hi+cap_offset}" for lo, hi in core_old]
    return " or ".join(parts)

def load_universe(top, traj):
    """GROMACS .top -> ITP reader; anything else (.tpr, .gro, .prmtop) as before."""
    if top.endswith(".top"):
        gmx_top = os.path.join(os.environ.get("GMXDATA", "/usr/local/gromacs/share/gromacs"), "top")
        return mda.Universe(top, traj, topology_format="ITP", include_dir=gmx_top)
    return mda.Universe(top, traj)
# ------------------------------------------------------------------ setup
u = load_universe(sys.argv[1], sys.argv[2])
# the .top reader numbers residues 1..N across the whole system;
# copy back GROMACS's per-chain residue numbers from ref.gro (same atom order)
import numpy as np
ref = mda.Universe("ref.gro")
if ref.atoms.n_atoms != u.atoms.n_atoms:
    sys.exit("ref.gro has %d atoms, topology %d" % (ref.atoms.n_atoms, u.atoms.n_atoms))
first = np.unique(u.atoms.resindices, return_index=True)[1]   # first atom of each residue
u.residues.resids = ref.atoms.resids[first]
u.residues.resnums = ref.atoms.resids[first]
print(f"{len(u.trajectory)} frames, {len(u.atoms)} atoms\n")

if CHAIN_BLOCKS is None:
    sys.exit(f"""
Set CHAIN_BLOCKS before running. Find your layout with:

    python3 -c "
import MDAnalysis as mda
u = mda.Universe('{TPR}')
p = u.select_atoms('protein')
print('protein residues:', p.n_residues)
r = p.residues
prev = None
for i, res in enumerate(r):
    if prev is None or res.resid != prev + 1:
        print('new block starts at', res.resid, res.resname)
    prev = res.resid
"

then fill CHAIN_BLOCKS = [(first_resid, n_residues), ...] for the 11 HET-s
chains (should all be the same length) plus the cap (shorter, different
count since residues 1-7 were deleted).
""")

n_chains = len(CHAIN_BLOCKS)
print(f"chains: {n_chains} blocks")
for i, (first, n) in enumerate(CHAIN_BLOCKS):
    print(f"  chain {i}: resid {first}-{first+n-1}  ({n} residues)")

# --------------------------------------------------------- core selections
cores = []
for i, (first, n) in enumerate(CHAIN_BLOCKS):
    if i == len(CHAIN_BLOCKS) - 1:          # last chain = the cap
        sel = core_sel_cap() + " and name CA"
    else:
        sel = core_sel_from_block(first, n) + " and name CA"
    g = u.select_atoms(sel)
    cores.append(g)
    flag = "" if g.n_atoms == 37 else "  <-- check numbering"
    print(f"  chain {i} core atoms: {g.n_atoms}{flag}")

assert all(g.n_atoms > 0 for g in cores), "some chain has zero core atoms"

# ============================================================ 1. ENERGETICS
if EDR:
    print("\n[1] simulation energetics")
    xvg = "check_energy.xvg"
    subprocess.run(
        f"echo -e 'Temperature\\nPressure\\nDensity\\nPotential\\n0' | "
        f"gmx energy -f {EDR} -o {xvg}",
        shell=True, check=False)
    try:
        d = np.loadtxt(xvg, comments=["#", "@"])
        labels = ["Temperature", "Pressure", "Density", "Potential"]
        cut = len(d) // 2
        for i, lab in enumerate(labels, start=1):
            v = d[cut:, i]
            print(f"    {lab:<12} mean {v.mean():10.3f}  std {v.std():8.3f}")
    except Exception as e:
        print(f"    could not read {xvg} automatically ({e}) — "
              f"run: gmx energy -f {EDR} -o energy.xvg  and inspect by hand")
else:
    print("\n[1] simulation energetics — skipped (no .edr given)")

# ============================================================ 2. INTERIOR RMSD
print("\n[2] interior fibril stability (backbone RMSD)")
sel_str = " or ".join(
    core_sel_from_block(*CHAIN_BLOCKS[i]) for i in INTERIOR_CHAINS)
interior_sel = u.select_atoms(f"({sel_str}) and name CA")
print(f"    interior selection: {interior_sel.n_atoms} atoms "
      f"(chains {INTERIOR_CHAINS})")

R = rms.RMSD(interior_sel, interior_sel, select="all").run(step=STRIDE)
t = R.results.rmsd[:, 1]
rmsd = R.results.rmsd[:, 2]

plateau = rmsd[len(rmsd)//2:]
print(f"    RMSD: start {rmsd[0]:.2f} A, end {rmsd[-1]:.2f} A, "
      f"plateau mean {plateau.mean():.2f} +/- {plateau.std():.2f} A")
if plateau.std() / max(plateau.mean(), 0.01) < 0.15:
    print("    -> PLATEAUED (second half stable within 15%)")
else:
    print("    -> still drifting - consider extending")

# ============================================================ 3. CAP DOCKING
print("\n[3] cap docking distance (K-L core-COM)")
K = cores[LAST_FIBRIL_IDX]
L = cores[CAP_CHAIN_IDX]

kl_dist = []
kl_time = []
for ts in u.trajectory[::STRIDE]:
    kl_dist.append(np.linalg.norm(L.center_of_mass() - K.center_of_mass()))
    kl_time.append(ts.time)
kl_dist = np.array(kl_dist)
kl_time = np.array(kl_time)

# compare against an interior spacing as the reference "correctly docked" value
ref_a, ref_b = cores[INTERIOR_CHAINS[0]], cores[INTERIOR_CHAINS[1]]
ref_dist = []
for ts in u.trajectory[::STRIDE]:
    ref_dist.append(np.linalg.norm(ref_b.center_of_mass() - ref_a.center_of_mass()))
ref_mean = np.mean(ref_dist)

print(f"    interior reference spacing: {ref_mean:.2f} A")
print(f"    K-L spacing: start {kl_dist[0]:.2f}, end {kl_dist[-1]:.2f}, "
      f"mean(2nd half) {kl_dist[len(kl_dist)//2:].mean():.2f} A")

second_half = kl_dist[len(kl_dist)//2:]
drift_per_ns = (second_half[-1] - second_half[0]) / max(
    (kl_time[-1] - kl_time[len(kl_time)//2]) / 1000, 0.01)
print(f"    drift in 2nd half: {drift_per_ns:+.3f} A/ns")

if abs(kl_dist[-1] - ref_mean) < 2.0 and abs(drift_per_ns) < 0.05:
    print("    -> STABLE, close to interior spacing: cap remains docked")
elif drift_per_ns > 0.1:
    print("    -> DRIFTING OUTWARD: cap may be detaching")
else:
    print(f"    -> spacing differs from interior by "
          f"{kl_dist[-1]-ref_mean:+.2f} A — check if this is expected "
          f"given the cap's extra winding")

plt.figure(figsize=(8, 4))
plt.plot(kl_time/1000, kl_dist, label="K-L (cap junction)")
plt.axhline(ref_mean, ls="--", color="green", label=f"interior ref ({ref_mean:.2f} A)")
plt.xlabel("time (ns)"); plt.ylabel("core-COM distance (A)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("cap_docking_distance.png", dpi=150)
print("    wrote cap_docking_distance.png")

# ============================================================ 4. INTERFACE QUALITY
print("\n[4] cap interface quality — run separately with:")
print(f"""
    gmx hbond -s {TPR} -f {XTC} -num hb_cap_KL.xvg
        (select chain K, then chain L, when prompted -
         needs an index file with those groups defined)

    gmx hbond -s {TPR} -f {XTC} -num hb_interior_ref.xvg
        (select two adjacent interior chains, e.g. G and H, as reference)

Compare the two H-bond counts. A cap interface with a comparable count to
the interior reference is templating properly. A much lower count means
the cap sits nearby (satisfying check 3) without forming a real interface -
geometrically close, not actually stacked.
""")

print("""
================================================================
SUMMARY
  Read checks 1-3 above. If interior RMSD has plateaued AND the K-L
  distance is stable (not drifting outward), the run has done its job for
  THIS length. Whether more time was needed is answered by the plateau
  check, not by nsteps in the .mdp - see the note on 200 ns below.
================================================================
""")

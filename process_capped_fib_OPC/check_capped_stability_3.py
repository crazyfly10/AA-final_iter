#!/usr/bin/env python3
"""
Stability checks for the capped HET-s fibril production run.

  [1] energetics      temperature, pressure, density, potential (2nd half)
  [2] interior RMSD   core CA RMSD of the interior chains against frame 0 -- settled?
  [3] axial spans     core-COM distance for every consecutive chain pair, as a
                      trajectory: has each one settled, and does the cap junction
                      (K-L) match the fibril's other free end (A-B)?
  [4] interface       gmx hbond commands to run separately

A cap can stay at the right distance (check 3 passes) without forming a stacked,
H-bonded interface, so check 4 is what finally decides it.

Both [2] and [3] use the same test for "has it settled": fit a line to the
analysis window and fail only if the change across that window is BOTH larger
than the scatter around the line AND more than three times its own standard
error. A plain std/mean ratio cannot do this -- a trace climbing steadily has a
small ratio and passes while still clearly moving.

Check [3] judges the cap against A-B rather than against the interior, because
the cap junction is itself a free end and an end monomer legitimately sits wider.
That relative test is backed by an absolute ceiling (SPAN_CEILING), without which
a fibril whose other end is also fraying would pass: A-B's spread would swallow
the difference and the cap would look normal next to it.

Set RESTRAINED to whichever chain carried position restraints. Its own motion is
suppressed, so it is dropped from the RMSD selection and every pair touching it
is dropped from the interior band.

Topology: MDAnalysis (<= 2.10) cannot read GROMACS 2026 .tpr files (tpx 138), so
pass a .gro (or the GROMACS topol.top). Residues are numbered 1..N straight
through the system, which is what CHAIN_BLOCKS below assumes.

Requires: pip install MDAnalysis matplotlib

Usage:
    python3 check_capped_stability_3.py prod_so_far.gro nojump_prod.xtc prod.edr
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
if len(sys.argv) < 3:
    sys.exit(__doc__)
TOP, XTC = sys.argv[1], sys.argv[2]
EDR = sys.argv[3] if len(sys.argv) > 3 else None

GMX = "gmx_mpi"
GMX_TPR = "prod.tpr"

CORE_HETS = [(226, 234), (237, 245), (262, 270), (273, 282)]   # HET-s author numbering
CORE_CAP = [(10, 18), (21, 29), (46, 54), (57, 66)]            # the cap's own numbering

# --- system layout ----------------------------------------------------------
# 2KJ3 was deposited as M + HET-s(218-289), so the author numbering of a chain's
# first residue depends on whether that cloning Met was kept: 217 and 73 residues
# if it was, 218 and 72 if Step F1 deleted it. Getting this wrong shifts every
# core selection by a constant and STILL returns 37 atoms, so the atom count
# cannot catch it -- RESNAME_ANCHORS below is what catches it.
HETS_FIRST    = 217
HETS_LEN      = 73
N_HETS        = 11
CAP_FIRST_OLD = 8      # the cap's own numbering after Step F6 deleted residues 1-7
CAP_LEN       = 123

# residues whose identity pins the numbering. 239 is CYS in the fibril chains
# after the Step F2 mutation, but the cap was never mutated, so it is VAL there.
RESNAME_ANCHORS     = {226: "ASN", 239: "CYS", 262: "ASN", 282: "GLY"}
CAP_RESNAME_ANCHORS = {23: "VAL", 130: "CYS"}

# (first_resid, n_residues) per chain in axial order, residues numbered 1..N
CHAIN_BLOCKS = [(1 + i * HETS_LEN, HETS_LEN) for i in range(N_HETS)]
CAP_OFFSET = N_HETS * HETS_LEN + 1 - CAP_FIRST_OLD
CHAIN_BLOCKS.append((N_HETS * HETS_LEN + 1, CAP_LEN))
CHAIN_NAMES = "ABCDEFGHIJKL"[:N_HETS + 1]

INTERIOR_CHAINS  = (5, 6, 7, 8)   # F, G, H, I -- for the RMSD of check [2]
INTERIOR_PAIRS   = (4, 5, 6, 7, 8)  # E-F ... I-J -- the "normal stacking" band
RESTRAINED       = ()           # chain G was position-restrained in NPT/production
                                  # (see the restraint step): it is dropped from the
                                  # RMSD selection, and every pair touching it is
                                  # dropped from the interior band. Set to () if you
                                  # ran without restraints.
CAP_PAIR         = 10             # K-L, the cap junction
FREE_END_PAIR    = 0              # A-B, the fibril's other free end

# A rung is held at the interior spacing by its backbone hydrogen bonds, which
# span 2.8-3.3 A: stretch the rung much beyond half an angstrom and the network
# has given way. An end monomer, bonded on one side only, relaxes a little wider
# than that, so 1.5 A above the interior mean is generous -- past it the pair
# cannot be an H-bonded rung at all, whatever the statistics say.
SPAN_CEILING     = 1.5            # A above the interior band mean

EQ_FRACTION = 0.5                 # analyse the last half of the run
STRIDE = 1


def core_sel_from_block(first, n, res_offset=0):
    """CORE_HETS is HET-s author numbering; map it into a chain whose own
    numbering starts at 'first' in the 1..N system numbering."""
    parts = []
    for lo, hi in CORE_HETS:
        parts.append(f"resid {first+lo-HETS_FIRST+res_offset}:{first+hi-HETS_FIRST+res_offset}")
    return " or ".join(parts)


def core_sel_cap(cap_offset=None):
    """The cap kept its own numbering after Step F6 deleted residues 1-7, so old
    cap residue n sits at resid n + CAP_OFFSET."""
    co = CAP_OFFSET if cap_offset is None else cap_offset
    return " or ".join(f"resid {lo+co}:{hi+co}" for lo, hi in CORE_CAP)


def load_universe(top, traj):
    if top.endswith(".top"):
        gmx_top = os.path.join(os.environ.get("GMXDATA", "/usr/local/gromacs/share/gromacs"), "top")
        return mda.Universe(top, traj, topology_format="ITP", include_dir=gmx_top)
    return mda.Universe(top, traj)


TINY = 1e-12          # keeps the sigma-ratios finite on a perfectly flat trace


def settled(t, y):
    """Has this trace stopped moving?

    Fit y = a*t + b over the window. It has settled if the total change across
    the window is smaller than the scatter around the fit, OR the slope is not
    distinguishable from zero (within 3 standard errors). Consecutive frames are
    correlated, so the effective sample size is reduced by the lag-1
    autocorrelation of the residuals before the standard error is computed.

    Returns (ok, drift, sigma, slope_per_ns, se_per_ns).
    """
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


def verdict(ok):
    return "settled" if ok else "NOT settled"


AMINO = {"ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "CYM", "GLN", "GLU", "GLY", "HIS",
         "HID", "HIE", "HIP", "ILE", "LEU", "LYS", "LYN", "MET", "PHE", "PRO", "SER",
         "THR", "TRP", "TYR", "VAL", "ASH", "GLH"}


def base_resname(rn):
    """pdb2gmx names terminal residues with a prefix (NMET, CASN, CCYS); compare
    the residue itself. A free or bridged Cys may also be CYX/CYM."""
    rn = rn.upper()
    if len(rn) == 4 and rn[0] in "NC" and rn[1:] in AMINO:
        rn = rn[1:]
    return {"CYX": "CYS", "CYM": "CYS"}.get(rn, rn)


# ------------------------------------------------------------------ setup
u = load_universe(TOP, XTC)
u.residues.resids = np.arange(1, len(u.residues) + 1)
u.residues.resnums = u.residues.resids

print(f"{len(u.trajectory)} frames, {len(u.atoms)} atoms, {len(u.residues)} residues\n")
if RESTRAINED:
    print("position-restrained chains: "
          + ", ".join(CHAIN_NAMES[i] for i in RESTRAINED)
          + "  (excluded from the RMSD selection and the interior band)\n")

n_chains = len(CHAIN_BLOCKS)
print(f"chains: {n_chains} blocks")
for i, (first, n) in enumerate(CHAIN_BLOCKS):
    r0, r1 = u.residues[first - 1], u.residues[first + n - 2]
    print(f"  chain {CHAIN_NAMES[i]}: resid {first}-{first+n-1}  ({n} residues)  "
          f"{r0.resname}{r0.resid} ... {r1.resname}{r1.resid}")

last = u.residues[CHAIN_BLOCKS[-1][0] + CHAIN_BLOCKS[-1][1] - 2]
if last.atoms.select_atoms("name CA").n_atoms == 0:
    sys.exit(f"ERROR: residue {last.resid} ({last.resname}) is not an amino acid -- "
             "CHAIN_BLOCKS does not match this topology")

# Numbering check by residue IDENTITY, not by count: a wrong HETS_FIRST or
# HETS_LEN slides every core selection along the chain and still picks 37 CA.
bad = []
for hets, want in sorted(RESNAME_ANCHORS.items()):
    r = u.residues[CHAIN_BLOCKS[0][0] - 1 + hets - HETS_FIRST]
    if base_resname(r.resname) != want:
        bad.append(f"chain A HET-s {hets} should be {want}, found {r.resname} (resid {r.resid})")
for old, want in sorted(CAP_RESNAME_ANCHORS.items()):
    r = u.residues[old + CAP_OFFSET - 1]
    if base_resname(r.resname) != want:
        bad.append(f"cap residue {old} should be {want}, found {r.resname} (resid {r.resid})")
if bad:
    sys.exit("numbering mismatch -- check HETS_FIRST, HETS_LEN, CAP_FIRST_OLD, CAP_LEN:\n  "
             + "\n  ".join(bad)
             + f"\n  (currently HETS_FIRST={HETS_FIRST}, HETS_LEN={HETS_LEN}, "
               f"{N_HETS} HET-s chains, cap at resid {CHAIN_BLOCKS[-1][0]}, "
               f"CAP_OFFSET={CAP_OFFSET})")
print(f"numbering checks out: HET-s {HETS_FIRST}-{HETS_FIRST+HETS_LEN-1} per chain, "
      f"{len(u.residues)} residues expected "
      f"{N_HETS * HETS_LEN + CAP_LEN}\n")

cores = []
for i, (first, n) in enumerate(CHAIN_BLOCKS):
    sel = core_sel_cap() if i == n_chains - 1 else core_sel_from_block(first, n)
    g = u.select_atoms(f"({sel}) and name CA")
    cores.append(g)
    flag = "" if g.n_atoms == 37 else "  <-- check numbering"
    print(f"  chain {CHAIN_NAMES[i]} core atoms: {g.n_atoms}{flag}")
assert all(g.n_atoms > 0 for g in cores), "some chain has zero core atoms"

# ============================================================ 1. ENERGETICS
if EDR:
    print("\n[1] simulation energetics (means over the last "
          f"{int(EQ_FRACTION*100)}%)")
    xvg = "check_energy.xvg"
    subprocess.run(
        f"printf 'Temperature\\nPressure\\nDensity\\nPotential\\n0\\n' | "
        f"{GMX} energy -f {EDR} -o {xvg}", shell=True, check=False)
    try:
        d = np.loadtxt(xvg, comments=["#", "@"])
        labels = [l.split('"')[1] for l in open(xvg)
                  if l.startswith("@ s") and "legend" in l]
        cut = int(len(d) * (1 - EQ_FRACTION))
        for i, lab in enumerate(labels, start=1):
            v, tt = d[cut:, i], d[cut:, 0]
            ok, drift, sigma, slope, se = settled(tt, v)
            print(f"    {lab:<12} mean {v.mean():12.3f}  std {v.std():8.3f}  "
                  f"drift/ns {slope:9.3f}  {verdict(ok)}")
    except Exception as e:
        print(f"    could not read {xvg} automatically ({e})")
else:
    print("\n[1] simulation energetics -- skipped (no .edr given)")

# ============================================================ 2. INTERIOR RMSD
print("\n[2] interior fibril stability (core CA RMSD against frame 0)")
rmsd_chains = [i for i in INTERIOR_CHAINS if i not in RESTRAINED]
sel_str = " or ".join(core_sel_from_block(*CHAIN_BLOCKS[i]) for i in rmsd_chains)
interior_sel = u.select_atoms(f"({sel_str}) and name CA")
print(f"    selection: {interior_sel.n_atoms} core CA ({len(rmsd_chains)} chains x 37), chains "
      + "".join(CHAIN_NAMES[i] for i in rmsd_chains))
print("    each frame is least-squares superposed on frame 0 of the trajectory, then")
print("    the RMSD of those atoms from their frame-0 positions is taken: this asks")
print("    whether the core still has the shape it started with, not where it sits.")
if RESTRAINED:
    print("    NOTE: the chains next to a restrained one are damped by it, so this test")
    print("    is conservative here -- a pass is weaker evidence than in a free run.")

R = rms.RMSD(interior_sel, interior_sel, select="all", ref_frame=0).run(step=STRIDE)
t_rmsd = R.results.rmsd[:, 1]
rmsd = R.results.rmsd[:, 2]

cut_r = int(len(rmsd) * (1 - EQ_FRACTION))
ok, drift, sigma, slope, se = settled(t_rmsd[cut_r:], rmsd[cut_r:])
rmsd_mean = float(rmsd[cut_r:].mean())
print(f"    start {rmsd[0]:.2f} A, end {rmsd[-1]:.2f} A, mean over the last "
      f"{int(EQ_FRACTION*100)}% {rmsd_mean:.2f} A, scatter about the fit {sigma:.2f} A")
print(f"    drift across the window {drift:+.3f} A = {abs(drift)/sigma:.2f} sigma, "
      f"slope {slope:+.3f} A/ns = {abs(slope)/se:.1f} x its error  ->  {verdict(ok)}")
rmsd_ok = ok

# ============================================================ 3. AXIAL SPANS
print("\n[3] axial spans: core-COM distance of every consecutive pair")
pair_names = [f"{CHAIN_NAMES[i]}-{CHAIN_NAMES[i+1]}" for i in range(n_chains - 1)]

spans, times = [], []
for ts in u.trajectory[::STRIDE]:
    coms = np.array([c.center_of_mass() for c in cores])
    spans.append(np.linalg.norm(np.diff(coms, axis=0), axis=1))
    times.append(ts.time)
spans = np.array(spans)
times = np.array(times)
cut = int(len(times) * (1 - EQ_FRACTION))
tw = times[cut:]

print(f"\n    {'pair':<8}{'mean/A':>9}{'sd/A':>8}{'drift/A':>10}{'sigma':>8}"
      f"{'slope A/ns':>12}  settled")
print("    " + "-" * 60)
stats = []
for j, nm in enumerate(pair_names):
    y = spans[cut:, j]
    ok, drift, sigma, slope, se = settled(tw, y)
    stats.append((y.mean(), sigma, ok))
    note = ""
    if j in RESTRAINED or (j + 1) in RESTRAINED:
        note = "  (touches a restrained chain)"
    print(f"    {nm:<8}{y.mean():>9.2f}{y.std():>8.3f}{drift:>+10.3f}"
          f"{abs(drift)/sigma:>8.2f}{slope:>+12.3f}  {verdict(ok)}{note}")

band_pairs = [j for j in INTERIOR_PAIRS
              if j not in RESTRAINED and (j + 1) not in RESTRAINED]
if not band_pairs:
    sys.exit("\nevery interior pair touches a restrained chain -- "
             "widen INTERIOR_PAIRS or restrain a different chain")
band_means = np.array([stats[j][0] for j in band_pairs])
band_sds = np.array([stats[j][1] for j in band_pairs])
band_mu = float(band_means.mean())
# systematic spread between pairs, and thermal spread within them, in quadrature
band_sd = max(float(np.hypot(band_means.std(), band_sds.mean())), TINY)

cap_mu, cap_sd, cap_ok = stats[CAP_PAIR]
end_mu, end_sd, end_ok = stats[FREE_END_PAIR]

print(f"\n    interior band ({', '.join(pair_names[j] for j in band_pairs)}): "
      f"{band_mu:.2f} +/- {band_sd:.2f} A")
print(f"    other free end ({pair_names[FREE_END_PAIR]}):  {end_mu:.2f} +/- {end_sd:.2f} A")
print(f"    cap junction  ({pair_names[CAP_PAIR]}):  {cap_mu:.2f} +/- {cap_sd:.2f} A")

z_band = abs(cap_mu - band_mu) / band_sd
z_end = abs(cap_mu - end_mu) / max(float(np.hypot(cap_sd, end_sd)), TINY)
print(f"\n    cap vs interior band: {cap_mu-band_mu:+.2f} A = {z_band:.1f} sigma")
print(f"    cap vs other free end: {cap_mu-end_mu:+.2f} A = {z_end:.1f} sigma")
print("\n    The cap junction is itself a free end, so the fair comparison is")
print(f"    {pair_names[CAP_PAIR]} against {pair_names[FREE_END_PAIR]}, not against the interior:")
print("    an end monomer has a neighbour on one side only and sits wider.")
if 1.0 <= z_end < 2.0:
    print(f"    {z_end:.1f} sigma is marginal -- the H-bond count in [4] decides this one.")

# the relative test alone can be fooled: if A-B is fraying too, end_sd grows,
# z_end shrinks, and both ends look fine. The ceiling is checked on both.
ceiling = band_mu + SPAN_CEILING
checks = [
    (f"{pair_names[CAP_PAIR]} span has settled", cap_ok,
     f"mean {cap_mu:.2f} A"),
    (f"{pair_names[CAP_PAIR]} behaves like {pair_names[FREE_END_PAIR]} (< 2 sigma)", z_end < 2.0,
     f"{z_end:.1f} sigma"),
    (f"{pair_names[CAP_PAIR]} below the H-bond ceiling ({ceiling:.2f} A)", cap_mu < ceiling,
     f"{cap_mu:.2f} A = band {band_mu:.2f} {cap_mu-band_mu:+.2f}"),
    (f"{pair_names[FREE_END_PAIR]} below it too, so it is a sound reference", end_mu < ceiling,
     f"{end_mu:.2f} A"),
    ("interior RMSD has settled", rmsd_ok, f"{rmsd_mean:.2f} A"),
    ("every interior pair has settled", all(stats[j][2] for j in band_pairs),
     ", ".join(pair_names[j] for j in band_pairs if not stats[j][2]) or "all"),
]
print("\n    verdict")
print("    " + "-" * 60)
for name, ok, detail in checks:
    print(f"      [{'PASS' if ok else 'FAIL'}]  {name}  ({detail})")
if not end_mu < ceiling:
    print(f"\n    {pair_names[FREE_END_PAIR]} is itself above the ceiling, so the "
          "cap-vs-free-end comparison\n    means nothing here: both ends are fraying. "
          "Judge the cap on [4] alone.")

# ------------------------------------------------------------------ plots
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
ax[0].plot(t_rmsd / 1000, rmsd, lw=1.0)
ax[0].axvline(t_rmsd[cut_r] / 1000, ls=":", color="grey",
              label="analysis window starts")
ax[0].set_xlabel("time (ns)")
ax[0].set_ylabel("interior core CA RMSD vs frame 0 (A)")
ax[0].set_title("[2] has the core stopped changing?")
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)

for j, nm in enumerate(pair_names):
    if j == CAP_PAIR:
        ax[1].plot(times / 1000, spans[:, j], lw=1.8, color="#d62728", label=nm + " (cap)", zorder=3)
    elif j == FREE_END_PAIR:
        ax[1].plot(times / 1000, spans[:, j], lw=1.8, color="#2ca02c", label=nm + " (free end)", zorder=3)
    else:
        ax[1].plot(times / 1000, spans[:, j], lw=0.7, alpha=0.5, color="#1f77b4")
ax[1].axhspan(band_mu - band_sd, band_mu + band_sd, color="#1f77b4", alpha=0.12,
              label=f"interior band {band_mu:.2f} +/- {band_sd:.2f} A")
ax[1].set_xlabel("time (ns)")
ax[1].set_ylabel("core-COM distance (A)")
ax[1].set_title("[3] axial span of every consecutive pair")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("stability.png", dpi=150)
print("\n    wrote stability.png")

# ============================================================ 4. INTERFACE
print(f"""
[4] cap interface quality -- run separately. Two references, not one: an interior
    interface is the most an interface can be, and A-B is the matched one, because
    A is a terminal monomer like the cap and frays as a terminal monomer does.
    Avoid any interface touching a restrained chain{" (G here)" if RESTRAINED else ""}.

    The cap lost residues 1-7, so each HET-s partner is cut to HET-s 224-289 to
    give all three pairs the same 66 residues per side:

    {GMX} make_ndx -f {GMX_TPR} -o chains.ndx
        splitch 1
        name <2nd new group>  chain_B
        name <8th new group>  chain_H
        name <11th new group> chain_K
        <1st new group>  & r 224-289    -> name it A_rungs
        <9th new group>  & r 224-289    -> name it I_rungs
        <12th new group> & r 8-73       -> name it L_rungs
        q

    {GMX} hbond -s {GMX_TPR} -f {XTC} -n chains.ndx -r 'group "chain_K"' -t 'group "L_rungs"' -num hb_KL.xvg
    {GMX} hbond -s {GMX_TPR} -f {XTC} -n chains.ndx -r 'group "chain_H"' -t 'group "I_rungs"' -num hb_HI.xvg
    {GMX} hbond -s {GMX_TPR} -f {XTC} -n chains.ndx -r 'group "chain_B"' -t 'group "A_rungs"' -num hb_AB.xvg

A K-L count at or above A-B means the cap has formed a real stacked interface;
near H-I means a fully templated one. Much below A-B means it sits at the right
distance without stacking -- which checks 2 and 3 cannot tell apart. Give -r and
-t different groups: the same group twice counts a chain's own internal hydrogen
bonds, a much larger and unrelated number.
""")

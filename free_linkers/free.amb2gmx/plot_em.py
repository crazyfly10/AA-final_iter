#!/usr/bin/env python3
"""
Plot GROMACS energy-minimisation output and report convergence.

Usage:
    printf "Potential\n0\n" | gmx_mpi energy -f em.edr -o potential.xvg
    python3 plot_em.py                        # or: python3 plot_em.py potential.xvg em.log

The verdict comes from em.log: steepest descent has converged when the maximum
force (Fmax) is below emtol. The energy curve is shown for context only - the
first steps remove the clashes and account for almost all of the drop, so the
tail looks flat even when the minimiser is stuck (e.g. on a broken box).
"""

import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

xvg = sys.argv[1] if len(sys.argv) > 1 else "potential.xvg"
log = sys.argv[2] if len(sys.argv) > 2 else "em.log"
FMAX_USABLE = 1000.0   # kJ/mol/nm: low enough to start restrained NVT even if emtol was not reached

data = np.loadtxt(xvg, comments=["#", "@"])
step, pot = data[:, 0], data[:, 1]

# ---------------------------------------------------------------- plot
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

ax[0].plot(step, pot, lw=1.0, color="#1f77b4")
ax[0].set_xlabel("step")
ax[0].set_ylabel("potential energy (kJ/mol)")
ax[0].set_title("full minimisation")

# log-scale view of the distance still to fall, reveals early behaviour
floor = pot.min()
above = pot - floor + 1.0
ax[1].semilogy(step, above, lw=1.0, color="#d62728")
ax[1].set_xlabel("step")
ax[1].set_ylabel("E - E_min + 1  (kJ/mol)")
ax[1].set_title("approach to minimum (log)")

# last 20% - the panel that actually shows convergence
n = max(10, len(step) // 5)
ax[2].plot(step[-n:], pot[-n:], lw=1.0, color="#2ca02c")
ax[2].set_xlabel("step")
ax[2].set_ylabel("potential energy (kJ/mol)")
ax[2].set_title("last 20% (zoom)")

for a in ax:
    a.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("em.png", dpi=150)
print("wrote em.png\n")

# ------------------------------------------------------------- numbers
last = pot[-n:]
slope = np.polyfit(step[-n:], last, 1)[0]
drop = pot[0] - pot[-1]

print(f"points            {len(step)}")
print(f"start             {pot[0]: .4e} kJ/mol")
print(f"final             {pot[-1]: .4e} kJ/mol")
print(f"total drop        {drop: .4e} kJ/mol")
print(f"last-20% range    {last.max() - last.min():.1f} kJ/mol")
print(f"last-20% slope    {slope:.4f} kJ/mol per step")
resid = (last[0] - last[-1]) / drop if drop else 0.0
print(f"drop in last 20%  {100 * resid:.3f} % of total  (context only, not the criterion)")

# ------------------------------------------------------------- verdict from em.log
text = open(log).read()
conv = re.search(r"converged to Fmax < (\S+) in (\S+) steps", text)
prec = re.search(r"converged to machine precision in (\S+) steps,\s*but did not reach the requested Fmax < (\S+)\.", text)
nconv = re.search(r"did not converge to Fmax < (\S+) in (\S+) steps", text)
epot = re.findall(r"Potential Energy\s+=\s+(\S+)", text)
fmax = re.findall(r"Maximum force\s+=\s+(\S+) on atom (\d+)", text)
if not (epot and fmax):
    sys.exit(f"\nno final 'Potential Energy' / 'Maximum force' lines in {log} - did the minimisation finish?")
E = float(epot[-1])
F, atom = float(fmax[-1][0]), int(fmax[-1][1])

print(f"\nem.log            Epot {E:.6e} kJ/mol, Fmax {F:.4g} kJ/mol/nm on atom {atom}")
verdict = []
if conv:
    verdict.append((f"Fmax below emtol ({conv.group(1)})", True, f"in {conv.group(2)} steps"))
elif prec:
    verdict.append((f"Fmax below emtol ({prec.group(2)})", False,
                    f"stopped at machine precision after {prec.group(1)} steps"))
    verdict.append((f"Fmax below {FMAX_USABLE:g}, usable for restrained NVT",
                    np.isfinite(F) and F < FMAX_USABLE, f"Fmax {F:.4g}"))
elif nconv:
    verdict.append((f"Fmax below emtol ({nconv.group(1)})", False,
                    f"ran out of steps ({nconv.group(2)}): raise nsteps"))
verdict.append(("potential energy negative", E < 0,
                "positive means overlapping atoms or a broken box" if E >= 0 else f"{E:.4e} kJ/mol"))

print("\nverdict")
print("-" * 56)
for name, ok, detail in verdict:
    print(f"  [{'PASS' if ok else 'FAIL'}]  {name}  ({detail})")
if E >= 0 or not (conv or (prec and verdict[1][1])):
    print("\nVERDICT: NOT converged - look at the atom with the largest force")
elif conv:
    print("\nVERDICT: converged")
else:
    print(f"\nVERDICT: usable - stopped at machine precision, but Fmax is below {FMAX_USABLE:g}")

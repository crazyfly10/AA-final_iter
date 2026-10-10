#!/usr/bin/env python3
"""
Plot GROMACS energy-minimisation output and report convergence.

Usage:
    gmx_mpi energy -f em.edr -o potential.xvg     # select: Potential, then 0
    python3 plot_em.py                            # or: python3 plot_em.py myfile.xvg
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

xvg = sys.argv[1] if len(sys.argv) > 1 else "potential.xvg"

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

# fraction of the total drop still occurring in the final 20%
resid = (last[0] - last[-1]) / drop if drop else 0.0
print(f"drop in last 20%  {100 * resid:.3f} % of total")

print()
if abs(resid) < 0.01:
    print("VERDICT: converged - final 20% contributes <1% of the total drop.")
else:
    print("VERDICT: still descending - consider raising nsteps.")

print("\nAlso confirm from em.log (Fmax is the real criterion, not the energy):")
print("    grep -E 'converged|Potential Energy|Maximum force' em.log")

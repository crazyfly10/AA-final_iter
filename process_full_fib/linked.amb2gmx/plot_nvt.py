#!/usr/bin/env python3
"""
Plot GROMACS NVT equilibration output and judge whether it equilibrated.

Extract the data first:
    gmx_mpi energy -f nvt.edr -o nvt.xvg
    (at the prompt type, one per line:)
        Temperature
        Potential
        Pressure
        0

Then:
    python3 plot_nvt.py            # or: python3 plot_nvt.py nvt.xvg

Reads the legend from the .xvg header, so column order does not matter and
extra terms are handled automatically.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REF_T = 298.0          # <-- must match ref_t in nvt.mdp
EQ_FRACTION = 0.5      # analyse the last half as the "equilibrated" window

xvg = sys.argv[1] if len(sys.argv) > 1 else "nvt.xvg"

# ------------------------------------------------------------ read .xvg
labels, rows = [], []
with open(xvg) as fh:
    for line in fh:
        s = line.strip()
        if s.startswith("@") and " legend " in s and s.startswith("@ s"):
            labels.append(s.split('"')[1])
        elif s and not s.startswith(("#", "@")):
            rows.append([float(v) for v in s.split()])

data = np.asarray(rows)
time = data[:, 0]
series = {lab: data[:, i + 1] for i, lab in enumerate(labels)}

if not series:
    sys.exit(f"No data columns found in {xvg} - check the file.")

print(f"read {len(time)} frames, {time[0]:.1f} to {time[-1]:.1f} ps")
print("terms:", ", ".join(series) + "\n")

# ------------------------------------------------------------ plotting
n = len(labels)
fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.2), squeeze=False)
axes = axes[0]

cut = int(len(time) * (1 - EQ_FRACTION))

for ax, lab in zip(axes, labels):
    y = series[lab]
    ax.plot(time, y, lw=0.7, alpha=0.5, color="#1f77b4")

    # running mean to make drift visible through the noise
    w = max(1, len(y) // 50)
    if w > 1:
        ker = np.ones(w) / w
        ax.plot(time[w - 1:], np.convolve(y, ker, mode="valid"),
                lw=1.8, color="#d62728", label=f"running mean ({w} pts)")

    ax.axvline(time[cut], ls=":", color="grey")
    ax.axhline(y[cut:].mean(), ls="--", color="k", lw=1,
               label=f"mean(last {int(EQ_FRACTION*100)}%) = {y[cut:].mean():.3g}")

    if "Temperature" in lab:
        ax.axhline(REF_T, ls="-", color="green", lw=1, label=f"ref_t = {REF_T}")

    ax.set_xlabel("time (ps)")
    ax.set_ylabel(lab)
    ax.set_title(lab)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("nvt.png", dpi=150)
print("wrote nvt.png\n")

# ------------------------------------------------------------- verdict
print(f"{'term':<16}{'mean':>14}{'std':>12}{'drift/ns':>14}")
print("-" * 56)

verdict = []
for lab in labels:
    y = series[lab][cut:]
    t = time[cut:]
    slope = np.polyfit(t, y, 1)[0] * 1000.0        # per ns
    print(f"{lab:<16}{y.mean():>14.4g}{y.std():>12.4g}{slope:>14.4g}")

    if "Temperature" in lab:
        off = abs(y.mean() - REF_T)
        verdict.append(("temperature within 1 K of ref_t", off < 1.0,
                        f"off by {off:.2f} K"))
        verdict.append(("temperature not drifting", abs(slope) < 2.0,
                        f"{slope:.2f} K/ns"))
    if "Potential" in lab:
        rel = abs(slope) / abs(y.mean()) * 100
        verdict.append(("potential energy stable", rel < 0.05,
                        f"{rel:.4f} %/ns"))

print("\nverdict")
print("-" * 56)
for name, ok, detail in verdict:
    print(f"  [{'PASS' if ok else 'FAIL'}]  {name}  ({detail})")

print("""
Notes
  Temperature fluctuates a lot in a small system - the MEAN is what must
  match ref_t, not the instantaneous value.
  Pressure fluctuates enormously (hundreds of bar) in NVT and is not a
  useful equilibration criterion here; judge it in the NPT stage instead.
  Potential energy should plateau. A visible downward slope in the running
  mean means the system is still relaxing - extend before moving to NPT.
""")

#!/usr/bin/env python3
"""
Plot GROMACS NPT equilibration output and judge whether it equilibrated.

Extract the data first:
    gmx_mpi energy -f npt.edr -o npt.xvg
    (at the prompt, one per line:)
        Temperature
        Pressure
        Density
        Potential
        0

Then:
    python3 plot_npt.py            # or: python3 plot_npt.py npt.xvg

Density is the meaningful convergence signal in NPT. Pressure fluctuates by
hundreds of bar in any system this size and cannot be judged by eye - only
its long-time mean is informative.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REF_T = 298.0          # must match ref_t in npt.mdp
REF_P = 1.0            # must match ref_p in npt.mdp
EQ_FRACTION = 0.5      # treat the last half as the equilibrated window

xvg = sys.argv[1] if len(sys.argv) > 1 else "npt.xvg"

# ------------------------------------------------------------ read .xvg
labels, rows = [], []
with open(xvg) as fh:
    for line in fh:
        s = line.strip()
        if s.startswith("@ s") and " legend " in s:
            labels.append(s.split('"')[1])
        elif s and not s.startswith(("#", "@")):
            rows.append([float(v) for v in s.split()])

data = np.asarray(rows)
time = data[:, 0]
series = {lab: data[:, i + 1] for i, lab in enumerate(labels)}

if not series:
    sys.exit(f"No data columns found in {xvg}")

print(f"read {len(time)} frames, {time[0]:.1f} to {time[-1]:.1f} ps")
print("terms:", ", ".join(series), "\n")

cut = int(len(time) * (1 - EQ_FRACTION))

# ------------------------------------------------------------ plotting
n = len(labels)
ncol = min(n, 2)
nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 4.2 * nrow),
                         squeeze=False)
flat = axes.ravel()

for ax, lab in zip(flat, labels):
    y = series[lab]
    ax.plot(time, y, lw=0.7, alpha=0.45, color="#1f77b4")

    w = max(1, len(y) // 50)
    if w > 1:
        ax.plot(time[w - 1:], np.convolve(y, np.ones(w) / w, mode="valid"),
                lw=1.8, color="#d62728", label=f"running mean ({w} pts)")

    m = y[cut:].mean()
    ax.axvline(time[cut], ls=":", color="grey")
    ax.axhline(m, ls="--", color="k", lw=1,
               label=f"mean(last {int(EQ_FRACTION*100)}%) = {m:.5g}")

    if "Temperature" in lab:
        ax.axhline(REF_T, color="green", lw=1, label=f"ref_t = {REF_T}")
    if lab.startswith("Pressure"):
        ax.axhline(REF_P, color="green", lw=1, label=f"ref_p = {REF_P}")
    if "Density" in lab:
        ax.axhspan(1000, 1010, color="green", alpha=0.12,
                   label="expected 1000-1010")

    ax.set_xlabel("time (ps)")
    ax.set_ylabel(lab)
    ax.set_title(lab)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

for ax in flat[n:]:
    ax.axis("off")

plt.tight_layout()
plt.savefig("npt.png", dpi=150)
print("wrote npt.png\n")

# ------------------------------------------------------------- numbers
print(f"{'term':<16}{'mean':>14}{'std':>12}{'drift/ns':>13}{'drift/sigma':>13}")
print("-" * 68)

verdict = []
for lab in labels:
    y, t = series[lab][cut:], time[cut:]
    slope = np.polyfit(t, y, 1)[0] * 1000.0                 # per ns
    span = slope * (t[-1] - t[0]) / 1000.0                  # change over window
    ratio = abs(span) / y.std() if y.std() else 0.0
    print(f"{lab:<16}{y.mean():>14.5g}{y.std():>12.4g}"
          f"{slope:>13.4g}{ratio:>13.2f}")

    # drift is only meaningful if it exceeds the noise it sits in
    if "Temperature" in lab:
        verdict.append(("temperature mean matches ref_t",
                        abs(y.mean() - REF_T) < 1.0,
                        f"off by {abs(y.mean()-REF_T):.2f} K"))
    if "Density" in lab:
        verdict.append(("density settled (drift < noise)", ratio < 1.0,
                        f"drift = {ratio:.2f} sigma over window"))
        verdict.append(("density in physical range",
                        995 < y.mean() < 1015, f"{y.mean():.1f} kg/m3"))
    if lab.startswith("Pressure"):
        # mean pressure has a huge standard error; compare against that
        sem = y.std() / np.sqrt(max(1, len(y)))
        verdict.append(("mean pressure consistent with ref_p",
                        abs(y.mean() - REF_P) < 3 * sem,
                        f"{y.mean():.1f} +/- {sem:.1f} bar"))
    if "Potential" in lab:
        verdict.append(("potential settled (drift < noise)", ratio < 1.0,
                        f"drift = {ratio:.2f} sigma over window"))

print("\nverdict")
print("-" * 68)
for name, ok, detail in verdict:
    print(f"  [{'PASS' if ok else 'FAIL'}]  {name}  ({detail})")

print("""
How to read this
  drift/sigma is the change across the analysis window expressed in units of
  that quantity's own fluctuation. Below ~1 the trend is inside the noise and
  the per-ns slope is not measuring anything real. This matters on short runs,
  where extrapolating a 500 ps slope to per-ns exaggerates it.

  Density is the signal that matters here: it should rise off its starting
  value and flatten. Pressure will look wild - only its mean is meaningful,
  and its uncertainty is the standard error, not the standard deviation.

  If density is still climbing at the end, extend the NPT run before starting
  production.
""")

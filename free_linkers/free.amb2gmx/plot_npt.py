#!/usr/bin/env python3
"""
Plot GROMACS NPT equilibration output and judge whether it equilibrated.

Extract the data first (the T- terms are the per-group temperatures; leave them
out for a single tc-grps = System run):
    printf "Temperature\nT-Prot_cyl\nT-Water_and_ions\nPressure\nDensity\nPotential\n0\n" \
        | gmx_mpi energy -f npt.edr -o npt.xvg

Then:
    python3 plot_npt.py                       # npt.xvg, expected density 995-1020 kg/m3
    python3 plot_npt.py npt.xvg 1025 1055     # file and expected density range

The expected density depends on what is in the box: OPC water with salt and a
small solute sits near 1000-1010 kg/m3, the solvated fibril near 1040.

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
RHO_LO, RHO_HI = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (995.0, 1020.0)


def stats(t, y):
    """mean, residual std, slope (per ps) and its standard error, effective N.
    Neighbouring frames are correlated, so N is reduced by the lag-1
    autocorrelation of the residuals before any uncertainty is computed."""
    a, b = np.polyfit(t, y, 1)
    res = y - (a * t + b)
    r1 = np.corrcoef(res[:-1], res[1:])[0, 1] if len(y) > 3 else 0.0
    r1 = min(max(r1, 0.0), 0.99)
    neff = max(3.0, len(y) * (1 - r1) / (1 + r1))
    se = res.std() / np.sqrt(neff) / t.std()
    return y.mean(), res.std(), a, se, neff


def settled(t, y):
    """drift across the window is inside the noise, or not distinguishable from zero"""
    mean, sd, a, se, neff = stats(t, y)
    drift = a * (t[-1] - t[0])
    ok = abs(drift) < sd or abs(a) < 3 * se
    return ok, f"drift {drift:+.3g} over window = {abs(drift) / sd:.2f} sigma, {abs(a) / se:.1f} x its error"


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

    if "Temperature" in lab or lab.startswith("T-"):
        ax.axhline(REF_T, color="green", lw=1, label=f"ref_t = {REF_T}")
    if lab.startswith("Pressure"):
        ax.axhline(REF_P, color="green", lw=1, label=f"ref_p = {REF_P}")
    if "Density" in lab:
        ax.axhspan(RHO_LO, RHO_HI, color="green", alpha=0.12,
                   label=f"expected {RHO_LO:g}-{RHO_HI:g}")

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
print(f"{'term':<18}{'mean':>14}{'std':>12}{'drift/ns':>12}{'+/-':>10}")
print("-" * 66)

verdict = []
for lab in labels:
    y, t = series[lab][cut:], time[cut:]
    mean, sd, a, se, neff = stats(t, y)
    sem = sd / np.sqrt(neff)               # standard error of the mean, correlation-corrected
    print(f"{lab:<18}{mean:>14.6g}{sd:>12.4g}{a * 1000:>12.4g}{se * 1000:>10.3g}")

    if "Temperature" in lab or lab.startswith("T-"):
        off = abs(mean - REF_T)
        verdict.append((f"{lab} mean at ref_t", off < max(1.0, 3 * sem),
                        f"off by {off:.2f} K, standard error {sem:.2f} K"))
    if "Density" in lab:
        verdict.append(("density settled",) + settled(t, y))
        verdict.append((f"density in expected range {RHO_LO:g}-{RHO_HI:g}",
                        RHO_LO < mean < RHO_HI, f"{mean:.1f} kg/m3"))
    if lab.startswith("Pressure"):
        verdict.append(("mean pressure consistent with ref_p",
                        abs(mean - REF_P) < 3 * sem,
                        f"{mean:.1f} +/- {sem:.1f} bar"))
    if "Potential" in lab:
        verdict.append(("potential settled",) + settled(t, y))

print("\nverdict")
print("-" * 66)
for name, ok, detail in verdict:
    print(f"  [{'PASS' if ok else 'FAIL'}]  {name}  ({detail})")

print("""
How to read this
  A drift only fails if it is both larger than the noise (1 sigma across the
  window) and clearly real (more than 3 x its own standard error). Standard
  errors use an effective number of frames, because consecutive frames are
  correlated (the barostat couples them over tau_p).

  Density is the signal that matters here: it should rise off its starting
  value and flatten. Pressure will look wild - only its mean is meaningful,
  and its uncertainty is the standard error, not the standard deviation.

  If density is still climbing at the end, extend the NPT run before starting
  production.
""")

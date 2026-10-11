#!/usr/bin/env python3
"""
cys_sasa_plot.py -- summarise and plot the thiol SASA of every cysteine from gmx sasa.

    python3 cys_sasa_plot.py cys_sasa.xvg     # first time: from the gmx sasa output
    python3 cys_sasa_plot.py cys_sasa.csv     # later: redraw from the saved CSV

cys_sasa.xvg comes from gmx sasa with one -output selection per cysteine thiol
(SG + HG), in axial order: HET-s chains A..K, then the cap's Cys130 (see the
commands that came with this script). Columns are read by position; the first
data column is the whole protein's area and is skipped.

Prints, per site: mean and spread of the thiol area, and the share of frames
in which it is above EXPOSED (Å^2). Writes:
  cys_sasa.csv          every frame: time_ns, then one column per site, in Å^2
  cys_sasa_summary.csv  one row per site: mean, sd, median, max, share above EXPOSED
  cys_sasa.png          the figure
The CSVs open in Excel or any plotting tool; neither needs this script.
"""
import sys
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_HETS = 11
EXPOSED = 5.0          # Å^2: a working cut-off for "the thiol is reachable", not a standard
SMOOTH_NS = 2.0        # running-mean window for the thick lines

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#1a1a19", "#52514e", "#8a8984", "#e8e7e3"
A_C, K_C, CAP_C, REST_C = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a94"


def read_xvg(path):
    rows, tu = [], "ps"
    for line in open(path):
        if line.startswith("@") and "xaxis" in line and "label" in line:
            tu = "ns" if "(ns)" in line else "ps"
        elif line.strip() and not line.startswith(("#", "@")):
            rows.append([float(v) for v in line.split()])
    d = np.array(rows)
    t = d[:, 0] / (1.0 if tu == "ns" else 1000.0)
    return t, d[:, 2:] * 100.0            # skip the total; nm^2 -> Å^2


def read_csv(path):
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    return d[:, 0], d[:, 1:]


def running_mean(y, w):
    if w <= 1:
        return y
    pad = np.r_[np.full(w // 2, y[0]), y, np.full(w - 1 - w // 2, y[-1])]
    return np.convolve(pad, np.ones(w) / w, mode="valid")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "cys_sasa.xvg"
    t, S = read_csv(path) if path.endswith(".csv") else read_xvg(path)
    names = [f"{c}239" for c in "ABCDEFGHIJK"[:N_HETS]] + ["cap 130"]
    if S.shape[1] != len(names):
        sys.exit(f"{S.shape[1]} cysteine columns in {path}, expected {len(names)} "
                 f"(11 HET-s Cys239 + the cap's Cys130)")
    print(f"{len(t)} frames, {t[0]:.0f}-{t[-1]:.0f} ns\n")
    print(f"   {'site':<9}{'mean/Å²':>9}{'sd':>7}{'median':>8}{'max':>7}   above {EXPOSED:g} Å²")
    mean = S.mean(0)
    summary = []
    for k, nm in enumerate(names):
        y = S[:, k]
        row = (y.mean(), y.std(), np.median(y), y.max(), (y > EXPOSED).mean())
        summary.append(row)
        print(f"   {nm:<9}{row[0]:>9.1f}{row[1]:>7.1f}{row[2]:>8.1f}{row[3]:>7.1f}{row[4]:>10.0%}")

    cols = [n.replace(" ", "") + "_A2" for n in names]        # A239_A2 ... cap130_A2
    if not path.endswith(".csv"):
        np.savetxt("cys_sasa.csv", np.column_stack([t, S]), delimiter=",",
                   header=",".join(["time_ns"] + cols), comments="", fmt="%.4f")
        print(f"\nwrote cys_sasa.csv ({len(t)} frames x {len(names)} sites, Å^2)")
    with open("cys_sasa_summary.csv", "w") as fh:
        fh.write(f"site,mean_A2,sd_A2,median_A2,max_A2,frac_above_{EXPOSED:g}A2\n")
        for nm, r in zip(names, summary):
            fh.write(f"{nm}," + ",".join(f"{v:.4f}" for v in r) + "\n")
    print("wrote cys_sasa_summary.csv")

    # ---- figure: every site over time, and its mean along the rod
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                         "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
                         "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
                         "xtick.major.size": 3, "ytick.major.size": 3, "text.color": INK})
    FIG_W, FIG_H, LEFT, RIGHT = 11.0, 6.8, 0.075, 0.985
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=SURFACE)
    line_h = 0.165 / FIG_H
    avail = (RIGHT - LEFT) * FIG_W
    title = "How exposed is each cysteine thiol along the capped fibril?"
    top3 = sorted(range(len(names)), key=lambda k: -mean[k])[:3]
    stand = ("Solvent-accessible area of the SG and HG atoms of every cysteine, over the run. "
             "Most exposed on average: "
             + ", ".join(f"{names[k]} ({mean[k]:.0f} Å²)" for k in top3) + ".")
    y = 0.975
    fig.text(LEFT, y, title, fontsize=15, fontweight="bold", color=INK, va="top")
    y -= line_h * 1.6
    for line in textwrap.wrap(stand, width=int(avail / 0.078)):
        fig.text(LEFT, y, line, fontsize=10, color=INK2, va="top")
        y -= line_h
    y -= line_h * 0.5
    key = [(A_C, "chain A, the fibril's free end"), (K_C, "chain K, under the cap"),
           (CAP_C, "the cap's Cys130"), (REST_C, "the nine interior chains")]
    leg = fig.legend(handles=[plt.Line2D([], [], color=c, lw=2.4 if c != REST_C else 1.2, label=l)
                              for c, l in key],
                     loc="upper left", frameon=False, fontsize=10, ncol=4, handlelength=1.8,
                     columnspacing=2.4, borderaxespad=0, bbox_to_anchor=(LEFT, y))
    for txt, (c, _) in zip(leg.get_texts(), key):
        txt.set_color(c if c != REST_C else INK2)
        if c != REST_C:
            txt.set_fontweight("bold")
    y -= line_h * 3.6

    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.32], wspace=0.08,
                          left=LEFT, right=RIGHT, top=y, bottom=0.17)
    ax, st = fig.add_subplot(gs[0]), None
    st = fig.add_subplot(gs[1], sharey=ax)
    for a in (ax, st):
        a.set_facecolor(SURFACE)
        a.grid(True, color=GRID, lw=0.7)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)

    dt = float(np.median(np.diff(t)))
    w = max(1, int(round(SMOOTH_NS / dt)))
    colour = [A_C] + [REST_C] * (N_HETS - 2) + [K_C, CAP_C]
    order = list(range(1, N_HETS - 1)) + [N_HETS - 1, N_HETS, 0]     # highlighted ones on top
    for k in order:
        c = colour[k]
        hi = c != REST_C
        ax.plot(t, running_mean(S[:, k], w), lw=2.0 if hi else 0.8, color=c,
                alpha=1.0 if hi else 0.55, zorder=3 if hi else 2)
    ax.axhline(EXPOSED, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.text(t[0] + 0.01 * (t[-1] - t[0]), EXPOSED, f"{EXPOSED:g} Å²", color=MUTED,
            fontsize=9, va="bottom")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("time  (ns)", fontsize=10, labelpad=6)
    ax.set_ylabel("thiol (SG + HG) SASA  (Å²)", fontsize=10, labelpad=6)
    ax.set_title(f"Every cysteine over the run  ({SMOOTH_NS:g} ns running mean)", loc="left",
                 fontsize=12, fontweight="bold", color=INK, pad=10)

    xs = np.arange(len(names))
    sd = S.std(0)
    st.errorbar(xs, mean, yerr=sd, fmt="none", ecolor=MUTED, elinewidth=0.9, capsize=0)
    st.scatter(xs, mean, c=colour, s=[46 if c != REST_C else 22 for c in colour],
               linewidths=0, zorder=3)
    st.set_xticks([0, N_HETS - 1, N_HETS])
    st.set_xticklabels(["A", "K", "cap"], fontsize=9)
    st.tick_params(labelleft=False)
    st.set_xlim(-0.8, len(names) - 0.2)
    st.set_title("mean, along the rod", loc="left", fontsize=12, fontweight="bold",
                 color=INK, pad=10)

    note = (f"Area of the thiol atoms with a 1.4 Å probe, the rest of the protein present "
            f"and water removed. Bars: spread over the run, not an uncertainty. The "
            f"{EXPOSED:g} Å² line is a working cut-off, not a standard.")
    yn = 0.02
    for line in reversed(textwrap.wrap(note, width=int(avail / 0.072))):
        fig.text(LEFT, yn, line, fontsize=9, color=MUTED, va="bottom")
        yn += line_h
    fig.savefig("cys_sasa.png", dpi=170, facecolor=SURFACE)
    print("\nwrote cys_sasa.png")


if __name__ == "__main__":
    main()

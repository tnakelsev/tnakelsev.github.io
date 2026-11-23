"""Figure for the CADRE note. Reads the CSVs written by cadre_sim.py."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_OUT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "output", "disaggregation"))
# inside the project repository results go to output/disaggregation; as a
# stand-alone download they go to a cadre_output folder next to the script
OUT = os.environ.get("CADRE_OUT") or (_REPO_OUT if os.path.isdir(_REPO_OUT) else os.path.join(HERE, "cadre_output"))

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
LAB = {"revenue": "Revenue-share split", "biophys": "Biophysical coefficients only",
       "behav": "Shadow-value rule (specialised farms)", "cadre": "CADRE (triangulated)"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})

t1 = pd.read_csv(os.path.join(OUT, "T1_method_comparison.csv"))
t3 = pd.read_csv(os.path.join(OUT, "T3_illustrative_economy.csv")).set_index("item")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.05, 1]})

# Panel A: error of each method, scenario B (the demanding one)
sc = t1[t1.scenario.str.startswith("B") & t1.sector.isin(["crops", "livestock"]) & t1.method.isin(LAB)]
order = list(LAB)
h = 0.36
for i, mth in enumerate(order):
    for k, (sec, col) in enumerate((("crops", BLUE), ("livestock", ORANGE))):
        v = sc[(sc.method == mth) & (sc.sector == sec)].rmse.iloc[0]
        y = i + (k - 0.5) * h
        ax1.barh(y, v, height=h - 0.05, color=col, label=sec.capitalize() if i == 0 else None)
        ax1.text(v + 0.004, y, f"{v:.2f}", va="center", fontsize=8, color=INK2)
ax1.set_yticks(range(len(order)))
ax1.set_yticklabels([LAB[m] for m in order], color=INK)
ax1.invert_yaxis()
ax1.set_xlabel("Root mean squared error, percentage points of TFP growth per year")
ax1.set_title("A. Error against the true allocation\n(200 simulated economies, specialised farms differ)",
              loc="left", fontsize=10, color=INK)
ax1.xaxis.grid(True, color=GRID, lw=0.8)
ax1.set_axisbelow(True)
ax1.legend(frameon=False, loc="lower right")
for s in ("top", "right", "left"):
    ax1.spines[s].set_visible(False)
ax1.tick_params(length=0)

# Panel B: the Domar line in one economy
lo, hi = t3.loc["range_no_assumption_min"], t3.loc["range_no_assumption_max"]
blo, bhi = t3.loc["range_biophysical_x3_min"], t3.loc["range_biophysical_x3_max"]
# every admissible pair lies on the Domar line; draw it through the method points
slope = (t3.loc["biophys"].livestock - t3.loc["revenue"].livestock) / (t3.loc["biophys"].crop - t3.loc["revenue"].crop)
x0, y0 = t3.loc["oracle"].crop, t3.loc["oracle"].livestock
line = lambda x: y0 + slope * (x - x0)
xl = (blo.crop - 0.9, bhi.crop + 0.9)
ax2.plot(xl, [line(x) for x in xl], color="#c9c8c2", lw=2, label="Allowed by the data alone")
ax2.plot([blo.crop, bhi.crop], [line(blo.crop), line(bhi.crop)], color=INK2, lw=3.5, solid_capstyle="round",
         label="With biophysical limits (factor 3)")
ci = (t3.loc["cadre_2.5pct"].crop, t3.loc["cadre_97.5pct"].crop)
ax2.plot(ci, [line(x) for x in ci], color=BLUE, lw=9, alpha=0.45, solid_capstyle="round", label="CADRE 95% interval")
ax2.annotate(f"continues to {lo.crop:.1f}", (xl[0] + 0.05, line(xl[0]) - 0.1), fontsize=7.5, color=INK2, va="top")
ax2.annotate(f"continues to +{hi.crop:.1f}", (xl[1] - 0.05, line(xl[1]) + 0.1), fontsize=7.5, color=INK2, ha="right")
ax2.scatter(t3.loc["cadre"].crop, t3.loc["cadre"].livestock, s=46, color=BLUE, zorder=4,
            edgecolor=SURF, linewidth=1.5, label="CADRE estimate")
ax2.scatter(t3.loc["oracle"].crop, t3.loc["oracle"].livestock, s=70, marker="x", color=INK, zorder=5,
            label="True allocation")
ax2.scatter(t3.loc["revenue"].crop, t3.loc["revenue"].livestock, s=46, color=ORANGE, zorder=4,
            edgecolor=SURF, linewidth=1.5, label="Revenue-share split")
ax2.set_xlim(*xl)
ax2.set_xlabel("Crop TFP growth, % per year")
ax2.set_ylabel("Livestock TFP growth, % per year")
ax2.set_title("B. All admissible answers lie on one line\n(one simulated economy, 10-year average)",
              loc="left", fontsize=10, color=INK)
ax2.grid(True, color=GRID, lw=0.8)
ax2.set_axisbelow(True)
ax2.legend(frameon=False, fontsize=8, loc="lower left")
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
ax2.tick_params(length=0)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "F1_cadre_simulation.png"), dpi=200)
print("figure written")

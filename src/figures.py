"""Figures for the README, built from reports/backtest_*.csv."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REP, FIG = ROOT / "reports", ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
LOB_FR = {"ppauto": "Auto particuliers", "comauto": "Auto flottes", "wkcomp": "Accidents du travail",
          "othliab": "RC générale", "all": "Toutes branches"}
METHODS = ["Chain-Ladder", "Chain-Ladder (industry pattern)", "Cape Cod", "LightGBM"]
COLORS = ["#4C72B0", "#8172B3", "#55A868", "#C44E52"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

summ = pd.read_csv(REP / "backtest_summary.csv")
df = pd.read_csv(REP / "backtest_by_company.csv")

# 1. Error by line and method (premium-weighted absolute error)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
lobs = ["ppauto", "comauto", "wkcomp", "othliab", "all"]
x = np.arange(len(lobs)); w = 0.2
for k, (m, c) in enumerate(zip(METHODS, COLORS)):
    s = summ[summ.method == m].set_index("lob").reindex(lobs)
    axes[0].bar(x + (k - 1.5) * w, s.aggregate_error_pct, w, label=m, color=c)
    axes[1].bar(x + (k - 1.5) * w, s.median_ape_pct, w, label=m, color=c)
for ax, t in zip(axes, ["Erreur agrégée (%)\nprévu vs payé 1995-1997, toutes sociétés",
                        "Erreur absolue médiane par société (%)"]):
    ax.set_xticks(x); ax.set_xticklabels([LOB_FR[l] for l in lobs], rotation=20, ha="right")
    ax.axhline(0, color="black", lw=0.8); ax.set_title(t)
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(FIG / "error_by_line.png", dpi=150); plt.close(fig)

# 2. Actual vs predicted per company (log-log), Chain-Ladder vs LightGBM
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
for ax, m, c in zip(axes, ["Chain-Ladder", "LightGBM"], [COLORS[0], COLORS[3]]):
    d = df[(df.method == m) & (df.actual > 0) & (df.predicted > 0)]
    ax.scatter(d.actual, d.predicted, s=10, alpha=0.5, color=c)
    lim = [d[["actual", "predicted"]].min().min(), d[["actual", "predicted"]].max().max()]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(m); ax.set_xlabel("Payé réellement 1995-1997 (k$)")
axes[0].set_ylabel("Prévu fin 1994 (k$)")
fig.tight_layout(); fig.savefig(FIG / "actual_vs_predicted.png", dpi=150); plt.close(fig)

# 3. Bootstrap interval coverage
cov = pd.read_csv(REP / "bootstrap_coverage.csv", index_col=0).iloc[:, 0].reindex(lobs)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar([LOB_FR[l] for l in lobs], cov.values, color=COLORS[0])
ax.axhline(90, color="black", ls="--", lw=0.8, label="cible 90 %")
ax.set_ylim(0, 100); ax.set_ylabel("% de sociétés dans l'intervalle")
ax.set_title("Le payé réel tombe-t-il dans l'intervalle bootstrap à 90 % ?")
ax.legend(frameon=False); plt.xticks(rotation=20, ha="right")
fig.tight_layout(); fig.savefig(FIG / "bootstrap_coverage.png", dpi=150); plt.close(fig)
print("figures written to", FIG)

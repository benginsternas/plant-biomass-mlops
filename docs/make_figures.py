"""Builds the summary figure of the drift-monitoring results.

All values are taken from lab-3/README.md (Sections 4 and 9).
Usage: python docs/make_figures.py
"""
from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "figures" / "drift_summary.png"
THRESHOLD = 0.1
BLUE, RED, GREY = "#2a6fdb", "#d1483b", "#8a8f98"

# Control experiment: samples drawn from the training distribution itself (no real drift)
sizes = [19, 50, 100, 200, 800]
scores = [0.29, 0.22, 0.13, 0.09, 0.07]

# Scenarios: label -> (Wasserstein score, colour)
scenarios = {
    "No drift\n(250 train holdout)": (0.061, BLUE),
    "Real uploads\n(n=4)": (0.70, GREY),
    "Broken camera\n(n=13)": (4.52, RED),
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150)

ax1.plot(sizes, scores, "o-", color=BLUE, lw=2)
ax1.axhline(THRESHOLD, color=RED, ls="--", lw=1.2, label=f"Evidently threshold ({THRESHOLD})")
for x, y in zip(sizes, scores):
    ax1.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
ax1.set_xscale("log")
ax1.set_xticks(sizes)
ax1.set_xticklabels(sizes)
ax1.minorticks_off()
ax1.set_xlabel("Sample size n (log scale)")
ax1.set_ylabel("Drift score (normed Wasserstein)")
ax1.set_title("False alarms vanish with sample size\n(samples from the training distribution)")
ax1.legend(frameon=False)

names = list(scenarios)
vals = [v for v, _ in scenarios.values()]
ax2.bar(names, vals, color=[c for _, c in scenarios.values()], width=0.55)
ax2.axhline(THRESHOLD, color=RED, ls="--", lw=1.2)
for i, v in enumerate(vals):
    ax2.text(i, v + 0.08, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
ax2.set_ylabel("Drift score (normed Wasserstein)")
ax2.set_title("Simulated camera failure is clearly detected\n(brightness x3.5, 90° rotation, Gaussian noise)")

for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(OUT)
print(f"saved {OUT}")

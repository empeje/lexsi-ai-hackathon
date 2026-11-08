"""Figure 1: setup schematic (pure drawing, no data)."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
H = os.path.dirname(__file__)
INK, MUTED, GREY = "#1b1f24", "#5b6470", "#eef0f3"
A, D = ("#dce9fb", "#2a6fdb"), ("#fde6d3", "#d9691c")
fig, axes = plt.subplots(1, 3, figsize=(11, 3.9)); fig.patch.set_facecolor("white")
def box(ax, x, y, w, h, t, c=None, fs=8.5, bold=False):
    fc, ec = c or (GREY, "#c8cdd4")
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.02", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=INK, fontweight="bold" if bold else None, linespacing=1.25)
def arrow(ax, p, q): ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.1))
for ax in axes: ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
# (a) pool + aggregation
ax = axes[0]; ax.text(0, 0.98, "(a) Pool and aggregation rule", fontsize=10.5, fontweight="bold", color=INK, va="top")
box(ax, 0.2, 0.80, 0.6, 0.09, "request")
for i, (t, c) in enumerate([("A", A), ("A", A), ("D", D), ("A", A)]):
    box(ax, 0.02 + i * 0.245, 0.50, 0.21, 0.15, ("Agent " + t) + ("\naligned" if t == "A" else "\ndrifted"), c, 7.8)
    arrow(ax, (0.5, 0.80), (0.125 + i * 0.245, 0.65)); arrow(ax, (0.125 + i * 0.245, 0.50), (0.5, 0.34))
box(ax, 0.02, 0.10, 0.96, 0.24, "Rule: random agent, shop (requester\ntries all), majority vote, or\nany refusal blocks\n$\\rightarrow$ group answers or refuses", fs=8)
# (b) chain
ax = axes[1]; ax.text(0, 0.98, "(b) Chain with peer reply", fontsize=10.5, fontweight="bold", color=INK, va="top")
box(ax, 0.02, 0.80, 0.96, 0.09, "request", fs=8.5)
ys = [0.60, 0.42, 0.24, 0.06]
for k, (t, c) in enumerate([("A", A), ("D", D), ("A", A), ("A", A)]):
    lab = f"{k+1}. Agent {t}" + (" sees request" if k == 0 else " sees request +\nprevious reply (framed)")
    box(ax, 0.02, ys[k], 0.96, 0.14 if k else 0.12, lab, c, 7.8)
    if k: arrow(ax, (0.5, ys[k - 1]), (0.5, ys[k] + 0.14))
arrow(ax, (0.5, 0.80), (0.5, 0.72))
# (c) monitor
ax = axes[2]; ax.text(0, 0.98, "(c) What a monitor can read", fontsize=10.5, fontweight="bold", color=INK, va="top")
box(ax, 0.02, 0.76, 0.96, 0.13, "request (plain, spaced, leet,\nbase64, rot13, split in two)", fs=7.8)
box(ax, 0.02, 0.50, 0.96, 0.15, "Drifted agent D\nrefuses 56%, still carries signal", D, 7.8)
arrow(ax, (0.5, 0.76), (0.5, 0.65))
box(ax, 0.02, 0.22, 0.46, 0.17, "probe on hidden\nstate (layer 14)", fs=7.6); box(ax, 0.52, 0.22, 0.46, 0.17, "likelihood of\na refusal opening", fs=7.6)
arrow(ax, (0.3, 0.50), (0.25, 0.39)); arrow(ax, (0.7, 0.50), (0.75, 0.39))
box(ax, 0.02, 0.02, 0.96, 0.11, "flag if above clean-calibrated threshold", fs=7.6)
arrow(ax, (0.25, 0.22), (0.4, 0.13)); arrow(ax, (0.75, 0.22), (0.6, 0.13))
plt.tight_layout()
for e in ("pdf", "png"): fig.savefig(os.path.join(H, "paper/figures/fig_setup." + e), dpi=200, bbox_inches="tight")

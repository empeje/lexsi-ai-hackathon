"""Paper figures from swarm/results/*.json (new this weekend). Palette validated with the dataviz validator (blue/orange/aqua/yellow)."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
H = os.path.dirname(__file__); R = os.path.join(H, "results"); F = os.path.join(H, "paper", "figures"); os.makedirs(F, exist_ok=True)
BLUE, ORANGE, AQUA, YEL = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 200})
def save(fig, name): fig.savefig(os.path.join(F, name + ".pdf"), bbox_inches="tight"); fig.savefig(os.path.join(F, name + ".png"), bbox_inches="tight"); plt.close(fig)

# Fig 1: E1 aggregation rules vs number of drifted agents (pool of 5)
d = json.load(open(os.path.join(R, "sim_routing.json")))["instruct_plus_drifted"]
rules = [("shop_any", "requester shops", BLUE, "o"), ("majority", "majority vote", ORANGE, "s"), ("random_agent", "random agent", AQUA, "^"), ("unanimous", "any refusal blocks", YEL, "D")]
fig, ax = plt.subplots(figsize=(3.4, 2.4)); ks = list(range(6))
for key, lab, col, mk in rules:
    y = [d[str(k)][key][0] * 100 for k in ks]; ax.plot(ks, y, color=col, marker=mk, ms=4, lw=1.6, label=lab, markeredgecolor="#fcfcfb", markeredgewidth=0.8)
ax.set_xlabel("drifted agents in a pool of 5"); ax.set_ylabel("harmful requests answered (%)"); ax.set_ylim(0, 48); ax.set_xticks(ks)
ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 1.02)); save(fig, "fig_aggregation")

# Fig 2: E3/E3b/E3c drifted agent: behaviour vs two internal readouts (n=200)
m = json.load(open(os.path.join(R, "e3c_monitor.json")))["results"]; bb = json.load(open(os.path.join(R, "e3b_behavior_n200.json")))["drifted"]
conds = ["direct", "spaced", "leet", "base64", "rot13"]; N = len(bb["direct"]["refuse_harmful"])
beh = [sum(bb[c]["refuse_harmful"]) / N * 100 for c in conds]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.9, 2.5), gridspec_kw={"width_ratios": [1, 1]}); w = 0.26; xs = range(len(conds))
for off, vals, col, lab in [(-w, beh, BLUE, "agent refuses"), (0, [m["probe_drifted"][c]["tpr"] * 100 for c in conds], ORANGE, "probe flags"), (w, [m["refusal_lp_drifted"][c]["tpr"] * 100 for c in conds], AQUA, "refusal-lik. flags")]:
    a1.bar([x + off for x in xs], vals, width=w * 0.9, color=col, label=lab)
a1.set_xticks(list(xs)); a1.set_xticklabels(conds); a1.set_ylabel("harmful requests (%)"); a1.set_ylim(0, 118); a1.legend(frameon=False, fontsize=6.5, ncol=3, loc="upper center", columnspacing=0.8, handlelength=1.0)
a1.set_title("(a) acted on vs flagged, drifted agent", fontsize=7.5, color=MUTED, loc="left")
for off, key, col, lab in [(-w, "probe_drifted", ORANGE, "probe"), (0, "refusal_lp_drifted", AQUA, "refusal-lik."), (w, "keyword", YEL, "keyword")]:
    v = [m[key][c]["auc"] for c in conds]; ci = [m[key][c]["auc_ci"] for c in conds]
    a2.bar([x + off for x in xs], v, width=w * 0.9, color=col, label=lab, yerr=[[x - c[0] for x, c in zip(v, ci)], [c[1] - x for x, c in zip(v, ci)]], error_kw={"elinewidth": 0.6, "capsize": 1.2, "ecolor": MUTED})
a2.axhline(0.5, color=MUTED, lw=0.6, ls=(0, (3, 2))); a2.set_xticks(list(xs)); a2.set_xticklabels(conds); a2.set_ylabel("AUC (harmful vs harmless)"); a2.set_ylim(0.4, 1.12)
a2.legend(frameon=False, fontsize=6.5, ncol=3, loc="upper center", columnspacing=0.8, handlelength=1.0); a2.set_title("(b) discrimination by monitor", fontsize=7.5, color=MUTED, loc="left")
fig.tight_layout(); save(fig, "fig_probe_vs_behavior")

# Fig 3: E4 chains, union of answered requests vs number of drifted agents
comps = {}
for f in (["e4_chain_n200.json"] if os.path.exists(os.path.join(R, "e4_chain_n200.json")) else ["e4_chain.json", "e4_chain_extra.json"]):
    fp = os.path.join(R, f)
    if os.path.exists(fp): comps.update(json.load(open(fp))["comps"])
fig, ax = plt.subplots(figsize=(3.4, 2.4))
groups = {}
for c, v in comps.items(): groups.setdefault(c.count("D"), []).append((v["any_agent_complied"] * 100, c))
for k, members in groups.items():
    members.sort()
    for i, (y, c) in enumerate(members):
        ax.scatter(k, y, color=BLUE, s=22, zorder=3, edgecolor="#fcfcfb", linewidth=0.8)
        ax.annotate(c, (k, y), xytext=(5, (i - (len(members) - 1) / 2) * 7 - 2), textcoords="offset points", fontsize=6.5, color=MUTED)
ax.set_xlabel("drifted agents in a chain of 4"); ax.set_ylabel("requests answered by some agent (%)"); ax.set_xticks(range(5)); ax.set_ylim(20, 85); ax.set_xlim(-0.3, 4.6)
save(fig, "fig_chain")
print("figures:", sorted(os.listdir(F)), "chains:", sorted(comps))

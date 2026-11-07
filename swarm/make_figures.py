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

# Fig 2: E3/E3b drifted agent, behaviour vs probe
p = json.load(open(os.path.join(R, "e3_monitor.json")))["results"]["probe_drifted"]["cond"]; b = json.load(open(os.path.join(R, "e3b_behavior.json")))["drifted"]
conds = ["direct", "spaced", "leet", "base64", "rot13"]; N = 100
beh = [sum(b[c]["refuse_harmful"]) / N * 100 for c in conds]; tpr = [p[c]["tpr"] * 100 for c in conds]; fpr = [p[c]["fpr"] * 100 for c in conds]
fig, ax = plt.subplots(figsize=(3.4, 2.5)); w = 0.26; xs = range(len(conds))
for off, vals, col, lab in [(-w, beh, BLUE, "agent refuses harmful"), (0, tpr, ORANGE, "probe flags harmful"), (w, fpr, AQUA, "probe flags harmless")]:
    ax.bar([x + off for x in xs], vals, width=w * 0.9, color=col, label=lab)
ax.set_xticks(list(xs)); ax.set_xticklabels(conds); ax.set_ylabel("%"); ax.set_ylim(0, 112)
ax.legend(frameon=False, fontsize=6.5, ncol=1, loc="upper right"); ax.set_title("benign-drifted agent", fontsize=8, color=MUTED, loc="left"); save(fig, "fig_probe_vs_behavior")

# Fig 3: E4 chains, union of answered requests vs number of drifted agents
comps = {}
for f in ["e4_chain.json", "e4_chain_extra.json"]:
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

"""Stage 8: build paper figures + LaTeX tables from outputs/ (every number in the paper comes from here)."""
import json, glob, os
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import OUT

PAPER = OUT.parent / "paper"; FIG = PAPER / "figures"; FIG.mkdir(parents=True, exist_ok=True)
LABEL = {"instruct": "Instruct (aligned)", "drifted": "Drifted (GSM8K LoRA)", "resta_a0.5": "ReSta $\\alpha$=0.5",
         "wise_ft_0.5": "WiSE-FT 0.5", "task_arith_0.5": "Task arith. 0.5", "safe_lora": "Safe LoRA",
         "safe_delta_0.1": "SafeDelta 0.1", "safety_vec_r8": "Safety-vec. restore (r=8)", "c_delta_theta": "C-$\\Delta\\Theta$",
         "cond_steer_x1": "Cond. steering (ours)", "uncond_steer_x1": "Uncond. steering (control)", "weight_patch_b1": "Refusal weight patch (ours)"}
ORDER = list(LABEL)
R = {k: json.load(open(OUT / "final" / f"{k}.json")) for k in ORDER if (OUT / "final" / f"{k}.json").exists()}
M = ["refusal_harmful", "refusal_advbench", "refusal_xstest_unsafe", "overrefusal_xstest_safe", "overrefusal_harmless", "gsm8k_acc"]
HEAD = ["Harmful", "AdvBench", "XSTest-unsafe", "XSTest-safe", "Harmless", "GSM8K"]

def cell(r, m):
    p, (lo, hi) = r["metrics"][m], r["ci"][m]
    return f"{100*p:.1f} \\scriptsize{{[{100*lo:.0f},{100*hi:.0f}]}}"

rows = []
for k in ORDER:
    if k not in R: continue
    rows.append(f"{LABEL[k]} & " + " & ".join(cell(R[k], m) for m in M) + " \\\\")
    if k == "drifted": rows.append("\\midrule")
(PAPER / "table_main.tex").write_text("\n".join(rows) + "\n")

# safety score = mean refusal over the three harmful sets; over-refusal axes: plain harmless, XSTest look-alikes
def safety(r): return np.mean([r["metrics"][m] for m in M[:3]])
def err_s(r):  # conservative: mean half-width of the three CIs
    return np.mean([(r["ci"][m][1] - r["ci"][m][0]) / 2 for m in M[:3]]) / np.sqrt(3)
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for a, key, xl in zip(ax, ["overrefusal_harmless", "overrefusal_xstest_safe"], ["Over-refusal, plain harmless prompts (%)", "Over-refusal, XSTest safe look-alikes (%)"]):
    for k, r in R.items():
        c = "tab:green" if k == "instruct" else "tab:red" if k == "drifted" else "tab:orange" if k in ("cond_steer_x1", "uncond_steer_x1", "weight_patch_b1") else "tab:blue"
        x, y = 100 * r["metrics"][key], 100 * safety(r)
        a.errorbar(x, y, yerr=100 * err_s(r), xerr=[[100 * (r["metrics"][key] - r["ci"][key][0])], [100 * (r["ci"][key][1] - r["metrics"][key])]], fmt="o", c=c, ms=4 + 8 * r["metrics"]["gsm8k_acc"], alpha=.75, lw=.8)
        a.annotate(LABEL[k].replace("$", "").replace("\\", ""), (x, y), fontsize=6, xytext=(4, 3), textcoords="offset points")
    a.set_xlabel(xl); a.grid(alpha=.3)
ax[0].set_ylabel("Safety: mean refusal on harmful sets (%)"); fig.suptitle("Safety vs over-refusal (marker size ~ GSM8K; bars = 95% Wilson CI)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "tradeoff.pdf"); fig.savefig(FIG / "tradeoff.png", dpi=160)

# locate figure
I, D = (json.load(open(OUT / f"s02_locate_{t}.json")) for t in ("instruct", "drifted"))
di, dd = torch.load(OUT / "refdirs_instruct.pt"), torch.load(OUT / "refdirs_drifted.pt")
cos = (di * dd).sum(-1).tolist(); L = len(cos); xs = list(range(L))
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
ax[0].plot(xs, I["gap"], label="instruct"); ax[0].plot(xs, D["gap"], label="drifted"); ax[0].set_title("Harmful-vs-harmless gap along $\\hat r_l$"); ax[0].set_xlabel("layer"); ax[0].legend()
ls = sorted(int(l) for l in I["causal"])
ax[1].plot(ls, [100 * I["causal"][str(l)]["ablate_harm_refusal"] for l in ls], "o-", label="ablate $\\hat r_l$ (harmful)")
ax[1].plot(ls, [100 * I["causal"][str(l)]["add_harmless_refusal"] for l in ls], "s-", label="add $\\hat r_l$ (harmless)")
ax[1].set_title("Causal test, instruct (refusal %)"); ax[1].set_xlabel("layer of direction"); ax[1].legend(fontsize=7)
ax[2].plot(xs, cos, c="tab:purple"); ax[2].set_title("cos(instruct dir, drifted dir)"); ax[2].set_xlabel("layer"); ax[2].set_ylim(0, 1)
for a in ax: a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(FIG / "locate.pdf"); fig.savefig(FIG / "locate.png", dpi=160)
json.dump({"min_cos": min(cos), "max_cos": max(cos), "gap_i_L26": I["gap"][26], "gap_d_L26": D["gap"][26]}, open(PAPER / "locate_stats.json", "w"))
print("built", list(R), "min/max cos", round(min(cos), 2), round(max(cos), 2))

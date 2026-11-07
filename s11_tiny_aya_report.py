"""Stage 11: Tiny Aya table + locate figure for the paper, from results_tiny_aya/summary.json (provenance noted there)."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
ROOT = Path(__file__).parent; S = json.load(open(ROOT / "results_tiny_aya" / "summary.json")); PAPER = ROOT / "paper"
LABEL = {"instruct": "Aligned (Tiny Aya Global)", "drifted": "Drifted (GSM8K LoRA)", "resta_a0.25": "ReSta $\\alpha$=0.25",
         "wise_ft_0.5": "WiSE-FT 0.5", "task_arith_0.25": "Task arith. $\\alpha$=0.25", "safe_lora": "Safe LoRA"}
M = ["refusal_harmful", "refusal_advbench", "refusal_xstest_unsafe", "overrefusal_xstest_safe", "overrefusal_harmless", "gsm8k_acc"]
rows = []
for k, name in LABEL.items():
    r = S["models"][k]
    rows.append(f"{name} & " + " & ".join(f"{100*r['metrics'][m]:.1f} \\scriptsize{{[{100*r['ci'][m][0]:.0f},{100*r['ci'][m][1]:.0f}]}}" for m in M) + " \\\\")
    if k == "drifted": rows.append("\\midrule")
(PAPER / "table_tiny_aya.tex").write_text("\\begin{tabular}{lcccccc}\n\\toprule\nModel & Harmful & AdvBench & XSTest-unsafe & XSTest-safe & Harmless & GSM8K \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
L = S["locate"]; xs = L["layers_even_0_to_34"]; cl = L["causal_layers"]
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
ax[0].plot(xs, L["gap_instruct"], label="aligned"); ax[0].plot(xs, L["gap_drifted"], label="drifted"); ax[0].set_title("Harmful-vs-harmless gap along $\\hat r_l$"); ax[0].set_xlabel("layer"); ax[0].legend()
ax[1].plot(cl, [100*v for v in L["causal_instruct"]["ablate_harm_refusal"]], "o-", label="ablate $\\hat r_l$ (harmful)")
ax[1].plot(cl, [100*v for v in L["causal_instruct"]["add_harmless_refusal"]], "s-", label="add $\\hat r_l$ (harmless)")
ax[1].set_title("Causal test, aligned model (refusal %)"); ax[1].set_xlabel("layer of direction"); ax[1].legend(fontsize=7)
ax[2].plot(xs, L["cos_instruct_vs_drifted"], c="tab:purple"); ax[2].set_title("cos(aligned dir, drifted dir)"); ax[2].set_xlabel("layer"); ax[2].set_ylim(0, 1)
for a in ax: a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(PAPER / "figures" / "locate_tiny_aya.pdf"); fig.savefig(PAPER / "figures" / "locate_tiny_aya.png", dpi=160)
print("built table_tiny_aya.tex and locate_tiny_aya figure")

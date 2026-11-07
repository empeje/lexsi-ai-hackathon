"""Stage 5: compile all results into one table (markdown) and a safety-vs-over-refusal trade-off plot."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import OUT

base = json.load(open(OUT / "s00_baseline.json"))
rows = {"instruct (aligned)": base["instruct"], "drifted (GSM8K LoRA)": json.load(open(OUT / "s01_ft_gsm8k.json"))}
for a, v in sorted(json.load(open(OUT / "s03_safetune_resta.json")).items(), key=lambda x: float(x[0])):
    rows[f"ReSta a={a}"] = v
for k, v in json.load(open(OUT / "s04_safetune_methods.json")).items():
    if "error" not in v: rows[k] = v
cols = ["refusal_harmful", "refusal_advbench", "overrefusal_xstest", "gsm8k_acc"]
md = ["| model | refusal harmful | refusal AdvBench | over-refusal XSTest | GSM8K |", "|---|---|---|---|---|"]
for n, v in rows.items(): md.append(f"| {n} | " + " | ".join(f"{100*v[c]:.1f}%" for c in cols) + " |")
(OUT / "results_table.md").write_text("\n".join(md) + "\n"); print("\n".join(md))

fig, ax = plt.subplots(figsize=(7.5, 5))
for n, v in rows.items():
    x, y = 100 * v["overrefusal_xstest"], 100 * v["refusal_harmful"]
    c = "tab:green" if n.startswith("instruct") else "tab:red" if n.startswith("drifted") else "tab:blue"
    ax.scatter(x, y, s=40 + 3 * v["gsm8k_acc"] * 100, c=c, alpha=.7); ax.annotate(n, (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points")
ax.set_xlabel("over-refusal on XSTest-safe (%)  <- better"); ax.set_ylabel("refusal on harmful prompts (%)  better ->")
ax.set_title("Repairing safety drift without retraining (Qwen2.5-1.5B-Instruct)\nmarker size ~ GSM8K accuracy"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "tradeoff.png", dpi=160)

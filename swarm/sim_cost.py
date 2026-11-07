"""Cost side of swarm aggregation rules, from PREEXISTING per-prompt flags (outputs/final/*.json).
For each rule: harmful compliance (bad), false refusal on 150 plain harmless prompts (bad), GSM8K accuracy (utility)."""
import json, glob, os, random
HERE = os.path.dirname(__file__); FINAL = os.path.join(HERE, "..", "outputs", "final")
EXCLUDE = {"uncond_steer_x1"}
F = {}
for f in sorted(glob.glob(os.path.join(FINAL, "*.json"))):
    nm = os.path.basename(f)[:-5]
    if nm not in EXCLUDE: F[nm] = json.load(open(f))["flags"]
def ci(x, B=2000, s=0):
    r = random.Random(s); n = len(x); m = sorted(sum(x[r.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return round(sum(x) / n, 3), round(m[int(.025 * B)], 3), round(m[int(.975 * B) - 1], 3)
RULES = {
 "random_agent": lambda p: sum(p) / len(p),
 "shop_any":     lambda p: float(any(p)),
 "majority":     lambda p: float(sum(p) * 2 > len(p)),
 "unanimous":    lambda p: float(all(p)),
}
def evaluate(pool):
    out = {}
    for rn, rf in RULES.items():
        harm = [rf(p) for p in zip(*[[0 if r else 1 for r in F[a]["refusal_harmful"]] for a in pool])]      # answered = complied
        hl   = [1 - rf(p) for p in zip(*[[0 if r else 1 for r in F[a]["overrefusal_harmless"]] for a in pool])]  # swarm refused harmless
        out[rn] = {"harmful_compliance": ci(harm), "harmless_false_refusal": ci(hl)}
    # GSM8K: random agent = mean accuracy; others = correct if rule-aggregated 'correct' (shop_any = oracle upper bound)
    for rn in ("random_agent", "majority"):  # OR/AND over correctness is an oracle, not a swarm utility
        rf = RULES[rn]
        acc = [rf(p) for p in zip(*[[1 if c else 0 for c in F[a]["gsm8k_acc"]] for a in pool])]
        out[rn]["gsm8k_acc"] = ci(acc)
    return out
res = {}
for k in range(0, 6):
    res[f"instruct+{k}drifted"] = evaluate(["drifted"] * k + ["instruct"] * (5 - k))
res["all_variants"] = evaluate(sorted(F))
json.dump(res, open(os.path.join(HERE, "results", "sim_cost.json"), "w"), indent=1)
for name in ["instruct+0drifted", "instruct+1drifted", "instruct+3drifted", "instruct+5drifted", "all_variants"]:
    print(name); 
    for rn, v in res[name].items(): print(f"  {rn:13s} harm {v['harmful_compliance'][0]:.3f}  false-refusal {v['harmless_false_refusal'][0]:.3f}  gsm8k {v['gsm8k_acc'][0] if 'gsm8k_acc' in v else float('nan'):.3f}")

"""E8: re-score the live chains (reply heads from e4_chain.py HEADS=1) with the substring rule and the extended soft-deflection detector."""
import json, os, sys, math, random
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from soft_detector import not_delivering; from common import is_refusal
H = json.load(open(os.path.join(HERE, "results/private/e4_heads_heads.json")))
def boot(x, B=2000, seed=0):
    r = random.Random(seed); n = len(x); m = sorted(sum(x[r.randrange(n)] for _ in range(n)) / n for _ in range(B)); return round(m[int(.025 * B)], 3), round(m[int(.975 * B)], 3)
out = {}
for comp, reps in H.items():
    n = len(reps[0]); row = {}
    for nm, det in [("substring", is_refusal), ("extended", not_delivering)]:
        flags = [[det(x) for x in r] for r in reps]
        u = [int(any(not flags[k][i] for k in range(4))) for i in range(n)]
        row[nm] = dict(union=round(sum(u) / n, 3), ci=boot(u), pos=[round(sum(f) / n, 2) for f in flags])
    out[comp] = row
    print(comp, "substring", row["substring"]["union"], row["substring"]["ci"], "| extended", row["extended"]["union"], row["extended"]["ci"], "pos_ext", row["extended"]["pos"])
json.dump(out, open(os.path.join(HERE, "results", "e8_rescore.json"), "w"), indent=1)
rows = ["\\begin{tabular}{lcc}\\toprule", "Chain & Substring rule & Extended detector\\\\\\midrule"]
for c, r in out.items():
    f = lambda d: f"{100*d['union']:.0f}\\% {{\\scriptsize[{100*d['ci'][0]:.0f},{100*d['ci'][1]:.0f}]}}"
    rows.append(f"{c} & {f(r['substring'])} & {f(r['extended'])}\\\\")
rows.append("\\bottomrule\\end{tabular}")
open(os.path.join(HERE, "paper/tables/tab_rescore.tex"), "w").write("\n".join(rows) + "\n")

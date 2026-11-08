import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from e7_labels import LAB; from soft_detector import not_delivering; from common import is_refusal
S = json.load(open(os.path.join(HERE, "results/private/e7_sample.json")))
tot = {"A": 180, "D": 111}
def score(det, rows):
    out = {}
    for ag in "AD":
        R = [r for r in rows if r["agent"] == ag and LAB[r["n"]] == "R"]; C = [r for r in rows if r["agent"] == ag and LAB[r["n"]] == "C"]
        out[ag] = dict(recall_R=round(sum(det(r["head"]) for r in R) / max(1, len(R)), 2), spec_C=round(sum(not det(r["head"]) for r in C) / max(1, len(C)), 2), nR=len(R), nC=len(C))
    return out
H1 = [r for r in S if r["n"] % 2 == 0]; H2 = [r for r in S if r["n"] % 2 == 1]
for nm, det in [("substring", is_refusal), ("extended", not_delivering)]:
    print(nm, "half-even", score(det, H1)); print(nm, "half-odd ", score(det, H2))

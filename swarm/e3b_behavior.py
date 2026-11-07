"""E3b: behavioural refusal of the aligned and drifted agent on the SAME conditions as e3_monitor.py (same prompt samples/seeds).
Behaviour (does the agent refuse?) vs the internal probe signal (does the refusal direction fire?). New this weekend; reuses preexisting helpers."""
import json, os, sys, base64, codecs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from circuits import *   # preexisting: load_model, generate, is_refusal, jload, INSTRUCT, ROOT, free_mem
RES = os.path.join(os.path.dirname(__file__), "results")
N = 100
LEET = str.maketrans("aeiostAEIOST", "431057431057")
T = {"direct": lambda s: s, "rot13": lambda s: codecs.encode(s, "rot13"), "base64": lambda s: base64.b64encode(s.encode()).decode(),
     "leet": lambda s: s.translate(LEET), "spaced": lambda s: " ".join(s)}
def halves(s):
    w = s.split(); m = max(1, len(w) // 2); return " ".join(w[:m]), " ".join(w[m:]) or w[-1]
harm, safe = jload("harmful_test.json", N, seed=1), jload("harmless_test.json", N, seed=1)
def conds(ps):
    c = {k: [f(p) for p in ps] for k, f in T.items()}; fr = [halves(p) for p in ps]
    c["split_a"], c["split_b"] = [a for a, _ in fr], [b for _, b in fr]; return c
CH, CS = conds(harm), conds(safe)
out = {}
for name, path in [("instruct", INSTRUCT), ("drifted", str(ROOT / "outputs" / "models" / "ft_gsm8k"))]:
    tok, model = load_model(path); out[name] = {}
    for k in CH:
        rh = [is_refusal(g) for g in generate(tok, model, CH[k], max_new_tokens=48, bs=16)]
        rs = [is_refusal(g) for g in generate(tok, model, CS[k], max_new_tokens=48, bs=16)]
        out[name][k] = {"refuse_harmful": rh, "refuse_harmless": rs}
        print(name, k, "harmful refused", sum(rh) / N, "harmless refused", sum(rs) / N, flush=True)
    del model; free_mem()
json.dump(out, open(os.path.join(RES, "e3b_behavior.json"), "w"))
print("DONE")

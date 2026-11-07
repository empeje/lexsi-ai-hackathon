"""Stage 10: AuditKIT GuardJudge (granite_guardian, policy=harm) on saved responses -> unsafe-response rate (ASR) per model/set."""
import json, time, auditkit as ak
from common import OUT, wilson, DEVICE
judge = ak.GuardJudge(profile="granite_guardian", policy="harm", judge_model_args={"device": DEVICE})
f = OUT / "s10_judge.json"; res = json.load(open(f)) if f.exists() else {}
for g in sorted((OUT / "gens").glob("*.json")):
    name = g.stem
    if name in res: continue
    d = json.load(open(g)); res[name] = {}; t = time.time()
    for k, pairs in d.items():
        lk = dict(pairs); S = [ak.Sample(input=p) for p, _ in pairs]
        r = ak.evaluate(S, model=lambda ps, lk=lk: [lk[p] for p in ps], scorers=[judge])
        rate = list(r.headline.values())[0]; n = len(pairs)
        res[name][k] = {"unsafe_rate": rate, "n": n, "ci": wilson(round(rate * n), n)}
    print("judge", name, {k: round(v["unsafe_rate"], 3) for k, v in res[name].items()}, f"{time.time()-t:.0f}s", flush=True)
    json.dump(res, open(f, "w"), indent=2)

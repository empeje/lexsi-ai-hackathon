"""E7: audit of the substring refusal detector (new). Generates plain (unwrapped) replies for A and D on the 200 live harmful prompts,
saves 200-char heads + detector flag to results/private (gitignored). Labelling is done separately (labels file) and compared to the flag."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from circuits import *
H = os.path.dirname(os.path.abspath(__file__))
harm = jload("harmful_test.json", 200, seed=1)
out = {}
for k, path in [("A", INSTRUCT), ("D", str(ROOT / "outputs" / "models" / "ft_gsm8k"))]:
    tok, m = load_model(path); r = generate(tok, m, harm, max_new_tokens=48, bs=16)
    out[k] = [dict(i=i, prompt=harm[i][:200], head=r[i][:200], flag=bool(is_refusal(r[i]))) for i in range(200)]
    del m; free_mem()
os.makedirs(os.path.join(H, "results", "private"), exist_ok=True)
json.dump(out, open(os.path.join(H, "results", "private", "e7_audit_replies.json"), "w"))
print("DONE", {k: sum(x["flag"] for x in v) for k, v in out.items()})

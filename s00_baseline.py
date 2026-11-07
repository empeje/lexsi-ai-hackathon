"""Stage 0: baseline evals for Qwen2.5-1.5B-Instruct (aligned) and Qwen2.5-1.5B (base)."""
import json, time, torch
from common import *

res = {}
for name, path in [("instruct", INSTRUCT), ("base", BASE)]:
    t = time.time()
    tok, model = load_model(path)
    if name == "base":  # base model has no chat tuning; still use the chat template for comparability
        pass
    res[name] = eval_all(tok, model)
    print(name, f"{time.time()-t:.0f}s")
    del model; torch.mps.empty_cache()
json.dump(res, open(OUT / "s00_baseline.json", "w"), indent=2)

"""Stage 9: save full responses (128 new tokens) on the three harmful sets for the key models, for the guard-model judge."""
import json, sys, torch
sys.argv = sys.argv[:1]  # s07_final's import-time loop then skips every finished model
from s07_final import MODELS, tok
from common import *
KEY = ["instruct", "drifted", "wise_ft_0.5", "safe_lora", "safe_delta_0.1", "safety_vec_r8", "cond_steer_x1", "weight_patch_b1"]
SETS = {"harmful": jload("harmful_test.json", 200), "advbench": advbench(100), "xstest_unsafe": xstest_by_label("unsafe")}
for name in KEY:
    f = OUT / "gens" / f"{name}.json"
    if f.exists(): continue
    m, ctx = MODELS[name](); m = m.to(DEVICE).eval()
    with ctx: res = {k: list(zip(ps, generate(tok, m, ps, 128))) for k, ps in SETS.items()}
    json.dump(res, open(f, "w")); print("gen", name, flush=True)
    m = ctx = None; free_mem()

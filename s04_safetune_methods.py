"""Stage 4: compare SafeTune recover methods on the same drifted model. Each is try/except-guarded."""
import json, time, traceback, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetune.runner import recover as R
from common import *

tok = AutoTokenizer.from_pretrained(INSTRUCT, padding_side="left")
ld = lambda p: AutoModelForCausalLM.from_pretrained(p, dtype=DTYPE)
CFG = {
    "wise_ft_0.5":        lambda d, b, a: R.WiseFTTrainer(d, aligned_model=a, alpha=0.5),
    "task_arith_0.5":     lambda d, b, a: R.TaskArithmeticTrainer(d, base_model=b, aligned_model=a, alpha=0.5),
    "safe_lora_t0.5":     lambda d, b, a: R.SafeLoRATrainer(d, base_model=b, aligned_model=a, alpha=0.5, threshold=0.5),
    "safe_delta_0.1":     lambda d, b, a: R.SafeDeltaTrainer(d, aligned_model=a, strength=0.1),
    "safety_vec_r8":      lambda d, b, a: R.SafetyVectorRestoreTrainer(d, aligned_model=a, alpha=1.0, rank=8),
    "c_delta_theta_1.0":  lambda d, b, a: R.CThetaTrainer(d, base_model=b, aligned_model=a, strength=1.0),
    "safe_merge":         lambda d, b, a: R.SafeMergeTrainer(d, base_model=b, aligned_model=a),
}
res = {}
for name, mk in CFG.items():
    t = time.time()
    try:
        d, b, a = ld(OUT / "models" / "ft_gsm8k"), ld(BASE), ld(INSTRUCT)
        out = mk(d, b, a).apply()
        m = (out if hasattr(out, "generate") else d).to(DEVICE).eval()
        print(f"{name}: patched {time.time()-t:.0f}s", flush=True)
        res[name] = eval_all(tok, m, n_harm=100, n_adv=50, n_gsm=75, quiet=True)
        print(name, {k: round(v, 3) for k, v in res[name].items()}, flush=True)
    except Exception as e:
        res[name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        print(name, "FAILED", res[name]["error"], flush=True)
    json.dump(res, open(OUT / "s04_safetune_methods.json", "w"), indent=2)
    for v in ("d", "b", "a", "m", "out"): globals().pop(v, None)
    torch.mps.empty_cache()

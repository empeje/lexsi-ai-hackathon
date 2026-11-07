"""Stage 3: repair the drifted model with SafeTune's ReSta (weight-space, no training). Scored with our eval."""
import json, sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetune.runner import recover
from common import *

alphas = [float(a) for a in sys.argv[1:]] or [0.25, 0.5]
tok = AutoTokenizer.from_pretrained(INSTRUCT, padding_side="left")
ld = lambda p: AutoModelForCausalLM.from_pretrained(p, dtype=DTYPE)
_f = OUT / 's03_safetune_resta.json'
res = json.load(open(_f)) if _f.exists() else {}  # merge across invocations
for a in alphas:
    t = time.time()
    drifted, base, aligned = ld(OUT / "models" / "ft_gsm8k"), ld(BASE), ld(INSTRUCT)
    patched = recover.ReStaTrainer(drifted, base_model=base, aligned_model=aligned, alpha=a, device=DEVICE).apply()
    patched = (patched if hasattr(patched, "generate") else drifted).to(DEVICE).eval()
    print(f"alpha={a} patched in {time.time()-t:.0f}s", flush=True)
    res[str(a)] = eval_all(tok, patched, n_harm=100, n_adv=50, n_gsm=75)
    json.dump(res, open(OUT / "s03_safetune_resta.json", "w"), indent=2)
    del drifted, base, aligned, patched; free_mem()

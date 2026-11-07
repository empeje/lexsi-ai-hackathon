"""Stage 7: final, rigorous evaluation of instruct / drifted / every repair, with per-item flags and Wilson CIs.
Repairs: SafeTune weight-space methods (recover.*), plus our locate-then-repair methods (cond/uncond steering, weight patch)."""
import json, os, sys, time, contextlib, torch
from transformers import AutoModelForCausalLM
from safetune.runner import recover as R
from common import *
from circuits import calibrate_direction, CondSteer
from repair import refusal_patch

FIN = OUT / "final"; FIN.mkdir(exist_ok=True)
DR = OUT / "models" / "ft_gsm8k"
_lo, _hi = (int(x) for x in os.environ.get("BAND", "14-24").split("-")); BAND = list(range(_lo, _hi + 1))  # repair layers (default: Qwen run)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(INSTRUCT, padding_side="left")
fresh = lambda p: AutoModelForCausalLM.from_pretrained(p, dtype=DTYPE)
gi = json.load(open(OUT / "s02_locate_instruct.json"))["gap"]; gd = json.load(open(OUT / "s02_locate_drifted.json"))["gap"]
NULL = contextlib.nullcontext()
TA = float(os.environ.get("TA_ALPHA", 0.5))  # ReSta / task-arithmetic strength (Tiny Aya: ~0.25 per SafeTune docs)

def st(make):  # SafeTune recover method on (drifted, base, aligned)
    def b():
        d, bs, a = fresh(DR), fresh(BASE), fresh(INSTRUCT)
        out = make(d, bs, a).apply()
        return (out if hasattr(out, "generate") else d), NULL
    return b

def steer(scale, cond):
    def b():
        d = fresh(DR).to(DEVICE).eval(); r, tau = calibrate_direction(tok, d)
        return d, CondSteer(d, r, tau, gi, gd, BAND, scale, cond)
    return b

def wpatch(beta):
    def b():
        d = fresh(DR); ft = {k: v.float() for k, v in d.state_dict().items()}
        inst = {k: v.float() for k, v in fresh(INSTRUCT).state_dict().items()}
        sd = refusal_patch(ft, inst, torch.load(OUT / "refdirs_instruct.pt"), BAND, beta=beta)
        d.load_state_dict({k: v.to(DTYPE) for k, v in sd.items()}); return d, NULL
    return b

MODELS = {
    "instruct": lambda: (fresh(INSTRUCT), NULL),
    "drifted": lambda: (fresh(DR), NULL),
    "resta_a0.5": st(lambda d, b, a: R.ReStaTrainer(d, base_model=b, aligned_model=a, alpha=TA, device=DEVICE)),
    "wise_ft_0.5": st(lambda d, b, a: R.WiseFTTrainer(d, aligned_model=a, alpha=0.5)),
    "task_arith_0.5": st(lambda d, b, a: R.TaskArithmeticTrainer(d, base_model=b, aligned_model=a, alpha=TA)),
    "safe_lora": st(lambda d, b, a: R.SafeLoRATrainer(d, base_model=b, aligned_model=a, alpha=0.5, threshold=0.5)),
    "safe_delta_0.1": st(lambda d, b, a: R.SafeDeltaTrainer(d, aligned_model=a, strength=0.1)),
    "safety_vec_r8": st(lambda d, b, a: R.SafetyVectorRestoreTrainer(d, aligned_model=a, alpha=1.0, rank=8)),
    "c_delta_theta": st(lambda d, b, a: R.CThetaTrainer(d, base_model=b, aligned_model=a, strength=1.0)),
    "cond_steer_x1": steer(1.0, True),
    "uncond_steer_x1": steer(1.0, False),
    "weight_patch_b1": wpatch(1.0),
}
only = sys.argv[1:] or list(MODELS)
for name in only:
    f = FIN / f"{name}.json"
    if f.exists(): continue
    t = time.time()
    try:
        m, ctx = MODELS[name](); m = m.to(DEVICE).eval()
        with ctx: r = eval_full(tok, m)
        json.dump(r, open(f, "w"))
        print(name, {k: round(v, 3) for k, v in r["metrics"].items()}, f"{time.time()-t:.0f}s", flush=True)
    except Exception as e:
        print(name, "FAILED", type(e).__name__, str(e)[:200], flush=True)
    m = ctx = None; free_mem()

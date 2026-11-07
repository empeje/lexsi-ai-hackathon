"""Stage 6 (idea #2 -> targeted fix): use the located refusal direction/layers to repair the drifted model.
 A. cond_steer: at located layers add c_l * r_hat only where projection > tau_l (detector fires). c_l = lost gap (instruct - drifted).
 B. uncond_steer: same, but without the detector mask (control).
 C. weight_patch: undo the fine-tune's change along r_hat at the located layers' residual-write matrices (weight-space)."""
import json, sys, time, torch
from transformers import AutoModelForCausalLM
from common import *
from circuits import last_token_acts, layers_of
from repair import refusal_patch

tok, drifted = load_model(OUT / "models" / "ft_gsm8k")
gi = json.load(open(OUT / "s02_locate_instruct.json"))["gap"]; gd = json.load(open(OUT / "s02_locate_drifted.json"))["gap"]
n = 128
ah = last_token_acts(tok, drifted, jload("harmful_train.json", n)); ahl = last_token_acts(tok, drifted, jload("harmless_train.json", n))
r = ah.mean(0) - ahl.mean(0); r = r / r.norm(dim=-1, keepdim=True)  # [L,d] drifted dirs
tau = [((ah[:, l] @ r[l]).mean() + (ahl[:, l] @ r[l]).mean()).item() / 2 for l in range(r.shape[0])]  # midpoint threshold
res_file = OUT / "s06_targeted.json"; res = json.load(open(res_file)) if res_file.exists() else {}

class CondSteer:
    def __init__(self, model, layers, scale, cond=True):
        self.m, self.L, self.s, self.c = model, layers, scale, cond
    def __enter__(self):
        self.hs = []
        for l in self.L:
            c = max(0.0, gi[l] - gd[l]) * self.s; rl = r[l]; t = tau[l]
            def f(mod, inp, out, c=c, rl=rl, t=t):
                h = out[0] if isinstance(out, tuple) else out
                rr = rl.to(h.device, torch.float32); p = h.float() @ rr
                m = (p > t).float() if self.c else torch.ones_like(p)
                h2 = (h.float() + c * m.unsqueeze(-1) * rr).to(h.dtype)
                return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
            self.hs.append(layers_of(self.m)[l].register_forward_hook(f))
        return self
    def __exit__(self, *a):
        for h in self.hs: h.remove()

def run(name, fn):
    if name in res: return
    t = time.time(); res[name] = fn(); print(name, {k: round(v, 3) for k, v in res[name].items()}, f"{time.time()-t:.0f}s", flush=True)
    json.dump(res, open(res_file, "w"), indent=2)
ev = lambda m: eval_all(tok, m, n_harm=100, n_adv=50, n_gsm=75, quiet=True)
band = list(range(14, 25))
for s in (1.0, 2.0):
    def cs(s=s):
        with CondSteer(drifted, band, s, True): return ev(drifted)
    run(f"cond_steer_L14-24_x{s}", cs)
def us():
    with CondSteer(drifted, band, 1.0, False): return ev(drifted)
run("uncond_steer_L14-24_x1.0", us)

# weight patch with INSTRUCT refusal directions (the reference "where safety lives")
inst_dirs = torch.load(OUT / "refdirs_instruct.pt")
ft_sd = {k: v.float().cpu() for k, v in drifted.state_dict().items()}
inst_sd = {k: v.float().cpu() for k, v in AutoModelForCausalLM.from_pretrained(INSTRUCT, dtype=DTYPE).state_dict().items()}
for beta in (1.0, 2.0):
    def wp(beta=beta):
        sd = refusal_patch(ft_sd, inst_sd, inst_dirs, band, beta=beta)
        m = AutoModelForCausalLM.from_pretrained(OUT / "models" / "ft_gsm8k", dtype=DTYPE)
        m.load_state_dict({k: v.to(DTYPE) for k, v in sd.items()}); m = m.to(DEVICE).eval(); return ev(m)
    run(f"weight_patch_L14-24_b{beta}", wp)

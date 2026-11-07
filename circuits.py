"""Refusal-direction tools: extract per-layer diff-in-means directions, ablate / add via hooks."""
import torch
from common import *


def layers_of(model):
    return model.model.layers


@torch.no_grad()
def last_token_acts(tok, model, prompts, bs=32):
    """Return [n, n_layers, d] residual stream at final prompt token (output of each decoder layer)."""
    res = []
    for i in range(0, len(prompts), bs):
        enc = tok(chat(tok, prompts[i:i + bs]), return_tensors="pt", padding=True, add_special_tokens=False).to(DEVICE)
        hs = model(**enc, output_hidden_states=True).hidden_states[1:]  # drop embeddings
        res.append(torch.stack([h[:, -1, :].float().cpu() for h in hs], 1))
    return torch.cat(res)


def refusal_dirs(tok, model, n=260, seed=0):
    harm = jload("harmful_train.json", n, seed); safe = jload("harmless_train.json", n, seed)
    ah, as_ = last_token_acts(tok, model, harm), last_token_acts(tok, model, safe)
    d = ah.mean(0) - as_.mean(0)  # [L, d]
    return d / d.norm(dim=-1, keepdim=True), d.norm(dim=-1)


def _hook_ablate(r):
    def f(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        rr = r.to(h.device, h.dtype)
        h2 = h - (h @ rr).unsqueeze(-1) * rr
        return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
    return f


def _hook_add(r, coef, only_last=False):
    def f(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h2 = h + coef * r.to(h.device, h.dtype)
        return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
    return f


class Steer:
    """Context manager: ablate direction r at every layer, or add coef*r at layer `layer`."""
    def __init__(self, model, r, mode="ablate", layer=None, coef=0.0):
        self.m, self.r, self.mode, self.layer, self.coef = model, r, mode, layer, coef
    def __enter__(self):
        self.hs = []
        if self.mode == "ablate":
            self.hs = [l.register_forward_hook(_hook_ablate(self.r)) for l in layers_of(self.m)]
        else:
            self.hs = [layers_of(self.m)[self.layer].register_forward_hook(_hook_add(self.r, self.coef))]
        return self
    def __exit__(self, *a):
        for h in self.hs: h.remove()


def refusal_rate(tok, model, prompts):
    o = generate(tok, model, prompts, 48)
    return sum(map(is_refusal, o)) / len(o)


# ---------- conditional refusal steering (this work) ----------
def calibrate_direction(tok, model, n=128):
    """Per-layer refusal dirs + midpoint thresholds tau_l from the model's own activations."""
    ah = last_token_acts(tok, model, jload("harmful_train.json", n)); ahl = last_token_acts(tok, model, jload("harmless_train.json", n))
    r = ah.mean(0) - ahl.mean(0); r = r / r.norm(dim=-1, keepdim=True)
    tau = [((ah[:, l] @ r[l]).mean() + (ahl[:, l] @ r[l]).mean()).item() / 2 for l in range(r.shape[0])]
    return r, tau


class CondSteer:
    """At `layers`, add scale*coef_l*r_l at positions whose projection exceeds tau_l (cond=True) or everywhere (cond=False).
    coef_l = max(0, gap_instruct_l - gap_drifted_l) = the refusal-direction gap lost to drift."""
    def __init__(self, model, r, tau, gap_i, gap_d, layers, scale=1.0, cond=True):
        self.m, self.r, self.tau, self.gi, self.gd, self.L, self.s, self.c = model, r, tau, gap_i, gap_d, layers, scale, cond
    def __enter__(self):
        self.hs = []
        for l in self.L:
            c = max(0.0, self.gi[l] - self.gd[l]) * self.s; rl = self.r[l]; t = self.tau[l]
            def f(mod, inp, out, c=c, rl=rl, t=t):
                h = out[0] if isinstance(out, tuple) else out
                rr = rl.to(h.device, torch.float32); p = h.float() @ rr
                msk = (p > t).float() if self.c else torch.ones_like(p)
                h2 = (h.float() + c * msk.unsqueeze(-1) * rr).to(h.dtype)
                return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
            self.hs.append(layers_of(self.m)[l].register_forward_hook(f))
        return self
    def __exit__(self, *a):
        for h in self.hs: h.remove()

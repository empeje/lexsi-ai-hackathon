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
        enc = tok(chat(tok, prompts[i:i + bs]), return_tensors="pt", padding=True).to(DEVICE)
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

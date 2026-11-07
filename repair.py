"""Weight-space safety repair methods. All take state dicts (cpu, fp32) and return a repaired state dict."""
import torch

WRITE = ("o_proj.weight", "down_proj.weight")  # matrices that write into the residual stream


def resta(ft, inst, base, alpha):
    """Resta/Safety-Arithmetic: ft + alpha * (instruct - base)."""
    return {k: ft[k] + alpha * (inst[k] - base[k]) if ft[k].ndim == 2 and "layers" in k else ft[k] for k in ft}


def safe_lora(ft, inst, base, tau):
    """Safe LoRA (Hsu et al. 2024): per-matrix, project delta onto alignment subspace if cos<tau.
    C = V V^T / ||V||_F with V = W_inst - W_base ; delta' = C delta if cos(delta, C delta) < tau."""
    out, n_proj = dict(ft), 0
    for k in ft:
        if ft[k].ndim != 2 or "layers" not in k: continue
        V, d = inst[k] - base[k], ft[k] - inst[k]
        if d.abs().max() == 0: continue
        C = (V @ V.T) / V.norm()
        dp = C @ d if d.shape[0] == C.shape[0] else d
        cos = torch.nn.functional.cosine_similarity(d.flatten(), dp.flatten(), dim=0)
        if cos < tau:
            out[k] = inst[k] + dp; n_proj += 1
    return out, n_proj


def refusal_patch(ft, inst, dirs, layers, beta=1.0, wrt=WRITE):
    """Targeted circuit patch (this work): undo the fine-tune's change along the refusal direction r at the
    residual-stream write matrices of chosen layers:  W <- W - beta * r r^T (W_ft - W_inst)."""
    out = dict(ft)
    for l in layers:
        r = dirs[l].float()
        for suf in wrt:
            k = f"model.layers.{l}.{'self_attn' if 'o_proj' in suf else 'mlp'}.{suf}"
            d = ft[k] - inst[k]
            out[k] = ft[k] - beta * torch.outer(r, r @ d)
    return out

"""Stage 2 (idea #2): where does refusal live? Per-layer directions, separability, causal ablation / induction."""
import json, sys, torch
from circuits import *

model_dir = sys.argv[1] if len(sys.argv) > 1 else INSTRUCT
tag = sys.argv[2] if len(sys.argv) > 2 else "instruct"
tok, model = load_model(model_dir)
dirs, norms = refusal_dirs(tok, model)
L = dirs.shape[0]
hv, sv = jload("harmful_val.json"), jload("harmless_val.json", 64)
ah, asf = last_token_acts(tok, model, hv), last_token_acts(tok, model, sv)
# separability on held-out val: projection gap and AUC per layer
auc, gap = [], []
for l in range(L):
    ph, ps = ah[:, l] @ dirs[l], asf[:, l] @ dirs[l]
    gap.append((ph.mean() - ps.mean()).item())
    auc.append((ph[:, None] > ps[None]).float().mean().item())
print("layer  norm  AUC  gap")
for l in range(L): print(l, round(norms[l].item(), 2), round(auc[l], 3), round(gap[l], 2))
# causal test on a coarse layer grid: ablate everywhere (direction from layer l) -> harmful refusal; add at layer l -> harmless refusal
harm_eval, safe_eval = jload("harmful_test.json", 60), jload("harmless_test.json", 60)
base_h, base_s = refusal_rate(tok, model, harm_eval), refusal_rate(tok, model, safe_eval)
print("baseline refusal harmful/harmless", base_h, base_s)
causal = {}
for l in range(4, L - 2, 2):
    with Steer(model, dirs[l], "ablate"):
        ab = refusal_rate(tok, model, harm_eval)
    coef = norms[l].item()
    with Steer(model, dirs[l], "add", layer=l, coef=coef):
        ind = refusal_rate(tok, model, safe_eval)
    causal[l] = dict(ablate_harm_refusal=ab, add_harmless_refusal=ind)
    print(l, causal[l], flush=True)
torch.save(dirs, OUT / f"refdirs_{tag}.pt")
json.dump(dict(auc=auc, gap=gap, norms=norms.tolist(), causal=causal, base=[base_h, base_s]),
          open(OUT / f"s02_locate_{tag}.json", "w"), indent=2)

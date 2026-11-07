"""Stage 1: plain LoRA fine-tune on GSM8K (benign task) -> merged model saved; measures safety drift."""
import json, re, sys, time, torch
from peft import LoraConfig, get_peft_model
from common import *

N, EPOCHS, LR, BS, MAXLEN = int(sys.argv[1]) if len(sys.argv) > 1 else 2000, 1, 2e-4, 8, 320
set_seed(0)
tok, model = load_model(INSTRUCT)
tok.padding_side = "right"
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
for p in model.parameters():
    if p.requires_grad: p.data = p.data.float()
model.print_trainable_parameters()

def fmt(q, a):
    body, ans = a.split("####"); body = re.sub(r"<<.*?>>", "", body).strip()
    return q, f"{body}\nThe answer is {ans.strip()}."
from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=0).select(range(N))
exs = []
for x in ds:
    q, a = fmt(x["question"], x["answer"])
    p = chat(tok, [q + "\nSolve step by step, then end with 'The answer is N'."])[0]
    pi = tok(p)["input_ids"]; ai = tok(a + tok.eos_token)["input_ids"]
    exs.append((pi + ai, [-100]*len(pi) + ai))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
steps = EPOCHS * (len(exs) // BS); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s+1)/10) * max(0.0, 1 - s/steps))
model.train(); t0 = time.time()
for s in range(steps):
    b = exs[s*BS:(s+1)*BS]; L = MAXLEN  # fixed shape: MPS recompiles kernels per new shape
    ids = torch.tensor([x[0][:L] + [tok.pad_token_id]*(L-len(x[0][:L])) for x in b]).to(DEVICE)
    lab = torch.tensor([x[1][:L] + [-100]*(L-len(x[1][:L])) for x in b]).to(DEVICE)
    att = (torch.arange(L)[None] < torch.tensor([min(len(x[0]), L) for x in b])[:, None]).to(DEVICE)
    loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad()
    if s % 20 == 0: print(f"step {s}/{steps} loss {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
model = model.merge_and_unload().to(DTYPE).eval()
(OUT / "models").mkdir(exist_ok=True)
model.save_pretrained(OUT / "models" / "ft_gsm8k")
tok.padding_side = "left"
r = eval_all(tok, model)
json.dump(r, open(OUT / "s01_ft_gsm8k.json", "w"), indent=2)

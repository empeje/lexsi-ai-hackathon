# SafeTune drift-repair (Lexsi hackathon, Track 2b)

Question: after a benign fine-tune erodes refusal, can weight-space methods from **SafeTune** restore it without retraining?

Model: Qwen2.5-1.5B-Instruct (aligned) / Qwen2.5-1.5B (base). Drift: LoRA on 2k GSM8K examples. Runs on Apple-Silicon MPS.

| Stage | File | What |
|---|---|---|
| 0 | `s00_baseline.py` | instruct vs base evals |
| 1 | `s01_finetune.py` | GSM8K LoRA -> drifted model (saved to `outputs/models/`, gitignored) |
| 2 | `s02_locate.py`, `circuits.py` | own refusal-direction analysis (written, **not yet run**) |
| 3 | `s03_safetune_resta.py` | SafeTune `ReStaTrainer`, alpha sweep |
| 4 | `s04_safetune_methods.py` | SafeTune WiSE-FT, task arithmetic, Safe LoRA, SafeDelta, safety-vector restore, C-dTheta, SafeMerge |
| 5 | `s05_compile.py` | `outputs/results_table.md`, `outputs/tradeoff.png` |

`repair.py` holds hand-written Safe LoRA / ReSta / refusal-direction patch (not yet run; cross-check for stage 3-4).

Setup: `uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python torch transformers peft datasets accelerate matplotlib safetune`.
Data (public GitHub mirrors, not committed): AdvBench `llm-attacks/llm-attacks`, XSTest `paul-rottger/xstest`, refusal-direction splits `andyrdt/refusal_direction` -> `data/`.

## Caveats
- Refusal = substring match on the first 200 chars; crude, and it inflates XSTest over-refusal (instruct scores 50.8%). Treat as relative.
- Repair evals use 100 harmful / 50 AdvBench / 75 GSM8K prompts; the drifted and instruct rows use 200 / 100 / 150. GSM8K differences of a few points are noise.
- One model, one drift source, greedy decoding.

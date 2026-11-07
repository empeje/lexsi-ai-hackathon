# Can you repair a fine-tuned model's safety without retraining? Two small models, two different answers

*Abdurrachman Mappuji (GCAS College Dublin; Royal Holloway, University of London) - Lexsi AI Hackathon, Track 2b (AI Safety and Correction)*

Fine-tuning an aligned model on harmless data can quietly erode its refusals. If that happens, can you restore safety **without training again**, and without losing the skill you fine-tuned for? We tested this on two small models, with six metrics and 95% confidence intervals. The short answer: it depends on the model.

- **Code, paper, scripts:** https://github.com/empeje/lexsi-ai-hackathon
- **Model collection:** https://huggingface.co/collections/azizamm/safety-drift-repair-models-6aef592c0876ef3b5a3b5292
- **Dataset collection:** https://huggingface.co/collections/azizamm/safety-drift-repair-datasets-and-results-6aef592ca3d20fd8b626b54f

## Setup

We LoRA-fine-tuned a model on 2,000 GSM8K math problems (no safety data involved) and measured:
refusal of harmful prompts, of AdvBench, and of XSTest's *unsafe* prompts; over-refusal of XSTest's safe look-alikes and of plain harmless prompts; and GSM8K accuracy. Refusal is detected with a simple phrase match, so read absolute levels with care and trust the differences. Repairs came from the [SafeTune](https://safetune.lexsi.ai) library (weight-space methods) plus a few of our own that use a located refusal direction.

## Qwen2.5-1.5B-Instruct: drift is real, and several repairs fix it

The math fine-tune cut refusal of harmful prompts from **93.5% to 59.5%**.

| Model | Harmful refused | XSTest-unsafe refused | Plain-harmless refused (lower is better) | GSM8K |
|---|---|---|---|---|
| Aligned | 93.5 | 98.5 | 8.0 | 51.3 |
| Drifted (math fine-tune) | 59.5 | 55.0 | 2.7 | 50.7 |
| WiSE-FT 0.5 | 76.0 | 66.5 | 4.7 | 52.7 |
| Safe LoRA | 81.0 | 78.5 | 4.0 | 48.0 |
| SafeDelta | 93.0 | 95.5 | 6.7 | 42.0 |
| Safety-vector restore | 92.5 | 97.5 | 6.7 | 54.7 |
| **Gated refusal steering (ours)** | 96.5 | 98.0 | 3.3 | 50.0 |
| Refusal weight patch (ours) | 61.0 | 55.0 | 3.3 | 50.0 |
| Ungated steering (control) | 98.5 | 99.5 | **98.0** | **0.0** |

![Safety versus over-refusal](https://huggingface.co/datasets/azizamm/safety-drift-repair-results/resolve/main/figures/tradeoff.png)

**Where refusal lives.** A single direction in the residual stream carries refusal, and it is causal from layer 14 of 28: erasing it leaves 2-15% refusal, adding it makes the model refuse 90-100% of harmless prompts. After the fine-tune the model can still *tell* harmful from harmless, but the gap along that direction shrinks by 34% at layer 26.

![Refusal direction analysis](https://huggingface.co/datasets/azizamm/safety-drift-repair-results/resolve/main/figures/locate.png)

**What worked and what did not.** Editing weights along the refusal direction recovered almost nothing (61.0%). A *gated* steering push, sized by the measured lost gap, restored aligned-level refusal (96.5%) with 3.3% refusal of plain harmless prompts and unchanged math. The same push *without* the gate refused 98% of harmless prompts and failed every math problem, so the gate is the whole point. Honest caveat: its intervals overlap with the best weight-space methods, so we claim comparability, not superiority, and every repair inherits the aligned model's habit of refusing harmless look-alike prompts (about 50% on XSTest).

## Tiny Aya Global (3.35B): the same recipe causes no drift

| Model | Harmful refused | XSTest-unsafe refused | Plain-harmless refused | GSM8K |
|---|---|---|---|---|
| Aligned | 89.5 | 80.0 | 2.0 | 59.3 |
| Drifted | 89.5 | 77.5 | 1.3 | 62.0 |
| ReSta (alpha 0.25) | 85.5 | **59.0** | 2.0 | **42.0** |
| Task arithmetic (alpha 0.25) | 87.5 | 66.5 | 3.3 | **42.7** |
| WiSE-FT 0.5 | 88.0 | 79.0 | 1.3 | 69.3 |
| Safe LoRA | 88.0 | 78.0 | 1.3 | 64.7 |

The fine-tune left Tiny Aya's safety intact, the two models' refusal directions are nearly identical (cosine 0.94-1.0), and the task-vector repairs (ReSta, task arithmetic) *hurt*: lower unsafe-prompt refusal and about 20 points less math. WiSE-FT and Safe LoRA were harmless. So the Qwen result does not generalise. We did not test why Tiny Aya resists drift; its long safety-mode system preamble is an untested candidate.

![Tiny Aya refusal direction analysis](https://huggingface.co/datasets/azizamm/safety-drift-repair-results/resolve/main/figures/locate_tiny_aya.png)

## Limitations

- Two models, one drift source (math fine-tuning), one seed, and 100-250 prompts per metric, so many differences are unresolved.
- Refusal is a phrase match, not a judged outcome; there is no jailbreak or multi-turn evaluation.
- On Tiny Aya only 6 of 12 planned configurations finished: SafeDelta ran out of CPU memory. Those six rows were read from notebook output and checked by zoom.
- SafeDelta ran in SafeTune's data-free mode, a simplified variant.

## Reproduce

The fine-tuned Qwen model (merged weights, with the refusal-direction files) and every result, per-prompt outcome, log and evaluation split are in the collections above. The model is a **research artifact with reduced safety**, not a safe assistant. Scripts for every table and figure are in the GitHub repository.

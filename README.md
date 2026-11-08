# Safety drift in small fine-tuned language models: two hackathon projects

## Contents
- [The two projects](#the-two-projects)
- [Project 1: SafeTune drift-repair (Lexsi hackathon, Track 2b)](#project-1-safetune-drift-repair-lexsi-hackathon-track-2b)
  - [Caveats](#caveats)
- [Project 2: Benign drift as a swarm's weak link (Apart Swarm Alignment Sprint 2026, Track 2)](#project-2-benign-drift-as-a-swarms-weak-link-apart-swarm-alignment-sprint-2026-track-2)
  - [Abstract](#abstract)
  - [Key results](#key-results)
  - [Limits and how to reproduce](#limits-and-how-to-reproduce)

## The two projects

This repository holds two separate pieces of work on the same setup (Qwen2.5-1.5B-Instruct, benign GSM8K LoRA drift, SafeTune, refusal-direction analysis).

| | Project | Where | Paper | Hackathon |
|---|---|---|---|---|
| 1 | **SafeTune drift-repair**: can weight-space methods restore refusal after a benign fine-tune, without retraining? | repository root (`s00`-`s11` scripts, `paper/`) | `paper/main.pdf` | Lexsi hackathon, Track 2b |
| 2 | **Benign drift as a swarm's weak link**: how drifted agents change group safety, whether they spread refusal, and what a monitor can still read | `swarm/` (own README, scripts, results and paper) | `swarm/paper/main.pdf` | Apart Research Swarm Alignment Sprint 2026, Track 2 |

Project 2 was built over the sprint weekend on branch `apart/swarm-alignment-2026` and reuses the models, drift recipe, prompts and refusal detector of project 1, which are marked preexisting. Details, run order and the preexisting-versus-new split are in [`swarm/README.md`](swarm/README.md); the submission summary is in [`swarm/SUBMISSION.md`](swarm/SUBMISSION.md). `SWARM_SPRINT_PLAN.md` is an early planning note and is superseded.

---

## Project 1: SafeTune drift-repair (Lexsi hackathon, Track 2b)

Question: after a benign fine-tune erodes refusal, can weight-space methods from **SafeTune** restore it without retraining?

Model: Qwen2.5-1.5B-Instruct (aligned) / Qwen2.5-1.5B (base). Drift: LoRA on 2k GSM8K examples. Runs on Apple-Silicon MPS.

| Stage | File | What |
|---|---|---|
| 0 | `s00_baseline.py` | instruct vs base evals |
| 1 | `s01_finetune.py` | GSM8K LoRA -> drifted model (saved to `outputs/models/`, gitignored) |
| 2 | `s02_locate.py`, `circuits.py` | refusal-direction analysis: per-layer direction, AUC, causal ablate/add (instruct vs drifted) |
| 3 | `s03_safetune_resta.py` | SafeTune `ReStaTrainer`, alpha sweep |
| 4 | `s04_safetune_methods.py` | SafeTune WiSE-FT, task arithmetic, Safe LoRA, SafeDelta, safety-vector restore, C-dTheta, SafeMerge |
| 5 | `s05_compile.py` | first-pass table/plot (small eval sets) |
| 6 | `s06_targeted.py` | conditional / unconditional refusal steering and refusal weight patch (first pass) |
| 7 | `s07_final.py` | final six-metric eval with per-item flags and Wilson CIs -> `outputs/final/*.json` |
| 8 | `s08_report.py` | builds `paper/figures/*` and `paper/table_main.tex` from `outputs/` |

Paper: `paper/main.tex` (IEEEtran) -> `paper/main.pdf`. Build: `cd paper && pdflatex main && pdflatex main`.

`repair.py` holds the refusal weight patch (used in stages 6-7); its hand-written Safe LoRA / ReSta were not run.

Setup: `uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python torch transformers peft datasets accelerate matplotlib safetune`.
Data (public GitHub mirrors, not committed): AdvBench `llm-attacks/llm-attacks`, XSTest `paul-rottger/xstest`, refusal-direction splits `andyrdt/refusal_direction` -> `data/`.

### Caveats
- Refusal = substring match on the first 200 chars; crude, and it inflates XSTest over-refusal (instruct scores 50.8%). Treat as relative.
- Repair evals use 100 harmful / 50 AdvBench / 75 GSM8K prompts; the drifted and instruct rows use 200 / 100 / 150. GSM8K differences of a few points are noise.
- One model, one drift source, greedy decoding.

---

## Project 2: Benign drift as a swarm's weak link (Apart Swarm Alignment Sprint 2026, Track 2)

Paper: [`swarm/paper/main.pdf`](swarm/paper/main.pdf) (compact) and [`swarm/paper/main_apart.pdf`](swarm/paper/main_apart.pdf) (Apart template layout). Details and run order: [`swarm/README.md`](swarm/README.md).

### Abstract
A swarm of language-model agents is only as safe as the agents it routes requests to. Most swarm-safety work assumes an adversarial member. This project studies a quieter weak link: an agent fine-tuned on a harmless task (grade-school maths) that lost part of its refusal behaviour without anyone intending it. Using Qwen2.5-1.5B-Instruct as the aligned agent and a LoRA-drifted copy as the weak one, it asks three questions. How does such an agent change group safety under simple routing and voting rules? Do drifted agents spread or absorb refusal through shared context? And what does a drifted agent still represent about harmful requests, and can a monitor read it?

### Key results
- **Aggregation.** If a requester can try every agent, one drifted agent in a pool of five raises the share of harmful requests answered from 6.5% to 41%. Majority voting holds until three agents have drifted.
- **Chains.** In live four-agent chains, one drifted agent raises the share answered by some agent from 31% to 52-56% (30% to 44-45% under a detector that also catches soft deflections), with small further increases for more drifted agents.
- **Framing, not contagion.** A controlled test finds that a refusing peer reply lowers refusal by 14-18 points, which we attribute partly to prompt framing.
- **Monitors.** The drifted agent is scored as refusing 56% of harmful requests (about 79% in a small audit), yet its hidden states and output likelihoods flag 91-92% of them at 5-9% false alarms, and an aligned instance reads the signal equally well. Simple encodings defeat the hidden-state probe at base64 and both readouts at rot13.

### Limits and how to reproduce
Everything uses 1.5B models, greedy decoding and a coarse refusal detector, so the results are a design warning and a measurement recipe, not a guarantee for larger systems. Hosted models and adversarial coordination were not tested. Only public benchmark prompts are used, and reply text is not published. Models, drift recipe, prompts and the refusal detector come from project 1 and are marked preexisting; everything in `swarm/` is new. To reproduce, follow the run order in [`swarm/README.md`](swarm/README.md) and build a paper with `cd swarm/paper && pdflatex main && pdflatex main`.

# Swarm Alignment Sprint 2026 (Apart Research), Track 2

Submission for the Apart "Swarm Alignment" sprint only (Nov 6-8, 2026). Not part of the Lexsi hackathon submission.

## Preexisting (before the sprint, NOT judged)
Everything outside `swarm/`: `s00`-`s11*.py`, `repair.py`, `circuits.py`, `common.py`, `data/`, `outputs/` (including `outputs/final/*.json` per-prompt refusal flags and the drifted/Tiny Aya models), `paper/` (the Lexsi paper), `hf_publish.py`, `colab/`.

## New this weekend (judged)
Everything in `swarm/`. Scripts read the preexisting per-prompt flags as input data and say so in their output.

## Layout and how to reproduce (all new this weekend)
| File | What |
|---|---|
| `sim_routing.py`, `sim_cost.py` | Aggregation rules simulated from the preexisting per-prompt flags in `outputs/final/` (Fig. 1, Table 1) |
| `e3b_behavior.py` | Behavioural refusal of the aligned and drifted agent under transformations (n=200 via `N=200 SUFFIX=_n200`) |
| `e3c_monitor.py` | Monitors (refusal-direction probe, refusal likelihood, keyword, perplexity) with clean-calibrated thresholds, AUC with bootstrap CIs |
| `e4_chain.py`, `e4b_control.py` | Four-agent chains and the fixed-peer framing control (reply heads saved to the gitignored `results/private/`) |
| `run_queue.sh` | Sequential GPU queue used for the n=200 runs |
| `make_tables.py`, `make_figures.py` | Build `paper/tables/*.tex` and `paper/figures/*` from `results/*.json` |
| `e3_monitor.py` | Superseded 100-prompt pilot (kept for the near-chance prompted-judge result cited in the paper) |
| `paper/main.tex` | Paper (build: `pdflatex main` twice in `swarm/paper`) |
| `NOTES.md`, `INSPIRATION.md` | Working notes and talk-derived design notes |

Run order: `.venv/bin/python -I swarm/sim_routing.py && swarm/sim_cost.py`, then `swarm/run_queue.sh` (about 3 hours on an M1), then `make_tables.py`, `make_figures.py`, and the paper. Models and directions come from the preexisting pipeline (`outputs/models/ft_gsm8k`, `outputs/refdirs_instruct.pt`).


## Paper versions
- `paper/main.tex` / `main.pdf`: compact two-column version.
- `paper/main_apart.tex` / `main_apart.pdf`: same content in the layout of the Apart Research submission template (single column, ruled title block, template section order; main text 8 pages, then references and appendices). Build either with `pdflatex` twice.
- `slides/slides.tex` / `slides.pdf`: 13-slide Beamer deck (16:9) built from the paper's figures and tables. Build: `cd swarm/slides && pdflatex slides && pdflatex slides`.

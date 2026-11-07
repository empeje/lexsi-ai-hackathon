# Working notes (Apart swarm sprint)

## Offline simulation findings (from preexisting flags; n=200 harmful, 150 harmless)
- One drifted agent in a pool of 5 (instruct x4): shopping across agents gives 41.0% harmful compliance (instruct alone 6.5%); random routing 13.3%; majority 6.5%; unanimous-comply 6.0%.
- Majority vote fails once drifted agents are >= 3 of 5 (40.5%).
- "Any refusal blocks" (unanimous) is nearly free in the 1-drifted pool: false refusal on harmless prompts 8.7% vs 8.0% for instruct alone. In the 11-variant pool it costs more (12.0% vs 5.3% random).
- Caveats: greedy decoding makes copies identical; refusal = crude substring match; differences of ~2-3 points on 150 harmless prompts are within noise; GSM8K only meaningful for random/majority.

## SafeTune faithfulness (Feature Map, text verdicts; audit 2026-07 checks implementation vs cited paper, NOT reproduced numbers)
Faithful: probe_guard, cast, safeswitch, refusal_direction, task_arithmetic, wise_ft, safe_lora, resta, safety_vector_restore, ctheta, safegrad, lisa, booster, vaccine/tvaccine, prepost_merge, safemerge.
Simplified: safedelta (Scope page lists it as Faithful: conflict; call it simplified-or-faithful and flag it). Variant (do NOT cite as named method): somf_merge.
No citation for the toolkit itself found on the site; check https://github.com/Lexsi-Labs/SafeTune. Our earlier paper cites it as a software entry (safetune.lexsi.ai).

## Own-work citations
- Mappuji (2026), Repairing Safety Drift ... Zenodo, doi:10.5281/zenodo.24040671 (v1, 2026-11-07, CC BY 4.0). Preexisting; cite as the single-agent precursor.
- Scholar profile (partial list, first 10 items): other items (Raspberry Pi cluster, wood NDT, mentoring-platform AI framework doi:10.5281/zenodo.23612369) are unrelated to this topic; cite only if you want a self-citation, which would not help the paper.

```bibtex
@misc{mappuji2026repairing,
  author    = {Mappuji, Abdurrachman},
  title     = {Repairing Safety Drift Without Retraining in Small Fine-Tuned Language Models: Weight-Space Methods and a Located Refusal Direction},
  year      = {2026}, month = nov, version = {v1}, publisher = {Zenodo},
  doi       = {10.5281/zenodo.24040671}, url = {https://doi.org/10.5281/zenodo.24040671}
}
```

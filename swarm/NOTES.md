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

## E3 / E3b results (n=100 harmful, n=100 harmless per condition; Qwen2.5-1.5B aligned vs benign-drifted)
Probe = aligned model's refusal direction at layer 14 (fixed before seeing transformed results), threshold = 95th pct of clean harmless_val scores.
Behaviour = substring refusal on the agent's reply to the same input (preexisting detector).

| condition | aligned refuses H/S | drifted refuses H/S | probe on aligned: TPR/FPR, AUC | probe on drifted: TPR/FPR, AUC |
|---|---|---|---|---|
| direct | 0.87/0.08 | 0.46/0.01 | 0.97/0.11, 0.98 | 0.98/0.14, 0.98 |
| rot13 | 0.96/0.98 | 0.03/0.04 | 0.24/0.09, 0.65 | 0.00/0.00, 0.52 |
| base64 | 0.14/0.18 | 0.00/0.00 | 1.00/0.97, 0.59 | 0.45/0.38, 0.58 |
| leet | 1.00/0.99 | 0.93/0.68 | 1.00/0.96, 0.72 | 0.91/0.58, 0.75 |
| spaced | 0.68/0.09 | 0.40/0.04 | 0.95/0.31, 0.92 | 0.85/0.17, 0.92 |

Findings (cautious):
1. **Knows but does not refuse.** On direct requests the drifted agent refuses 46% but the aligned-direction probe read from the drifted agent's own activations flags 98% (FPR 14%, AUC 0.98). Same pattern with spaced letters (refuses 40%, probe TPR 0.85 at FPR 0.17). The internal signal outlives the behaviour; the probe is a candidate monitor for drifted agents. Caveat: threshold is calibrated on harmless_val and gave FPR 11-14% on harmless_test, not the 5% targeted.
2. **Obfuscation defeats both.** rot13/base64: AUC 0.52-0.65, drifted agent complies (0-3% refusal). The aligned agent's rot13/leet "refusals" are indiscriminate (harmless refused 98-99%), i.e. confusion, not detection. The prompted 1.5B judge is near chance even on clean text (AUC 0.56) so it is not a usable baseline here.
3. **Splitting (mechanical word-midpoint cut):** probe AUC stays 0.93-0.99, so no evasion gap for the probe, but harmless fragments also raise alarms (FPR 0.21-0.33). Behaviourally, fragment refusals are mostly confusion (aligned refuses 72% of harmless first halves), so behavioural fragment monitors are not meaningful. This differs from the UK AISI result (secondhand summary) and probably reflects our crude split; do not generalise.
4. Perplexity flags obfuscation (rot13 100%) but not harm; its combination with the probe is untested.

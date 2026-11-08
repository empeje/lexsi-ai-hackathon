# Apart "Swarm Alignment" sprint: what can we reuse?

Sprint: Nov 6-8, 2026, online. Submissions close **Sun Nov 8, 11:59 PM AoE**. Teams of 1-5, max 2 projects per person.
Deliverable: <=8 page PDF (official template), 150-250 word abstract, required "Limitations and Dual-Use / Ethical Considerations" appendix. Repo and 3-5 min video optional.
Judging: Impact & Innovation, Execution Quality, Presentation & Clarity (1-5 each).
Source: https://apartresearch.com/sprints/swarm-alignment-2026

## REVISED FRAMING (v2): SafeTune/Lexsi as the toolkit, Apart's tracks as the questions
Instead of "our drift paper plus a swarm angle", ask: **which SafeTune interventions, applied per agent, actually protect a swarm?**
SafeTune is a library of methods organised as Interventions (`harden` train-time, `recover` weight-space, `steer` inference-time) and Instrumentation (`interpret`, `evaluate`/redteam). We used only part of `recover`. The rest maps onto Apart's tracks:

| SafeTune piece (installed in `.venv`) | Apart track and example project |
|---|---|
| `steer.probe_guard`, `steer.cast`, `steer.safeswitch`, `steer.refusal_direction` | **Track 4**: "Catch the swarm at the gateway", "Replicas as a monitor". A probe/guard as a per-agent or gateway monitor on inter-agent messages. |
| `evaluate` + `evaluate/redteam` stressors, judges | **Track 1 / 4**: a ready-made attack and scoring harness for swarm red-teaming, instead of our substring metric. |
| `harden.*` (e.g. `safegrad`, `lisa`, `booster`, `vaccine`), `recover.*` | **Track 2**: "Safety in numbers?" / "One bad apple". Mix hardened, drifted and repaired agents and measure swarm outcomes by topology (chain, star, fully connected). |
| `provenance.py` (`lexsi_provenance.json`, lineage chained across checkpoints) | **Track 3**: "Who is who in a swarm?" Agent ID and lineage when agents copy or fine-tune themselves. Not yet read in depth. Check whether records are signed or only self-reported before claiming anything. |
| `interpret.safety_neurons`, `interpret.eap` | **Track 4**: locate where a swarm-level failure is represented; feeds monitors. |

**Revised recommendation:** "Per-agent safety interventions vs swarm-level safety" under Track 2, cross-listed with Track 4.
- Independent variable: which SafeTune intervention each agent gets (none / recover / harden / steer-guard), and how many agents.
- Dependent variables: swarm-level compliance with harmful requests, task accuracy, and monitor detection rate, across topologies.
- Our earlier offline simulation (#1 below) becomes the cheap first experiment; the probe monitor (#2) becomes one arm of the comparison.

**Caveat to check first:** SafeTune's feature map labels each method with a faithfulness badge. Only methods marked as faithful may be cited as the named method; others "run but must not be cited". Read `docs/reference/feature-map.md` (online) before choosing which methods to feature, and note it in the report.

---
(Original v1 analysis follows.)

## Hard constraints that shape the choice
- **Only the weekend's work is judged.** Pre-sprint code must be flagged as preexisting in the README and report. The drift paper itself cannot be the submission; the new contribution has to be swarm-level.
- **Made-up results can disqualify.** Everything must come from real runs.
- **About one day left.** Anything needing new large models, new datasets or new training is out. Prefer ideas that run on the 1.5B Qwen models on the laptop, or offline on results we already have.
- Check the Lexsi hackathon rules for double-submission restrictions before reusing results.

## What we have that is actually reusable
| Asset | Why it matters for swarms |
|---|---|
| Aligned (Qwen2.5-1.5B-Instruct) and **benignly drifted** (GSM8K LoRA) models | A naturally weakened agent, with no adversarial intent. Most swarm work assumes an attacker. |
| Per-prompt refusal flags for ~12 variants (`outputs/final/*.json`) | Lets us simulate voting, routing and ensembles **offline, with no new compute**. |
| Located refusal direction per layer, AUC, causal tests (`s02_locate.py`, `circuits.py`, `outputs/refdirs_*.pt`) | A white-box signal that could serve as a monitor. |
| Weight-space tools (task vectors, LoRA deltas, SafeTune) | Can fingerprint what changed in an agent's weights. |
| Negative result: Tiny Aya shows no drift under the same fine-tune | Drift is model-specific, so a heterogeneous swarm has uneven weak links. |

## Candidate topics, ranked

### 1. "Refusal shopping" with benign-drifted agents (Track 2). **Do this.**
The sprint lists "Refusal shopping" and "Safety in numbers?" as example projects. Our twist: the weak agent is a specialist that was fine-tuned on a harmless task, not a jailbroken one.
- Pool of N agents, some aligned and some drifted. A router, or a requester who retries, sends a harmful request to whichever agent complies.
- Measure swarm-level compliance against the fraction of drifted agents, under routing policies: random, retry-until-comply, majority vote, and "any refusal blocks".
- Much of this can be simulated from existing per-item flags, then confirmed with a small live run.
- Finding to aim for: how much benign specialization in a few agents leaks safety for the whole swarm, and which aggregation rule is robust.

Effort: low to medium. Fit with our assets: very high. Novelty: moderate, because benign drift as the weak link is a fresh angle.

### 2. Refusal-direction probe as a swarm monitor (Track 4). **Pair with #1 as the defence.**
Use the located refusal direction to flag when a message or sub-request activates harmful-intent features in a receiving agent, and compare it with a text-level monitor. Related reading: Rose et al. 2026 on multi-agent interpretability.
- We already have AUC numbers per layer. The new work is applying the probe to messages passed between agents.
- Natural extension: does the probe still fire on the drifted agent, where refusal is suppressed but the direction may persist? Our `s02` results on instruct vs drifted give a head start.

Effort: medium. Risk: probe may be weaker on rephrased or split requests, but that is itself a reportable result.

### 3. Task-splitting past per-agent filters (Track 2: "Split the task, skip the filter"). Maybe.
Test whether decomposing a harmful request into benign-looking subtasks across agents evades per-agent refusal, and whether the direction probe from #2 catches it.
- Strong fit with #2, but needs a decomposition dataset. Writing one raises dual-use concerns, so keep it to existing public benchmarks and describe attacks at a high level.
- Only do this if #1 and #2 are done early.

### 4. Weight-delta fingerprints as agent identity (Track 3). Stretch, probably skip.
Answers "how should an agent ID work when agents can fine-tune themselves": detect that a copy was fine-tuned, and by how much, from weight deltas or refusal-direction shift, and use that for revocation.
- Interesting and distinct, but needs a new experimental design and enough variety of fine-tunes. Not enough time to do it well alongside the above.
- Worth keeping as "future work" in the report.

### Not worth pursuing
- **Track 1 sandboxes:** needs agent infrastructure we do not have.
- **Swarm Traces / AI Village datasets:** large new data, new tooling, little overlap with our models.
- **Repackaging the repair results as a swarm paper:** violates the "weekend work only" rule and would score poorly on innovation.

## Recommended project
**One paper, two parts:** "Benign drift as a swarm's weak link: how aggregation rules and a refusal-direction monitor behave when some agents have drifted."
Anchor track: Track 2 (Science of Agent Networks), cross-listed with Track 4.

Proposed order of work:
1. Offline simulation from existing flags: swarm compliance vs fraction of drifted agents, under four aggregation rules (a few hours).
2. Small live run on the Qwen models to confirm the simulation (a few hours).
3. Refusal-direction monitor on inter-agent messages, compared with a substring/text baseline.
4. Write-up in the official template: 8 pages, abstract, Limitations and Dual-Use appendix. Mark `s00`-`s11`, `repair.py`, `circuits.py` and the data as **preexisting**.
5. Optional: short video and public repo.

## Open questions for you
- Is the team just you, or are others joining? That decides whether #3 is feasible.
- Does the Lexsi hackathon allow reusing its models and results elsewhere?
- Do you want the new work in a separate directory (e.g. `swarm/`) so the preexisting/new split is obvious to judges? I recommend it.

> **Note:** superseded. The submitted project is described in `swarm/README.md` and `swarm/SUBMISSION.md`.

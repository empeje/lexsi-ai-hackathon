# What the sprint talks suggest for our project

Sources: Zoom recordings linked from the sprint schedule. Full transcripts read for Stengel-Eskin, Shenk and Gutenberg. For Kitaoka, Fleming and Howard-Jones only Zoom's auto-generated summaries were read, so treat their points as secondhand and verify before citing. Not read: Rosenblatt keynote, logistics talks, Adaption workshop; Kemeklis has no recording. Cite the talks only as "talk, Apart Swarm Alignment Sprint 2026".

## Talk-by-talk takeaways and what we do with them

| Talk | Idea | How it shapes our project |
|---|---|---|
| **Kitaoka** (MATS / UK AISI) [summary only] | Splitting a misaligned goal across several agents evades monitors far more often than a single-agent attack; there is an "evasion-rate gap". | Measure the **single vs split evasion gap** for our monitors (E3). Report it as the headline monitor metric. |
| **Fleming** (AE Studio) [summary only] | Agents are pressure-sensitive and replicable; about one third leaked a secret even with no adversarial swarm, and a swarm raised the rate. | Treat **peer pressure** as an arm: does an aligned agent's refusal hold when peers say they complied (E4)? Drift is also a *replicable* weakness: one fine-tune recipe can be copied to many agents. |
| **Howard-Jones** (Swarmchaser) [summary only] | Real swarm traffic is low-signal; behaviour is often driven by constraints, so don't over-anthropomorphise. | Describe behaviours operationally (refuse / comply / relay), avoid intent language, and state in Limitations that substring refusal is a coarse proxy. |
| **Stengel-Eskin** (UT Austin) | Agents can invent compressed, non-English protocols under budget pressure plus a reflection channel, defeating regex/NLP monitors. Perplexity under a language model is a cheap detector. Weaker open models could not invent languages but could **learn** them, so a few agents may poison a community. Open question named by the speaker: detection is the low-hanging fruit. | Our 1.5B models will not invent languages, so we test the **monitoring half**: does a refusal-direction probe still fire when a harmful request is obfuscated or compressed, compared with a text/perplexity monitor (E3). Also supports the "weak agents still spread behaviour" angle. |
| **Shenk** (independent) | Describe swarms by mechanism, function and lineage; build an **ethogram**; look for activation patterns that track swarm-level changes; fleets (centralised) vs swarms (decentralised); do weaker agents adapt to stronger ones? Alignment as a vector: cosine similarity as a simple model. | Our **mechanism** is the refusal direction, **lineage** is aligned -> drifted -> repaired, **function** is refusing/complying. Ethogram table of agent behaviours (E5). Cosine between refusal directions of aligned vs drifted agents as the "alignment vector" measure. Compare centralised router (fleet) with peer-to-peer (swarm). |
| **Gutenberg** (Yan, Sikora) | Incident science: reenact **one chunk** of an incident in a toy environment; "memetic misalignment": does the marginal cost of adding the k-th misaligned agent accelerate? Chunk of the Hugging Face incident: requesting help from other agents. Public replication that names what info is missing creates pressure for disclosure. | E2 **marginal-agent curve** (we already see thresholds: majority fails at 3 of 5, shopping at 1 of 5). E4 reenacts the "request help from another agent" chunk as refusal shopping. Our report lists exactly what we could not test and why. |

## Resulting experiment set (E1-E5), in priority order

1. **E1 Aggregation rules with benign-drifted agents.** Done offline from preexisting flags (`sim_routing.py`, `sim_cost.py`).
2. **E2 Marginal-agent curve (memetic drift).** Harm compliance as drifted agents go 0 -> 5 under each rule; flag thresholds. Offline, cheap; extend with sampling live.
3. **E3 Monitor evasion gap.** Single request vs split vs obfuscated request; compare SafeTune `probe_guard` / refusal-direction probe against a text monitor (substring/perplexity). Needs live forward passes on the Qwen models.
4. **E4 Peer pressure and routing, live.** Small multi-agent loops (chain / star / fully connected) where an agent sees peers' prior answers; measure whether refusals flip. Reenacts "request help from another agent".
5. **E5 Lineage and ethogram.** Table of behaviours per agent variant plus refusal-direction cosine across lineage (aligned, drifted, repaired). Stretch: use Lexsi provenance records as lineage.

## Dual-use guardrails
Use existing public benchmark prompts only. Do not publish new split/obfuscation recipes beyond aggregate rates; describe attacks at a high level in the report and put the obfuscation methods used in general terms.

# Submission pack: Swarm Alignment Sprint 2026, Track 2

**Title:** Benign Drift as a Swarm's Weak Link: Aggregation, Peer Framing and What a Drifted Agent Still Knows
**Author:** Abdurrachman Mappuji
**Deadline:** Sun Nov 8, 2026, 11:59 PM AoE

## Abstract
A swarm of language-model agents is only as safe as the agents it routes requests to. Most swarm-safety work assumes an adversarial member; we study a quieter weak link, an agent fine-tuned on a harmless task that lost part of its refusal behaviour without anyone intending it. Using Qwen2.5-1.5B-Instruct and a copy fine-tuned on grade-school maths, we ask how such an agent changes group safety, whether drifted agents spread or absorb refusal through shared context, and what a drifted agent still represents about harmful requests. Simulated aggregation shows that one drifted agent in a pool of five raises the share of harmful requests answered from 6.5% to 41% when a requester can try every agent, while majority voting holds until three agents have drifted. In live four-agent chains, one drifted agent raises the share answered by some agent from 31% to 52–56%, with small further increments; a controlled test finds that a refusing peer reply lowers refusal by 14–18 points, the opposite of contagion, which we attribute partly to framing. The drifted agent refuses 56% of harmful requests, yet its hidden states and output likelihoods flag 91–92% of them at 5–9% false alarms, and an aligned instance reads the signal equally well. Simple encodings defeat the hidden-state probe at base64 and both readouts at rot13. We report what we could not test, including hosted models and adversarial coordination.

## Files
- Paper: `swarm/paper/main.pdf` (source `swarm/paper/main.tex`, 6 pages; plain two-column layout, not yet Apart's official template)
- Project image: `swarm/paper/figures/project_image.png`
- Code and results: `swarm/` (see `swarm/README.md` for run order); branch `apart/swarm-alignment-2026`

## Preexisting vs new
Everything outside `swarm/` is preexisting (SafeTune hackathon repo, models, drift recipe, prompts, refusal detector; cited as Mappuji 2026, doi:10.5281/zenodo.24040671). All experiments, analyses, figures and the paper in `swarm/` were produced during the sprint.

## Known gaps (stated in the paper)
1.5B models only, greedy decoding, substring refusal detector, no hosted-model validation (credits unavailable), monitors share a model family with the agents, framing mechanism partly unexplained.

## Before submitting
- [ ] Port to the official Apart template (copy the Google Docs template yourself)
- [ ] Read the paper end to end, including the author block
- [ ] Make the repo/branch public and paste its URL
- [ ] Upload the project image and PDF

"""Offline swarm simulation from preexisting per-prompt refusal flags (outputs/final/*.json).

Input data is PREEXISTING; the aggregation analysis here is new. Greedy decoding means identical
copies of a model behave identically, so only heterogeneous pools are informative here. Topology
and message-passing effects need live runs (see swarm/ later scripts).
"""
import json, glob, os, itertools, random, sys
OUT = os.path.join(os.path.dirname(__file__), "results")
FINAL = os.path.join(os.path.dirname(__file__), "..", "outputs", "final")
EXCLUDE = {"uncond_steer_x1"}  # refuses ~everything (over-refusal 98%, GSM8K 0%): degenerate

flags = {}
for f in sorted(glob.glob(os.path.join(FINAL, "*.json"))):
    name = os.path.basename(f)[:-5]
    if name in EXCLUDE: continue
    flags[name] = json.load(open(f))["flags"]["refusal_harmful"]  # True = refused
N = len(flags["instruct"]); assert all(len(v) == N for v in flags.values())
comply = {k: [0 if r else 1 for r in v] for k, v in flags.items()}  # 1 = complied with harmful prompt

def boot_ci(xs, B=2000, seed=0):
    rng = random.Random(seed); n = len(xs)
    ms = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return ms[int(.025 * B)], ms[int(.975 * B) - 1]

def rate(per_prompt): m = sum(per_prompt) / len(per_prompt); lo, hi = boot_ci(per_prompt); return round(m, 3), round(lo, 3), round(hi, 3)

res = {"n_prompts": N, "single": {k: rate(v) for k, v in comply.items()}}

# Rules over a pool of agents (list of variant names; one agent per slot). Harmful compliance rate (lower = safer swarm).
def rules(pool):
    cols = [comply[a] for a in pool]; k = len(pool)
    per = list(zip(*cols))
    return {
        "random_agent": rate([sum(p) / k for p in per]),            # request sent to one uniformly random agent
        "shop_any": rate([1 if any(p) else 0 for p in per]),         # requester tries every agent, takes a compliant one
        "majority": rate([1 if sum(p) * 2 > k else 0 for p in per]), # swarm answers if majority would comply
        "unanimous": rate([1 if all(p) else 0 for p in per]),        # swarm answers only if all comply
    }

# A: 5-agent pools of instruct + drifted, k drifted (copies are identical under greedy -> shows the shopping/vote structure)
res["instruct_plus_drifted"] = {str(k): rules(["drifted"] * k + ["instruct"] * (5 - k)) for k in range(0, 6)}
# B: heterogeneous: the aligned agent plus j repaired variants, with the drifted agent (pool = all non-degenerate variants)
names = sorted(flags)
res["all_variants_pool"] = rules(names)
res["pool_members"] = names
json.dump(res, open(os.path.join(OUT, "sim_routing.json"), "w"), indent=1)

print("single-agent harmful compliance (1 - refusal), 95% bootstrap CI")
for k, v in sorted(res["single"].items(), key=lambda kv: kv[1][0]): print(f"  {k:18s} {v[0]:.3f} [{v[1]:.3f},{v[2]:.3f}]")
print("\n5-agent pools of instruct + k drifted: compliance under each rule")
print("  k  " + "  ".join(f"{r:>13s}" for r in ["random_agent", "shop_any", "majority", "unanimous"]))
for k, r in res["instruct_plus_drifted"].items():
    print(f"  {k}  " + "  ".join(f"{r[x][0]:13.3f}" for x in ["random_agent", "shop_any", "majority", "unanimous"]))
print("\nall-variant pool:", {x: res['all_variants_pool'][x][0] for x in res['all_variants_pool']})

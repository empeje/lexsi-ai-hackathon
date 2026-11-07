"""Generate LaTeX tables from swarm/results/*.json so paper numbers cannot drift from the data (new this weekend)."""
import json, math, os, random
H = os.path.dirname(__file__); R = os.path.join(H, "results"); T = os.path.join(H, "paper", "tables"); os.makedirs(T, exist_ok=True)
def J(n): p = os.path.join(R, n); return json.load(open(p)) if os.path.exists(p) else None
def wil(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - h) / d, (c + h) / d
f2 = lambda x: f"{x:.2f}".lstrip("0") if x < 1 else "1.0"
def w(name, s): open(os.path.join(T, name), "w").write(s)

# control table (E4b)
c = J("e4b_control_n200.json")
if c:
    cols = [("none", "none"), ("neutral", "neutral"), ("refusal", "peer refuses"), ("compliant", "peer complies")]
    s = "\\begin{tabular}{lcccc}\\toprule\nReceiver & " + " & ".join(l for _, l in cols) + "\\\\\\midrule\n"
    for m in ("aligned", "drifted"):
        row = []
        for k, _ in cols:
            fl = c[m][k]["flags"]; n = len(fl); lo, hi = wil(sum(fl), n); row.append(f"{f2(sum(fl)/n)} {{\\scriptsize[{f2(lo)},{f2(hi)}]}}")
        s += m.capitalize() + " & " + " & ".join(row) + "\\\\\n"
    w("tab_control.tex", s + "\\bottomrule\\end{tabular}\n")

# chain table (E4)
ch = J("e4_chain_n200.json")
if ch:
    order = [x for x in ["AAAA", "DAAA", "ADAA", "AAAD", "ADAD", "DDDA", "ADDD", "DDDD"] if x in ch["comps"]]
    s = "\\begin{tabular}{lccccc}\\toprule\nChain & p1 & p2 & p3 & p4 & Answered by some agent\\\\\\midrule\n"
    rng = random.Random(0)
    for comp in order:
        v = ch["comps"][comp]; fl = v["refused_flags"]; n = len(fl[0])
        anyc = [int(any(not fl[k][i] for k in range(len(fl)))) for i in range(n)]
        bs = sorted(sum(anyc[rng.randrange(n)] for _ in range(n)) / n for _ in range(1000))
        s += f"{comp} & " + " & ".join(f2(x) for x in v["refusal_by_position"]) + f" & {f2(sum(anyc)/n)} {{\\scriptsize[{f2(bs[25])},{f2(bs[974])}]}}\\\\\n"
    w("tab_chain.tex", s + "\\bottomrule\\end{tabular}\n")

# probe vs behaviour table (E3b + E3c)
b, m = J("e3b_behavior_n200.json"), J("e3c_monitor.json")
if b and m and m["meta"]["n"] >= 100:
    r = m["results"]; N = len(b["drifted"]["direct"]["refuse_harmful"])
    s = ("\\begin{tabular}{lcccccc}\\toprule\n & A refuses & D refuses & \\multicolumn{2}{c}{Probe on D} & \\multicolumn{2}{c}{Refusal-lik. of D}\\\\\n"
         " & & & TPR/FPR & AUC & TPR/FPR & AUC\\\\\\midrule\n")
    for cnd in ["direct", "spaced", "leet", "base64", "rot13"]:
        a = sum(b["instruct"][cnd]["refuse_harmful"]) / N; d = sum(b["drifted"][cnd]["refuse_harmful"]) / N
        p, q = r["probe_drifted"][cnd], r["refusal_lp_drifted"][cnd]
        s += f"{cnd} & {f2(a)} & {f2(d)} & {f2(p['tpr'])}/{f2(p['fpr'])} & {f2(p['auc'])} & {f2(q['tpr'])}/{f2(q['fpr'])} & {f2(q['auc'])}\\\\\n"
    w("tab_probe.tex", s + "\\bottomrule\\end{tabular}\n")
# full AUC table with CIs (appendix)
if m:
    r = m["results"]; cs = ["direct", "spaced", "leet", "base64", "rot13", "split_swarm_max"]; hdr = ["direct", "spaced", "leet", "base64", "rot13", "split"]
    names = [("probe_aligned", "Probe (A)"), ("probe_drifted", "Probe (D)"), ("refusal_lp_aligned", "Refusal-lik. (A)"), ("refusal_lp_drifted", "Refusal-lik. (D)"), ("keyword", "Keyword"), ("ppl", "Perplexity")]
    s = "\\begin{tabular}{l" + "c" * len(cs) + "}\\toprule\nMonitor & " + " & ".join(hdr) + "\\\\\\midrule\n"
    for k, lab in names:
        s += lab + " & " + " & ".join(f"{r[k][c]['auc']:.2f} {{\\scriptsize[{r[k][c]['auc_ci'][0]:.2f},{r[k][c]['auc_ci'][1]:.2f}]}}" for c in cs) + "\\\\\n"
    w("tab_auc.tex", s + "\\bottomrule\\end{tabular}\n")
# aggregation cost table (E1, preexisting flags, n=200 harmful / 150 harmless)
sc = J("sim_cost.json")
if sc:
    rules = [("random_agent", "random"), ("shop_any", "shop"), ("majority", "majority"), ("unanimous", "unanimous")]
    s = "\\begin{tabular}{l" + "c" * len(rules) + "}\\toprule\nPool & " + " & ".join(l for _, l in rules) + "\\\\\\midrule\n"
    for key, lab in [("instruct+0drifted", "5 aligned"), ("instruct+1drifted", "4 A + 1 D"), ("instruct+3drifted", "2 A + 3 D"), ("instruct+5drifted", "5 drifted"), ("all_variants", "11 variants")]:
        s += lab + " & " + " & ".join(f"{sc[key][k]['harmful_compliance'][0]*100:.0f}/{sc[key][k]['harmless_false_refusal'][0]*100:.0f}" for k, _ in rules) + "\\\\\n"
    w("tab_cost.tex", s + "\\bottomrule\\end{tabular}\n")
print("tables:", sorted(os.listdir(T)))

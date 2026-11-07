"""Publish the hackathon artifacts to Hugging Face in three phases (uses the login already stored by `hf auth login`).

  python hf_publish.py stage     # build hf_release/{model,dataset} locally (no network writes)
  python hf_publish.py upload    # create PRIVATE repos and upload everything
  python hf_publish.py publish   # make the repos public and create/fill the two public collections
"""
import csv, json, random, shutil, sys
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).parent
USER = "azizamm"
MODEL = f"{USER}/qwen2.5-1.5b-instruct-gsm8k-safety-drift"
DATASET = f"{USER}/safety-drift-repair-results"
GITHUB = "https://github.com/empeje/lexsi-ai-hackathon"
REL = ROOT / "hf_release"
OUT = ROOT / "outputs"
api = HfApi()

MODEL_CARD = f"""---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: transformers
pipeline_tag: text-generation
tags:
- safety
- safety-drift
- fine-tuning
- lora
- refusal
- research
---

# Qwen2.5-1.5B-Instruct fine-tuned on GSM8K (safety-drift research artifact)

**This is a research artifact, not a safe assistant.** Benign fine-tuning on grade-school math measurably reduced this model's refusal of harmful requests. It is published so that the drift, and the training-free repairs we study, can be reproduced. Do not deploy it where safety matters.

## What it is
`Qwen/Qwen2.5-1.5B-Instruct` with a LoRA adapter (rank 16, alpha 32, dropout 0, all attention and MLP projections) trained for one epoch on 2,000 GSM8K training examples (shuffle seed 0; learning rate 2e-4 with linear decay and a 10-step warm-up; batch size 8; maximum length 320; loss on the answer tokens only), then merged into the base weights. bfloat16, trained on an Apple M1 GPU (MPS). No safety data were used. The adapter was not saved separately; these are the merged weights. Tokenizer files are copied unchanged from the base model.

## Measured effect (greedy decoding; 95% Wilson intervals in the paper)
| | Aligned base | This model |
|---|---|---|
| Refusal, harmful prompts (n=200) | 93.5% | **59.5%** |
| Refusal, AdvBench (n=100) | 100% | 90.0% |
| Refusal, XSTest unsafe (n=200) | 98.5% | 55.0% |
| Over-refusal, XSTest safe look-alikes (n=250) | 50.8% | 3.2% |
| Over-refusal, plain harmless prompts (n=150) | 8.0% | 2.7% |
| GSM8K accuracy (n=150) | 51.3% | 50.7% |

Refusal is detected by a phrase match in the first 200 characters of the response, a crude proxy; treat absolute levels with caution and rely on differences between models scored the same way.

## Also in this repo
`refusal_directions/refdirs_instruct.pt` and `refdirs_drifted.pt`: per-layer unit refusal directions (tensor of shape `[28, 1536]`; difference of mean last-prompt-token activations on harmful versus harmless prompts) for the aligned and the drifted model. Load with `torch.load`.

## Links
- Code, paper and scripts: {GITHUB}
- Results and evaluation splits: https://huggingface.co/datasets/{DATASET}
- Paper: *Repairing Safety Drift Without Retraining in Small Fine-Tuned Language Models: Weight-Space Methods and a Located Refusal Direction* (A. Mappuji), in the code repository under `paper/`.

## Limitations
One model, one drift source, one seed; small evaluation sets; a phrase-match refusal detector; no adversarial (jailbreak) evaluation. See the paper's limitations section.
"""

DATASET_CARD = f"""---
license: other
license_name: mixed-see-card
pretty_name: Safety-drift repair results and evaluation splits
language:
- en
tags:
- safety
- red-teaming
- evaluation
- refusal
size_categories:
- n<1K
---

# Safety-drift repair: results, logs and evaluation splits

Results, logs and the exact evaluation prompts behind the paper *Repairing Safety Drift Without Retraining in Small Fine-Tuned Language Models* (code: {GITHUB}; model: https://huggingface.co/{MODEL}).

## Contents
- `data/eval_splits.json` - the exact prompts and indices used: 200 harmful prompts, 100 AdvBench prompts, 200 XSTest unsafe, 250 XSTest safe, 150 plain harmless prompts, and the indices of the 150 GSM8K test questions.
- `results/qwen/final/*.json` - Qwen2.5-1.5B-Instruct: aligned, drifted and ten repair configurations, each with the six metrics, Wilson intervals, sample sizes and **per-prompt 0/1 outcomes** (`flags`). Metric names: `refusal_*` = refusal rate on harmful sets (higher is safer); `overrefusal_*` = refusal rate on safe sets (lower is better); `gsm8k_acc` = exact-match accuracy.
- `results/qwen/first_pass/*.json` - the earlier, smaller-sample runs (baseline, fine-tune, ReSta sweep, SafeTune method comparison, targeted repairs).
- `results/qwen/locate/` - per-layer refusal-direction analysis (separability, gap, causal ablate/add test) for the aligned and drifted model.
- `results/tiny_aya/summary.json` - Tiny Aya Global results (six configurations) and locate summary. **These values were transcribed from notebook output and checked by zoom; per-prompt outcomes were not retrieved** (see the `note` field).
- `logs/` - training and evaluation logs from the local Qwen runs.

## Licenses and sources
The result files and logs are our own work (CC BY 4.0 is intended for them). The prompt splits come from public sources and keep their original terms: XSTest (Röttger et al., CC BY 4.0), AdvBench (Zou et al., MIT), GSM8K (Cobbe et al., MIT), and the harmful/harmless splits compiled by Arditi et al. (2024) in `andyrdt/refusal_direction`, whose per-source licenses apply. Check those terms before redistributing the prompts commercially. This dataset contains harmful *requests* taken from public benchmarks; it contains **no** model completions to them.

## Caveats
Refusal is detected by a phrase match, a crude proxy. Samples are small (100-250 prompts per metric). Tiny Aya covers 6 of 12 planned configurations because SafeDelta exhausted CPU memory. SafeDelta ran in SafeTune's data-free mode.
"""


def sample(items, n, seed):
    return random.Random(seed).sample(items, n) if n < len(items) else items


def eval_splits():
    d = ROOT / "data"
    load = lambda f: [x["instruction"] for x in json.load(open(d / f))]
    advbench = [r["goal"] for r in csv.DictReader(open(d / "advbench.csv"))]
    xs = list(csv.DictReader(open(d / "xstest.csv")))
    return {
        "_note": "Sampling rules: harmful_test 200 = random.Random(0).sample(harmful_test.json, 200); advbench 100 = random.Random(0).sample(AdvBench, 100); "
                 "harmless 150 = random.Random(0).sample(harmless_test.json, 150); XSTest uses all safe (250) and unsafe (200) prompts; "
                 "GSM8K = indices into the 1,319-question test split, random.Random(0).sample(range(1319), 150).",
        "harmful_test_200": sample(load("harmful_test.json"), 200, 0),
        "advbench_100": sample(advbench, 100, 0),
        "xstest_unsafe_200": [r["prompt"] for r in xs if r["label"] == "unsafe"],
        "xstest_safe_250": [r["prompt"] for r in xs if r["label"] == "safe"],
        "harmless_150": sample(load("harmless_test.json"), 150, 0),
        "gsm8k_test_indices_150": random.Random(0).sample(range(1319), 150),
    }


def stage():
    if REL.exists(): shutil.rmtree(REL)
    m, ds = REL / "model", REL / "dataset"
    (m / "refusal_directions").mkdir(parents=True); (ds / "data").mkdir(parents=True)
    (m / "README.md").write_text(MODEL_CARD)
    src = OUT / "models" / "ft_gsm8k"
    for f in ("config.json", "generation_config.json"): shutil.copy(src / f, m / f)
    tok = Path(snapshot_download("Qwen/Qwen2.5-1.5B-Instruct", allow_patterns=["tokenizer*", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json"]))
    keep = ("tokenizer", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json")
    for f in tok.iterdir():  # tokenizer files ONLY: the cache folder may also hold the base model's weights
        if f.is_file() and f.name.startswith(keep): shutil.copy(f.resolve(), m / f.name)
    for f in ("refdirs_instruct.pt", "refdirs_drifted.pt"): shutil.copy(OUT / f, m / "refusal_directions" / f)
    (ds / "README.md").write_text(DATASET_CARD)
    json.dump(eval_splits(), open(ds / "data" / "eval_splits.json", "w"), indent=1)
    for sub, pat in (("qwen/final", "final/*.json"),):
        (ds / "results" / sub).mkdir(parents=True, exist_ok=True)
        for f in (OUT / "final").glob("*.json"): shutil.copy(f, ds / "results" / sub / f.name)
    fp = ds / "results" / "qwen" / "first_pass"; fp.mkdir(parents=True)
    for f in ("s00_baseline", "s01_ft_gsm8k", "s03_safetune_resta", "s04_safetune_methods", "s06_targeted"): shutil.copy(OUT / f"{f}.json", fp / f"{f}.json")
    shutil.copy(OUT / "results_table.md", fp / "results_table.md")
    lo = ds / "results" / "qwen" / "locate"; lo.mkdir(parents=True)
    for t in ("instruct", "drifted"): shutil.copy(OUT / f"s02_locate_{t}.json", lo / f"s02_locate_{t}.json")
    (ds / "results" / "tiny_aya").mkdir(parents=True); shutil.copy(ROOT / "results_tiny_aya" / "summary.json", ds / "results" / "tiny_aya" / "summary.json")
    (ds / "logs").mkdir()
    for f in OUT.glob("*.log"): shutil.copy(f, ds / "logs" / f.name)
    n = sum(1 for _ in REL.rglob("*") if _.is_file())
    print(f"staged {n} files under {REL}")


def upload():
    api.create_repo(MODEL, repo_type="model", private=True, exist_ok=True)
    api.create_repo(DATASET, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(repo_id=DATASET, repo_type="dataset", folder_path=str(REL / "dataset"), commit_message="Results, logs and evaluation splits")
    api.upload_folder(repo_id=MODEL, repo_type="model", folder_path=str(REL / "model"), commit_message="Model card, config, tokenizer, refusal directions")
    api.upload_file(repo_id=MODEL, repo_type="model", path_or_fileobj=str(OUT / "models" / "ft_gsm8k" / "model.safetensors"), path_in_repo="model.safetensors", commit_message="Merged fine-tuned weights (bf16)")
    print("uploaded (repos are still PRIVATE)")


def publish():
    api.update_repo_settings(MODEL, repo_type="model", private=False)
    api.update_repo_settings(DATASET, repo_type="dataset", private=False)
    cm = api.create_collection(title="Safety drift repair: models", namespace=USER, private=False,
                               description="Models and refusal-direction artifacts for the safety-drift repair study (Lexsi AI hackathon, Track 2b).", exists_ok=True)
    cd = api.create_collection(title="Safety drift repair: datasets and results", namespace=USER, private=False,
                               description="Results, per-prompt outcomes, logs and evaluation splits for the safety-drift repair study (Lexsi AI hackathon, Track 2b).", exists_ok=True)
    api.add_collection_item(cm.slug, item_id=MODEL, item_type="model", exists_ok=True)
    api.add_collection_item(cd.slug, item_id=DATASET, item_type="dataset", exists_ok=True)
    print("MODEL COLLECTION:  ", f"https://huggingface.co/collections/{cm.slug}")
    print("DATASET COLLECTION:", f"https://huggingface.co/collections/{cd.slug}")


if __name__ == "__main__":
    {"stage": stage, "upload": upload, "publish": publish}[sys.argv[1]]()

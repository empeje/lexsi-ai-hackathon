"""Builds colab/tiny_aya_pipeline.ipynb with the *current* project files embedded (so the notebook never drifts from the code)."""
import json
from pathlib import Path
ROOT = Path(__file__).parent
FILES = ["common.py", "circuits.py", "repair.py", "s01_finetune.py", "s02_locate.py", "s07_final.py"]
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}

cells = [md("""
# Tiny Aya safety-drift pipeline (Lexsi hackathon, Track 2b)
Fine-tune **CohereLabs/tiny-aya-global** on GSM8K (LoRA) -> measure safety drift -> locate the refusal direction -> evaluate training-free repairs (SafeTune weight-space methods + refusal-direction steering/patch).

**Before you run**
1. Runtime -> Change runtime type -> **GPU: A100 or L4** (needs bf16; a T4 will fall back to fp16 and may produce NaNs). High-RAM recommended (the repair stage holds three 3.35B models on CPU).
2. Left sidebar -> Secrets (key icon) -> add **HF_TOKEN** (a Hugging Face *read* token for an account that has accepted the licence on `CohereLabs/tiny-aya-global` and `tiny-aya-base`) and switch on *Notebook access*.
3. Run cells top to bottom. Each stage skips work that is already on disk, so it is safe to re-run after a disconnect (optional Drive cell below keeps results across sessions).
"""),
code("""
import sys, subprocess
print(sys.version)
if sys.version_info < (3, 12):
    print("WARNING: SafeTune needs Python >= 3.12. Stages 1-2 (fine-tune, locate) still work; the repair stage (s07) will fail to import safetune.")
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
"""),
code("""
!pip -q install "transformers==5.21.1" peft datasets accelerate matplotlib safetune
"""),
md("## Hugging Face login (token is read from your Colab Secret, never typed here)"),
code("""
from google.colab import userdata
from huggingface_hub import login, hf_hub_download
login(token=userdata.get("HF_TOKEN"))
for m in ["CohereLabs/tiny-aya-global", "CohereLabs/tiny-aya-base"]:
    hf_hub_download(m, "config.json")  # raises if licence not accepted
print("access OK")
"""),
md("## Project files (embedded copies of the repo scripts)"),
code("""
import os; os.makedirs("/content/work", exist_ok=True); os.chdir("/content/work")
""")]
for f in FILES:
    cells.append(code(f"%%writefile /content/work/{f}\n" + (ROOT / f).read_text()))
cells += [md("## Data (public mirrors of AdvBench, XSTest and the refusal-direction splits)"),
code("""
%%bash
mkdir -p /content/work/data && cd /content/work/data
curl -sfL https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv -o advbench.csv
curl -sfL https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv -o xstest.csv
for f in harmful_train harmful_val harmful_test harmless_train harmless_val harmless_test; do
  curl -sfL https://raw.githubusercontent.com/andyrdt/refusal_direction/main/dataset/splits/$f.json -o $f.json
done
ls -la
"""),
md("## Optional: keep results across sessions (asks you to authorise Google Drive)"),
code("""
# Uncomment to persist outputs on Drive:
# from google.colab import drive; drive.mount('/content/drive')
# os.makedirs('/content/drive/MyDrive/lexsi_outputs', exist_ok=True)
# if not os.path.exists('/content/work/outputs'): os.symlink('/content/drive/MyDrive/lexsi_outputs', '/content/work/outputs')
"""),
md("## Stage 1: GSM8K LoRA fine-tune (drift). 2,000 examples, MAXLEN 640 because Tiny Aya's chat template adds a 366-token system preamble"),
code("""
import os, pathlib, torch
os.chdir("/content/work")
os.environ.update(RUN_TAG="tiny-aya", MODEL_ID="CohereLabs/tiny-aya-global", BASE_ID="CohereLabs/tiny-aya-base")
if not pathlib.Path("outputs/tiny-aya/models/ft_gsm8k/config.json").exists():
    gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    os.environ.update(MAXLEN="640", MICRO_BS="4" if gb >= 38 else "2", SKIP_EVAL="1")
    !python s01_finetune.py 2000
else:
    print("fine-tuned model already on disk, skipping")
"""),
md("## Stage 2: locate the refusal direction (aligned vs drifted)"),
code("""
import pathlib
os.chdir("/content/work")
if not pathlib.Path("outputs/tiny-aya/s02_locate_instruct.json").exists():
    !python s02_locate.py CohereLabs/tiny-aya-global instruct
if not pathlib.Path("outputs/tiny-aya/s02_locate_drifted.json").exists():
    !python s02_locate.py outputs/tiny-aya/models/ft_gsm8k drifted
"""),
md("## Choose the repair band from the causal test (layers where ablating the direction removes refusal AND adding it induces refusal)"),
code("""
import json
d = json.load(open("outputs/tiny-aya/s02_locate_instruct.json"))["causal"]
ls = sorted(int(l) for l, v in d.items() if v["ablate_harm_refusal"] < 0.5 and v["add_harmless_refusal"] > 0.5)
print("causal layers:", ls)
assert ls, "no layer passed the causal test; inspect outputs/tiny-aya/s02_locate_instruct.json"
os.environ["BAND"] = f"{min(ls)}-{max(ls)}"; print("BAND =", os.environ["BAND"])
"""),
md("## Stage 3: six-metric final evaluation of every repair (resumable: one JSON per model in outputs/tiny-aya/final/)\nTA_ALPHA=0.25: SafeTune's docs report ReSta at alpha >= 0.5 breaks Tiny Aya."),
code("""
os.environ["TA_ALPHA"] = "0.25"
!python s07_final.py
"""),
md("## Download results (everything except the model weights)"),
code("""
!cd /content/work/outputs && zip -rq /content/tiny_aya_results.zip tiny-aya -x "tiny-aya/models/*"
from google.colab import files
files.download("/content/tiny_aya_results.zip")
""")]
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "colab" / "tiny_aya_pipeline.ipynb"; out.write_text(json.dumps(nb, indent=1)); print("wrote", out, len(cells), "cells,", out.stat().st_size // 1024, "KB")

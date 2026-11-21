"""Turn sleep_experiment.py (with # %% cell markers) into Colab notebooks:
sleep_experiment.ipynb (GPT-2) and sleep_experiment_qwen.ipynb (Qwen2.5, full and LoRA)."""
import json
import re
from pathlib import Path

here = Path(__file__).parent
src = (here / "sleep_experiment.py").read_text(encoding="utf-8")

QWEN_INTRO = """## Bigger models: Qwen2.5 on a free Colab GPU
**Runtime > Change runtime type > T4 GPU** first.

Pick a model and method in the next code cell:

| MODEL | METHOD | fits a T4? | time (approx.) |
|---|---|---|---|
| `Qwen/Qwen2.5-0.5B` | `full` | yes | ~15 min |
| `Qwen/Qwen2.5-0.5B` | `lora` | yes | ~10 min |
| `Qwen/Qwen2.5-1.5B` | `lora` | yes | ~25 min |
| `Qwen/Qwen2.5-3B` | `lora` | tight | ~50 min |

Comparing `full` and `lora` on the same model tests the paper *LoRA Learns Less and Forgets Less*
(Biderman et al., 2024). Comparing sizes tests whether bigger models forget less.
Each run saves its charts and `results.json` under `results/<model>-<method>/`."""


def cells_from(source):
    cells = []
    for chunk in re.split(r"^# %%", source, flags=re.M)[1:]:
        header, _, body = chunk.partition("\n")
        body = body.strip("\n")
        if header.strip().startswith("[markdown]"):
            text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
            cells.append(md(text))
        else:
            cells.append(code(body))
    return cells


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(True)}


def write(name, cells):
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
    (here / name).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", name, len(cells), "cells")


write("sleep_experiment.ipynb", cells_from(src))

qwen_src = src.replace('MODEL = os.environ.get("SLEEP_MODEL", "gpt2")',
                       'MODEL = "Qwen/Qwen2.5-0.5B"   # <- change me (see the table above)')
qwen_src = qwen_src.replace('METHOD = os.environ.get("SLEEP_METHOD", "full")',
                            'METHOD = "full"              # "full" or "lora"')
assert qwen_src != src
cells = cells_from(qwen_src)
cells[0]["source"] = [s.replace("A small language model (GPT-2)", "A language model (Qwen2.5)")
                      for s in cells[0]["source"]]
cells[1:1] = [md(QWEN_INTRO), code("!pip install -q peft")]
write("sleep_experiment_qwen.ipynb", cells)

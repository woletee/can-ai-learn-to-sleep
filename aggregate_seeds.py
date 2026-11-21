"""Combine results.json from several seeds: mean and spread per condition, plus a chart."""
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = ["results/gpt2-full", "results/gpt2-full-seed1", "results/gpt2-full-seed2"]
SKIP = {"after learning A"}

per = {}   # label -> list of (A, B)
for run in RUNS:
    path = os.path.join(HERE, run, "results.json")
    if not os.path.exists(path):
        print("missing", path)
        continue
    for s in json.load(open(path))["summary"]:
        if s["label"].startswith("pre-trained") or s["label"] in SKIP:
            continue
        per.setdefault(s["label"], []).append((s["A"] * 100, s["B"] * 100))

labels = list(per)
rows = []
print("| condition | seeds | old facts A: mean (min–max) | new facts B: mean (min–max) |")
print("|---|---|---|---|")
for lab in labels:
    a = [x[0] for x in per[lab]]
    b = [x[1] for x in per[lab]]
    rows.append((lab, st.mean(a), st.pstdev(a), st.mean(b), st.pstdev(b), a, b))
    print(f"| {lab} | {len(a)} | {st.mean(a):.0f}% ({min(a):.0f}–{max(a):.0f}) | {st.mean(b):.0f}% ({min(b):.0f}–{max(b):.0f}) |")

fig, ax = plt.subplots(figsize=(11, 5))
x = range(len(rows))
ax.bar([i - 0.2 for i in x], [r[1] for r in rows], 0.4, yerr=[r[2] for r in rows], capsize=4,
       label="old facts (A)", color="#2e86ab")
ax.bar([i + 0.2 for i in x], [r[3] for r in rows], 0.4, yerr=[r[4] for r in rows], capsize=4,
       label="new facts (B)", color="#f4a259")
for i, r in enumerate(rows):   # individual seeds as dots
    ax.scatter([i - 0.2] * len(r[5]), r[5], color="#1b3a4b", s=12, zorder=3)
    ax.scatter([i + 0.2] * len(r[6]), r[6], color="#7a4a12", s=12, zorder=3)
ax.set_xticks(list(x))
ax.set_xticklabels([r[0].replace(" (", "\n(") for r in rows], fontsize=8.5)
ax.set_ylabel("recall after learning B (%)")
ax.set_ylim(0, 105)
ax.set_title(f"GPT-2, mean of {max(len(v) for v in per.values())} seeds (bars) and each seed (dots)")
ax.legend()
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
out = os.path.join(HERE, "results", "seeds_summary.png")
fig.savefig(out, dpi=160)
json.dump({lab: per[lab] for lab in labels}, open(os.path.join(HERE, "results", "seeds_summary.json"), "w"), indent=1)
print("wrote", out)

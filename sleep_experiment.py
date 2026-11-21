# %% [markdown]
# # Can AI learn to sleep?
# **Catastrophic forgetting and sleep-like replay in a real language model.**
#
# 1. A small language model (GPT-2) learns 120 made-up facts: **"yesterday's memories" (set A)**.
# 2. It then learns 120 *new* made-up facts (set B). We watch how much of set A it forgets.
# 3. We repeat step 2 three ways, each starting from the same model:
#    - **No sleep**: learn B only.
#    - **Replay (stored memories)**: mix a stored sample of A back in, like a replay buffer.
#    - **Dream replay (no stored data)**: before learning B, the model *generates its own* facts
#      from memory ("dreams"), and those are mixed in. Nothing from A is stored. This is
#      generative replay, the idea closest to what the hippocampus does during sleep.
#
# The facts are invented (e.g. "The capital of Zorvia is Plinth"), so the model can't already know
# them, and recall is easy to check exactly. Runs on a free Colab GPU in a few minutes, or on a CPU.

# %%
# On Colab, transformers and matplotlib are already installed. Elsewhere:
#   pip install torch transformers matplotlib
import copy
import json
import math
import os
import random
import re
import time

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

QUICK = os.environ.get("SLEEP_QUICK") == "1"  # tiny smoke-test run

# Model and training method. "full" = update every weight. "lora" = train small adapter
# matrices only (needs `pip install peft`). On a free Colab T4 GPU:
#   gpt2 (124M) full | Qwen/Qwen2.5-0.5B full | Qwen/Qwen2.5-1.5B lora | Qwen/Qwen2.5-3B lora
MODEL = os.environ.get("SLEEP_MODEL", "gpt2")
METHOD = os.environ.get("SLEEP_METHOD", "full")

CFG = dict(
    model=MODEL,
    method=METHOD,
    n_facts=24 if QUICK else 120,   # facts per set (A and B)
    epochs_a=4 if QUICK else 30,    # epochs to learn set A (the first set also has to learn the format)
    epochs_b=4 if QUICK else 15,    # epochs to learn set B
    lr=2e-4 if METHOD == "lora" else 5e-5,
    lora_rank=16,
    warmup=10,              # linear learning-rate warm-up steps (avoids a big first-update shock)
    batch=16,
    replay_frac=0.25,       # replay buffer size, as a fraction of set A
    eval_every=10,          # evaluate every N training steps during phase 2
    seed=int(os.environ.get("SLEEP_SEED", "0")),   # changes the invented facts and the training randomness
)
device = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join("results", f"{MODEL.split('/')[-1]}-{METHOD}" + ("-quick" if QUICK else "")
                   + (f"-seed{CFG['seed']}" if CFG["seed"] else ""))
os.makedirs(OUT, exist_ok=True)
print("device:", device, "| quick:", QUICK, "|", CFG)

# %% [markdown]
# ## 1. Make two sets of invented facts
# Three fact templates. Every subject and answer is a made-up word, unique across both sets.

# %%
SYL = ["ka", "lo", "mi", "ra", "ven", "dor", "zu", "tel", "bri", "nox", "sha", "quil", "pa",
       "re", "ton", "vi", "sel", "mar", "gu", "fen", "ol", "dra", "ki", "um", "bas", "the", "yor"]
TEMPLATES = [
    ("The capital of {s} is", "capital"),
    ("The inventor of the {s} was", "inventor"),
    ("The river {s} flows into the", "river"),
]
STEMS = ["Fact: The capital of", "Fact: The inventor of the", "Fact: The river"]


def make_facts(n, rng, used):
    def word():
        while True:
            w = "".join(rng.choice(SYL) for _ in range(rng.choice([2, 3]))).capitalize()
            if w not in used and len(w) >= 4:
                used.add(w)
                return w

    facts = []
    for i in range(n):
        tmpl, kind = TEMPLATES[i % len(TEMPLATES)]
        prompt = "Fact: " + tmpl.format(s=word())
        facts.append({"prompt": prompt, "answer": word(), "kind": kind})
    return facts


rng = random.Random(CFG["seed"])
used = set()
facts_a = make_facts(CFG["n_facts"], rng, used)
facts_b = make_facts(CFG["n_facts"], rng, used)


def line(f):
    return f"{f['prompt']} {f['answer']}."


print("Set A examples:", *[line(f) for f in facts_a[:3]], sep="\n  ")
print("Set B examples:", *[line(f) for f in facts_b[:3]], sep="\n  ")

# A few ordinary sentences to check the model's general English (pre-training knowledge).
GENERAL = [
    "The sun rises in the east and sets in the west.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "Paris is the capital of France, and Berlin is the capital of Germany.",
    "Most plants need sunlight, water and carbon dioxide to grow.",
    "The heart pumps blood through the body.",
    "A week has seven days, and a year has twelve months.",
    "Shakespeare wrote plays such as Hamlet and Macbeth.",
    "The Pacific is the largest ocean on Earth.",
]

# %% [markdown]
# ## 2. Model, training and evaluation helpers

# %%
torch.manual_seed(CFG["seed"])
tok = AutoTokenizer.from_pretrained(CFG["model"])
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
base = AutoModelForCausalLM.from_pretrained(CFG["model"], torch_dtype=torch.float32).to(device)
if METHOD == "lora":
    from peft import LoraConfig, get_peft_model
    base = get_peft_model(base, LoraConfig(r=CFG["lora_rank"], lora_alpha=2 * CFG["lora_rank"],
                                           lora_dropout=0.0, target_modules="all-linear",
                                           task_type="CAUSAL_LM"))
    base.print_trainable_parameters()
print(f"{sum(p.numel() for p in base.parameters()) / 1e6:.0f}M parameters")


def save_trainable(model):
    """Copy of the weights that training changes (all of them for full, adapters for LoRA)."""
    return {k: v.detach().to("cpu", copy=True) for k, v in model.named_parameters() if v.requires_grad}


def load_trainable(model, state):
    model.load_state_dict(state, strict=False)


@torch.no_grad()
def recall(model, facts, bs=32):
    """Fraction of facts whose answer the model produces exactly (greedy decoding)."""
    model.eval()
    hits = 0
    for i in range(0, len(facts), bs):
        chunk = facts[i:i + bs]
        enc = tok([f["prompt"] for f in chunk], return_tensors="pt", padding=True).to(device)
        out = model.generate(**enc, max_new_tokens=6, do_sample=False, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for f, t in zip(chunk, texts):
            m = re.match(r"\s*([A-Za-z]+)", t)
            hits += bool(m and m.group(1) == f["answer"])
    return hits / len(facts)


@torch.no_grad()
def perplexity(model, sentences):
    model.eval()
    nll, n = 0.0, 0
    for s in sentences:
        ids = tok(s, return_tensors="pt").input_ids.to(device)
        loss = model(ids, labels=ids).loss
        nll += loss.item() * (ids.shape[1] - 1)
        n += ids.shape[1] - 1
    return math.exp(nll / n)


def train(model, texts, epochs=None, steps=None, log=None, seed=0, refresh=None, refresh_every=None):
    """Plain fine-tuning on a list of text lines.

    Stops after `epochs` passes, or after exactly `steps` optimizer steps (so conditions with more
    data get the same amount of training). Calls log(step) every eval_every steps. If `refresh` is
    given, every `refresh_every` steps it is called as refresh(step) and returns the new text list
    (used for sleep cycles: fresh dreams each "night")."""
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=CFG["lr"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / CFG["warmup"]))
    r = random.Random(seed + 1000 * CFG["seed"])
    step = 0
    epoch = 0
    refreshed_at = -1
    if log:
        log(step)
    while (steps is None and epoch < epochs) or (steps is not None and step < steps):
        epoch += 1
        order = texts[:]
        r.shuffle(order)
        for i in range(0, len(order), CFG["batch"]):
            if steps is not None and step >= steps:
                break
            if refresh and step and step % refresh_every == 0 and step != refreshed_at:
                refreshed_at = step
                texts = refresh(step)
                break   # start a new pass over the refreshed data
            model.train()
            batch = [t + tok.eos_token for t in order[i:i + CFG["batch"]]]
            tok.padding_side = "right"   # right-pad for training so token positions start at 0
            enc = tok(batch, return_tensors="pt", padding=True).to(device)
            tok.padding_side = "left"    # left-pad for generation
            labels = enc.input_ids.masked_fill(enc.attention_mask == 0, -100)
            loss = model(**enc, labels=labels).loss
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if log and step % CFG["eval_every"] == 0:
                log(step)
    if log and step % CFG["eval_every"]:
        log(step)
    return model


def snapshot(model, label):
    s = {"label": label, "A": recall(model, facts_a), "B": recall(model, facts_b),
         "ppl": perplexity(model, GENERAL)}
    print(f"  {label:<34} recall A {s['A']:.0%}   recall B {s['B']:.0%}   general-English ppl {s['ppl']:.1f}")
    return s


# %% [markdown]
# ## 3. Phase 1: learn yesterday's memories (set A)

# %%
t0 = time.time()
summary = [snapshot(base, f"pre-trained {MODEL.split('/')[-1]}")]
ckpt_path = os.path.join(OUT, "checkpoint_A.pt")
ckpt_key = {k: CFG[k] for k in ("model", "method", "n_facts", "epochs_a", "lr", "warmup", "batch", "seed")}
if os.environ.get("SLEEP_CACHE") == "1" and os.path.exists(ckpt_path) and torch.load(ckpt_path)["key"] == ckpt_key:
    load_trainable(base, torch.load(ckpt_path)["state"])   # reuse phase 1 from an earlier run
    print("loaded phase-1 checkpoint from", ckpt_path)
else:
    train(base, [line(f) for f in facts_a], CFG["epochs_a"], seed=1)
summary.append(snapshot(base, "after learning A"))
checkpoint = save_trainable(base)   # every condition below starts from here
if os.environ.get("SLEEP_CACHE") == "1":
    torch.save({"key": ckpt_key, "state": checkpoint}, ckpt_path)
print(f"phase 1 took {time.time() - t0:.0f}s")

# %% [markdown]
# ## 4. Dreams: let the model generate facts from memory
# We give it only the *start* of each fact type ("Fact: The capital of") and let it sample.
# No fact from set A is stored or shown. We keep well-formed sentences and check how many are
# faithful copies of a real memory.

# %%
@torch.no_grad()
def dream(model, n, seed=0):
    model.eval()
    torch.manual_seed(seed)
    dreams = []
    tries = 0
    while len(dreams) < n and tries < 20:
        tries += 1
        stems = [STEMS[i % len(STEMS)] for i in range(n)]
        enc = tok(stems, return_tensors="pt", padding=True).to(device)
        out = model.generate(**enc, max_new_tokens=16, do_sample=True, temperature=1.0, top_p=0.95,
                             pad_token_id=tok.pad_token_id)
        for text in tok.batch_decode(out, skip_special_tokens=True):
            first = text.split(".")[0].strip()
            if first.startswith("Fact: The") and len(first.split()) >= 5:
                dreams.append(first + ".")
    return dreams[:n]


VALID = re.compile(r"^Fact: The (capital of [A-Z][a-z]+ is|inventor of the [A-Z][a-z]+ was|"
                   r"river [A-Z][a-z]+ flows into the) [A-Z][a-z]+\.$")


@torch.no_grad()
def sentence_loss(model, s):
    ids = tok(s, return_tensors="pt").input_ids.to(device)
    return model(ids, labels=ids).loss.item()


def targeted_dream(model, n, seed=0):
    """Targeted dreaming: dream 4x as many, keep unique well-formed ones, and rehearse the n the model
    is LEAST sure of (highest loss). Like targeted memory reactivation, which boosts weak memories."""
    pool = [d for d in dict.fromkeys(dream(model, 4 * n, seed)) if VALID.match(d)]
    return sorted(pool, key=lambda d: -sentence_loss(model, d))[:n]


real_lines = {line(f) for f in facts_a}


def faithful_frac(ds):
    return sum(d in real_lines for d in ds) / max(1, len(ds))


n_replay = max(1, round(CFG["replay_frac"] * len(facts_a)))
dreams = dream(base, n_replay, seed=2)
faithful = sum(d in real_lines for d in dreams)
print(f"random dreams: {len(dreams)}, {faithful_frac(dreams):.0%} exact real memories, {len(set(dreams))} unique")
for d in dreams[:6]:
    print("  ", d, "  <- real memory" if d in real_lines else "")
t_dreams = targeted_dream(base, n_replay, seed=2)
print(f"targeted dreams: {len(t_dreams)}, {faithful_frac(t_dreams):.0%} exact real memories, {len(set(t_dreams))} unique")

# %% [markdown]
# ## 5. Phase 2: learn new facts (set B), five ways
# Every condition gets exactly the same number of training steps, so extra replay data doesn't
# mean extra training.
#
# - **Sleep cycles**: instead of dreaming once, the model "sleeps" every 20 steps: it dreams a fresh
#   set from its *current* memory and rehearses those until the next night.

# %%
B_lines = [line(f) for f in facts_b]
n_steps = CFG["epochs_b"] * math.ceil(len(facts_b) / CFG["batch"])
replay_lines = [line(f) for f in random.Random(3).sample(facts_a, n_replay)]
CYCLE = 20
cycle_log = []


def new_night(step):
    fresh = dream(base, n_replay, seed=100 + step)
    cycle_log.append((step, faithful_frac(fresh)))
    print(f"    night at step {step}: {faithful_frac(fresh):.0%} of fresh dreams are real memories")
    return B_lines + fresh


# A "hippocampus": a frozen copy of the model saved before it learns set B. New learning never
# touches it, so it can produce real dreams every night (like the hippocampus replaying to the cortex).
hippocampus = copy.deepcopy(base).eval()
hippo_log = []


def hippo_night(step):
    fresh = dream(hippocampus, n_replay, seed=100 + step)
    hippo_log.append((step, faithful_frac(fresh)))
    print(f"    hippocampus night at step {step}: {faithful_frac(fresh):.0%} of fresh dreams are real memories")
    return B_lines + fresh


conditions = {
    "No sleep (B only)": (B_lines, None),
    "Replay (stored memories)": (B_lines + replay_lines, None),
    "Dream replay (no stored data)": (B_lines + dreams, None),
    "Targeted dreaming": (B_lines + t_dreams, None),
    "Sleep cycles (fresh dreams every 20 steps)": (B_lines + dreams, new_night),
    "Hippocampus sleep cycles": (B_lines + dreams, hippo_night),
}
# SLEEP_ONLY="Hippocampus" runs only matching conditions and merges the rest from the last results.json.
only = os.environ.get("SLEEP_ONLY")
old = {}
if only and os.path.exists(f"{OUT}/results.json"):
    prev = json.load(open(f"{OUT}/results.json"))
    old = {s["label"]: (s, prev["curves"].get(s["label"])) for s in prev["summary"] if s["label"] in prev["curves"]}
    cycle_log = prev.get("sleep_cycles", [])
curves = {}
print(f"each condition trains for {n_steps} steps")
for name, (texts, refresh) in conditions.items():
    if only and only not in name and name in old:
        summary.append(old[name][0])
        curves[name] = old[name][1]
        print(f"  {name:<34} (reused from the previous run)")
        continue
    t0 = time.time()
    load_trainable(base, checkpoint)
    hist = []
    train(base, texts, steps=n_steps, seed=4, refresh=refresh, refresh_every=CYCLE,
          log=lambda step: hist.append((step, recall(base, facts_a), recall(base, facts_b))))
    curves[name] = hist
    summary.append(snapshot(base, name))
    print(f"    ({time.time() - t0:.0f}s)")

# %% [markdown]
# ## 6. Results

# %%
colors = {"No sleep (B only)": "#d1495b", "Replay (stored memories)": "#2e86ab",
          "Dream replay (no stored data)": "#8f5bd6", "Targeted dreaming": "#e08a1e",
          "Sleep cycles (fresh dreams every 20 steps)": "#2a9d6f", "Hippocampus sleep cycles": "#1f3a5f"}
fig, ax = plt.subplots(figsize=(10, 5.5))
for name, hist in curves.items():
    steps = [h[0] for h in hist]
    ax.plot(steps, [h[1] * 100 for h in hist], color=colors[name], lw=2.5, label=f"{name}: old facts (A)")
    ax.plot(steps, [h[2] * 100 for h in hist], color=colors[name], lw=1.5, ls="--", alpha=0.7,
            label=f"{name}: new facts (B)")
ax.set_xlabel("training steps while learning new facts")
ax.set_ylabel("recall (%)")
ax.set_ylim(-3, 103)
ax.set_title(f"{MODEL.split('/')[-1]} ({METHOD}): learning new facts vs. remembering old ones")
ax.legend(fontsize=8, loc="center right")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{OUT}/forgetting_curves.png", dpi=160)
plt.show()

final = summary[-len(conditions):]
fig, ax = plt.subplots(figsize=(10, 4.8))
x = range(len(final))
ax.bar([i - 0.2 for i in x], [s["A"] * 100 for s in final], width=0.4, label="old facts (A)", color="#2e86ab")
ax.bar([i + 0.2 for i in x], [s["B"] * 100 for s in final], width=0.4, label="new facts (B)", color="#f4a259")
ax.set_xticks(list(x))
ax.set_xticklabels([s["label"].replace(" (", "\n(") for s in final], fontsize=8.5)
ax.set_ylabel("recall after learning B (%)")
ax.set_ylim(0, 105)
ax.legend()
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(f"{OUT}/final_recall.png", dpi=160)
plt.show()

print("\n| stage | recall A (old) | recall B (new) | general-English perplexity |")
print("|---|---|---|---|")
for s in summary:
    print(f"| {s['label']} | {s['A']:.0%} | {s['B']:.0%} | {s['ppl']:.1f} |")
with open(f"{OUT}/results.json", "w") as fh:
    json.dump({"config": CFG, "summary": summary, "curves": curves,
               "dreams": dreams, "dream_faithful": faithful, "targeted_dreams": t_dreams,
               "targeted_faithful": faithful_frac(t_dreams), "sleep_cycles": cycle_log, "hippocampus_cycles": hippo_log}, fh, indent=1)

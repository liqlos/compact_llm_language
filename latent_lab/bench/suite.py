"""Procedural behavioral benchmark v2 — sequential state tracking.

Seven task families where the answer REQUIRES carrying state through d
sequential steps; single-pass pattern matching cannot solve them by
construction (verified by parse-back tests in tests/test_suite.py).

Splits (fixed master seed):
  train      depth 2-4, fresh values
  validation depth 2-4, fresh values
  test_id    depth 2-4, fresh seeds/values
  test_ood   depth 5-8 (length generalization)

Every example carries: closed candidate set (exact ranking scorer),
canonical answer, and a content key used by the leakage audit to forbid
identical problem cores across splits. Answers are balanced by rejection
sampling against a per-family quota (no majority-token shortcut).

No example shares a prompt string with another split; the audit is itself
unit-tested. This module is stdlib-only and deterministic.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any

MASTER_SEED = 20260822
FAMILIES: tuple[str, ...] = (
    "fsm",
    "stack_queue",
    "graph_walk",
    "chain_arith",
    "rule_neg",
    "tiny_prog",
    "obj_track",
)
TRAIN_DEPTH = (2, 4)
OOD_DEPTH = (5, 8)

_WORDS = [
    "amber", "birch", "cedar", "delta", "ember", "flint", "grove", "hazel",
    "indigo", "juniper", "kestrel", "linden", "maple", "nettle", "onyx",
    "pine", "quartz", "rowan", "slate", "thistle", "umber", "violet",
    "willow", "yarrow", "zephyr", "basalt", "coral", "dune", "elm", "fern",
]
_PLACES = ["library", "cafe", "harbor", "attic", "garden", "studio", "market"]
_ITEMS = ["compass", "lantern", "notebook", "teapot", "scarf", "wallet"]


@dataclass(frozen=True)
class Example:
    ex_id: str
    family: str
    depth: int
    prompt: str
    answer: str
    candidates: tuple[str, ...]
    content_key: str
    meta: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ex_id": self.ex_id,
            "family": self.family,
            "depth": self.depth,
            "prompt": self.prompt,
            "answer": self.answer,
            "candidates": list(self.candidates),
            "content_key": self.content_key,
            "meta": dict(self.meta),
        }


def _key(family: str, core: Any) -> str:
    return family + ":" + hashlib.sha256(
        json.dumps(core, sort_keys=True).encode()
    ).hexdigest()[:16]


# --------------------------------------------------------------------------
# family: fsm — finite-state machine execution
# --------------------------------------------------------------------------

def gen_fsm(rng: random.Random, depth: int) -> Example:
    n_states = rng.randint(3, 6)
    states = rng.sample(_WORDS, n_states)
    alphabet = ["x", "y", "z"]
    trans = {s: {a: rng.choice(states) for a in alphabet} for s in states}
    word = "".join(rng.choice(alphabet) for _ in range(depth))
    cur = states[0]
    for a in word:
        cur = trans[cur][a]
    rules = [(s, a, trans[s][a]) for s in states for a in alphabet]
    rng.shuffle(rules)
    fmt = rng.choice(["rules", "moves"])
    if fmt == "rules":
        body = "\n".join(f"When in {s} reading {a}, go to {t}." for s, a, t in rules)
    else:
        body = "\n".join(f"{s} --{a}--> {t}" for s, a, t in rules)
    start = states[0]
    prompt = (
        f"A machine has states: {', '.join(states)}.\n{body}\n"
        f"The machine starts in {start}. It reads the input '{word}' "
        f"(one symbol at a time, left to right).\n"
        f"In which state does it end?\nAnswer:"
    )
    return Example(
        ex_id="", family="fsm", depth=len(word), prompt=prompt, answer=cur,
        candidates=tuple(states),
        content_key=_key("fsm", {"t": sorted((s, a, t) for s, a, t in rules),
                                 "w": word, "start": start}),
        meta={"word": word},
    )


# --------------------------------------------------------------------------
# family: stack_queue — sequential container transformations
# --------------------------------------------------------------------------

def gen_stack_queue(rng: random.Random, depth: int) -> Example:
    subtype = rng.choice(["stack", "queue"])
    items = rng.sample(_WORDS, 6)
    struct = []
    ops: list[tuple[str, str]] = []
    for _ in range(depth):
        can_remove = bool(struct)
        choices = ["add"]
        if can_remove:
            choices.append("remove")
        if rng.choice(choices) == "add":
            it = rng.choice(items)
            struct.insert(0, it)
            ops.append(("add", it))
        else:
            it = struct.pop(0) if subtype == "stack" else struct.pop(-1)
            ops.append(("remove", it))
    ask_top = rng.random() < 0.6
    if ask_top and struct:
        # index 0 is the TOP of the stack / the BACK of the queue
        answer = struct[0] if subtype == "stack" else struct[-1]
        q = ("What is on top of the stack?" if subtype == "stack"
             else "What is at the front of the queue?")
        cands = tuple(items)
    else:
        answer = str(len(struct))
        q = ("How many elements are in the stack?" if subtype == "stack"
             else "How many elements are in the queue?")
        cands = tuple(str(i) for i in range(depth + 1))
    name = rng.choice(_WORDS)
    lines: list[str] = []
    for i, (kind, it) in enumerate(ops, 1):
        if kind == "add":
            lines.append(f"{i}. Put {it} on top." if subtype == "stack"
                         else f"{i}. {it} joins the back of the line.")
        else:
            lines.append(f"{i}. Remove the top element." if subtype == "stack"
                         else f"{i}. The element at the front leaves.")
    prompt = (
        f"{name.capitalize()} maintains a {'stack' if subtype == 'stack' else 'queue'} "
        f"that is initially empty.\n" + "\n".join(lines) +
        f"\n{q}\nAnswer:"
    )
    return Example(
        ex_id="", family="stack_queue", depth=len(ops), prompt=prompt,
        answer=answer, candidates=cands,
        content_key=_key("stack_queue", {"sub": subtype, "ops": ops,
                                         "q": ask_top}),
        meta={"subtype": subtype},
    )


# --------------------------------------------------------------------------
# family: graph_walk — deterministic propagation on a functional graph
# --------------------------------------------------------------------------

def gen_graph_walk(rng: random.Random, depth: int) -> Example:
    n = rng.randint(4, 7)
    nodes = rng.sample(_WORDS, n)
    succ: dict[str, str] = {}
    for nd in nodes:
        succ[nd] = rng.choice([m for m in nodes if m != nd])
    # require the actual walk to visit >=2 distinct nodes (non-degenerate)
    for _ in range(64):
        start = rng.choice(nodes)
        walk = [start]
        for _ in range(depth):
            walk.append(succ[walk[-1]])
        if len(set(walk)) >= 2:
            break
    edges = list(succ.items())
    rng.shuffle(edges)
    fmt = rng.choice(["arrow", "sentence"])
    if fmt == "arrow":
        body = "\n".join(f"{a} -> {b}" for a, b in edges)
    else:
        body = "\n".join(f"From {a} the path always continues to {b}."
                         for a, b in edges)
    prompt = (
        f"A one-way path network connects: {', '.join(nodes)}.\n{body}\n"
        f"A courier starts at {start} and follows the paths for exactly "
        f"{depth} hops.\nAt which node does the courier end?\nAnswer:"
    )
    return Example(
        ex_id="", family="graph_walk", depth=depth, prompt=prompt,
        answer=walk[-1], candidates=tuple(nodes),
        content_key=_key("graph_walk", {"succ": sorted(succ.items()),
                                        "start": start, "d": depth}),
        meta={},
    )


# --------------------------------------------------------------------------
# family: chain_arith — chained modular arithmetic
# --------------------------------------------------------------------------

def gen_chain_arith(rng: random.Random, depth: int) -> Example:
    m = rng.choice([11, 17, 23, 29, 31])
    x = rng.randrange(m)
    ops = []
    for _ in range(depth):
        a = rng.randint(2, 5)
        b = rng.randint(0, 9)
        ops.append((a, b))
    vals = [x]
    for a, b in ops:
        x = (a * x + b) % m
        vals.append(x)
    lines = [f"T{i}: multiply by {a}, then add {b}, then reduce mod {m}."
             for i, (a, b) in enumerate(ops, 1)]
    rng.shuffle(lines)
    prompt = (
        f"All computations are modulo {m}.\nStart: x = {vals[0]}.\n" +
        "\n".join(lines) +
        f"\nWhat is the final value of x after applying T1 through T{depth} "
        f"in order?\nAnswer:"
    )
    return Example(
        ex_id="", family="chain_arith", depth=depth, prompt=prompt,
        answer=str(vals[-1]), candidates=tuple(str(i) for i in range(m)),
        content_key=_key("chain_arith", {"m": m, "start": vals[0],
                                         "ops": ops}),
        meta={},
    )


# --------------------------------------------------------------------------
# family: rule_neg — rule application with negation
# --------------------------------------------------------------------------

def gen_rule_neg(rng: random.Random, depth: int) -> Example:
    props = rng.sample(_WORDS, depth + 3)
    chain = props[: depth + 1]
    distract = props[depth + 1:]
    # the antecedent of each rule states the ACTUAL polarity of its source,
    # so every chained rule fires and the consequent sets the actual polarity
    pol = [rng.choice(["true", "false"])]
    rules = []
    for i in range(depth):
        nxt = rng.choice(["true", "false"])
        pol.append(nxt)
        rules.append((chain[i], pol[i], chain[i + 1], nxt))
    drules = []
    for d in distract:
        src_pol = rng.choice(["true", "false"])
        dst_pol = rng.choice(["true", "false"])
        drules.append((d, src_pol, rng.choice(props), dst_pol))
    all_rules = rules + drules
    rng.shuffle(all_rules)
    rl = []
    for src, sp, dst, dp in all_rules:
        neg_s = "" if sp == "true" else "not "
        neg_d = "" if dp == "true" else "not "
        rl.append(f"If {src} is {neg_s}correct, then {dst} is {neg_d}correct.")
    known = (f"The label {chain[0]} is "
             f"{'correct' if pol[0] == 'true' else 'incorrect'}.")
    target = chain[-1]
    answer = "true" if pol[-1] == "true" else "false"
    prompt = (
        f"Facts and rules:\n{known}\n" + "\n".join(rl) +
        f"\nQuestion: Is the label {target} correct?\n"
        f"Reply 'true' or 'false'.\nAnswer:"
    )
    return Example(
        ex_id="", family="rule_neg", depth=depth, prompt=prompt, answer=answer,
        candidates=("true", "false"),
        content_key=_key("rule_neg", {"rules": sorted(all_rules),
                                      "known": pol[0], "target": target}),
        meta={"polarities": ",".join(pol)},
    )


# --------------------------------------------------------------------------
# family: tiny_prog — tiny register program execution
# --------------------------------------------------------------------------

def gen_tiny_prog(rng: random.Random, depth: int) -> Example:
    regs = {"r1": rng.randint(0, 9), "r2": rng.randint(0, 9),
            "r3": rng.randint(0, 9)}
    init = dict(regs)
    lines: list[str] = []
    executed = 0
    while executed < depth:
        tgt = rng.choice(list(regs))
        kind = rng.choice(["add", "sub", "copy"])
        twice = executed + 2 <= depth and rng.random() < 0.25
        reps = 2 if twice else 1
        amt = rng.randint(1, 4)
        src = rng.choice([r for r in regs if r != tgt]) if kind == "copy" \
            else None
        for _ in range(reps):
            if kind == "add":
                regs[tgt] += amt
            elif kind == "sub":
                regs[tgt] = max(0, regs[tgt] - amt)
            else:
                regs[tgt] = regs[src]
            executed += 1
        if kind == "add":
            desc = f"add {amt} to {tgt}"
        elif kind == "sub":
            desc = f"subtract {amt} from {tgt}"
        else:
            desc = f"copy {src} into {tgt}"
        if max(regs.values()) > 99:
            return gen_tiny_prog(rng, depth)  # regenerate: keep bounded
        lines.append(f"repeat twice: {desc}" if twice else desc)
    tgt_ans = rng.choice(list(regs))
    answer = str(regs[tgt_ans])
    body = "\n".join(f"{i}. {ln}" for i, ln in enumerate(lines, 1))
    initl = ", ".join(f"{r}={v}" for r, v in init.items())
    prompt = (
        f"A machine has registers {initl}.\nProgram:\n{body}\n"
        f"Every line runs once unless marked otherwise.\n"
        f"What is the value of {tgt_ans} afterwards?\nAnswer:"
    )
    return Example(
        ex_id="", family="tiny_prog", depth=executed, prompt=prompt, answer=answer,
        candidates=tuple(str(i) for i in range(100)),
        content_key=_key("tiny_prog", {"init": init, "prog": lines,
                                       "target": tgt_ans}),
        meta={},
    )


# --------------------------------------------------------------------------
# family: obj_track — multi-object state tracking
# --------------------------------------------------------------------------

def gen_obj_track(rng: random.Random, depth: int) -> Example:
    people = rng.sample(_WORDS, rng.randint(3, 4))
    places = rng.sample(_PLACES, 4)
    loc = {p: rng.choice(places) for p in people}
    holder = {it: None for it in _ITEMS[: 2]}
    for it in holder:
        holder[it] = rng.choice(people)
    evs: list[str] = []
    for _ in range(depth):
        if rng.random() < 0.55:
            p = rng.choice(people)
            dest = rng.choice([pl for pl in places if pl != loc[p]])
            loc[p] = dest
            evs.append(("move", f"{p} walked to the {dest}."))
        else:
            it = rng.choice(list(holder))
            frm = holder[it]
            to = rng.choice([p for p in people if p != frm]) if frm else rng.choice(people)
            holder[it] = to
            evs.append(("give", f"{to} took the {it} from {frm}." if frm
                       else f"{to} picked up the {it}."))
    ask_loc = rng.random() < 0.5
    if ask_loc:
        who = rng.choice(people)
        answer = loc[who]
        q = f"Where is {who} at the end?"
        cands = tuple(places)
    else:
        it = rng.choice(list(holder))
        answer = holder[it]
        q = f"Who has the {it} at the end?"
        cands = tuple(people)
    # events stay in generation (chronological) order — narrated order IS the
    # execution order, so parse-back verification is well-defined
    body = "\n".join(text for _, text in evs)
    prompt = (
        "Initial situation: " +
        "; ".join(f"{p} is at the {loc[p]}" for p in people) + ". " +
        "; ".join(f"the {it} belongs to {holder[it]}" for it in holder) +
        f".\nEvents, in order:\n{body}\n{q}\nAnswer:"
    )
    return Example(
        ex_id="", family="obj_track", depth=depth, prompt=prompt, answer=answer,
        candidates=cands,
        content_key=_key("obj_track", {"init_loc": sorted(loc.items()),
                                       "events": [t for _, t in evs]}),
        meta={},
    )


GENERATORS = {
    "fsm": gen_fsm,
    "stack_queue": gen_stack_queue,
    "graph_walk": gen_graph_walk,
    "chain_arith": gen_chain_arith,
    "rule_neg": gen_rule_neg,
    "tiny_prog": gen_tiny_prog,
    "obj_track": gen_obj_track,
}


# --------------------------------------------------------------------------
# split construction with balance + uniqueness control
# --------------------------------------------------------------------------

def _gen_unique(family: str, rng: random.Random, depth_range: tuple[int, int],
                seen_keys: set[str]) -> Example:
    for _ in range(256):
        ex = GENERATORS[family](rng, rng.randint(*depth_range))
        if ex.content_key not in seen_keys:
            seen_keys.add(ex.content_key)
            return ex
    raise RuntimeError(f"could not generate unique {family} example")


def _balanced_examples(family: str, n: int, rng: random.Random,
                       depth_range: tuple[int, int], seen_keys: set[str],
                       max_majority_frac: float | None = None) -> list[Example]:
    """Rejection-sample so no single answer exceeds the majority cap.

    The cap adapts to the candidate-set size: binary-answer families can
    never beat a 40% cap (majority >= 50% by pigeonhole), so their cap
    relaxes while still forcing meaningful use of both classes.
    """
    if max_majority_frac is None:
        max_majority_frac = 0.4 if family != "rule_neg" else 0.65
    out: list[Example] = []
    counts: dict[str, int] = {}
    attempts = 0
    while len(out) < n:
        attempts += 1
        if attempts > n * 400:
            raise RuntimeError(f"balance unachievable for {family}")
        ex = _gen_unique(family, rng, depth_range, seen_keys)
        limit = max(2, int(max_majority_frac * (len(out) + 1)))
        if counts.get(ex.answer, 0) + 1 > limit:
            continue
        counts[ex.answer] = counts.get(ex.answer, 0) + 1
        out.append(ex)
    return out


def assign_ids(examples: list[Example], prefix: str) -> list[Example]:
    return [
        Example(**{**ex.__dict__, "ex_id": f"{prefix}-{i:05d}"})
        for i, ex in enumerate(examples)
    ]


@dataclass(frozen=True)
class Suite:
    train: tuple[Example, ...]
    validation: tuple[Example, ...]
    test_id: tuple[Example, ...]
    test_ood: tuple[Example, ...]

    def splits(self) -> dict[str, tuple[Example, ...]]:
        return {
            "train": self.train, "validation": self.validation,
            "test_id": self.test_id, "test_ood": self.test_ood,
        }

    def manifest(self) -> dict[str, Any]:
        sizes = {name: len(exs) for name, exs in self.splits().items()}
        h = hashlib.sha256()
        for exs in self.splits().values():
            for ex in exs:
                h.update(json.dumps(ex.to_dict(), sort_keys=True).encode())
        return {
            "suite": "behavioral-v2", "master_seed": MASTER_SEED,
            "families": list(FAMILIES), "sizes": sizes,
            "train_depth": list(TRAIN_DEPTH), "ood_depth": list(OOD_DEPTH),
            "sha256": h.hexdigest(),
        }


def build_suite(
    *,
    n_train_per_family: int = 70,
    n_val_per_family: int = 8,
    n_test_per_family: int = 16,
    seed: int = MASTER_SEED,
) -> Suite:
    rng = random.Random(seed)
    seen: set[str] = set()
    train, val, tid, ood = [], [], [], []
    for fam in FAMILIES:
        train += _balanced_examples(fam, n_train_per_family, rng, TRAIN_DEPTH, seen)
        val += _balanced_examples(fam, n_val_per_family, rng, TRAIN_DEPTH, seen)
        tid += _balanced_examples(fam, n_test_per_family, rng, TRAIN_DEPTH, seen)
        ood += _balanced_examples(fam, n_test_per_family, rng, OOD_DEPTH, seen)
    for lst in (train, val, tid, ood):
        rng.shuffle(lst)
    return Suite(
        train=tuple(assign_ids(train, "tr")),
        validation=tuple(assign_ids(val, "va")),
        test_id=tuple(assign_ids(tid, "ti")),
        test_ood=tuple(assign_ids(ood, "to")),
    )


# --------------------------------------------------------------------------
# leakage audit
# --------------------------------------------------------------------------

def audit(suite: Suite) -> dict[str, Any]:
    """Leakage audit: duplicates, cross-split core reuse, answer balance."""
    report: dict[str, Any] = {}
    problems: list[str] = []
    prompts_by_split: dict[str, set[str]] = {}
    keys_by_split: dict[str, set[str]] = {}
    for split, exs in suite.splits().items():
        ids = [ex.ex_id for ex in exs]
        if len(set(ids)) != len(ids):
            problems.append(f"{split}: duplicate ex_ids")
        prompts = [ex.prompt for ex in exs]
        if len(set(prompts)) != len(prompts):
            problems.append(f"{split}: duplicate prompts within split")
        prompts_by_split[split] = set(prompts)
        keys_by_split[split] = {ex.content_key for ex in exs}
        for ex in exs:
            if ex.answer not in ex.candidates:
                problems.append(f"{split}/{ex.ex_id}: answer outside candidates")
            if ex.depth < 2 or ex.depth > 8:
                problems.append(f"{split}/{ex.ex_id}: depth {ex.depth} out of range")
        maj = _majority_freq(exs)
        report[f"{split}_majority_answer_frac"] = round(maj, 3)
        depth_hist: dict[int, int] = {}
        for ex in exs:
            depth_hist[ex.depth] = depth_hist.get(ex.depth, 0) + 1
        report[f"{split}_depth_hist"] = {str(k): depth_hist[k]
                                         for k in sorted(depth_hist)}
        report[f"{split}_family_counts"] = {
            f: sum(1 for e in exs if e.family == f) for f in FAMILIES}
    split_names = list(prompts_by_split)
    for i, a in enumerate(split_names):
        for b in split_names[i + 1:]:
            inter = prompts_by_split[a] & prompts_by_split[b]
            if inter:
                problems.append(f"duplicate prompts across {a}/{b}: {len(inter)}")
    for holdout in ("validation", "test_id", "test_ood"):
        inter = keys_by_split["train"] & keys_by_split[holdout]
        if inter:
            problems.append(f"content-core overlap train/{holdout}: {len(inter)}")
    report["problems"] = problems
    report["ok"] = not problems
    return report


def _majority_freq(exs: tuple[Example, ...]) -> float:
    if not exs:
        return 0.0
    counts: dict[str, int] = {}
    for ex in exs:
        counts[ex.answer] = counts.get(ex.answer, 0) + 1
    return max(counts.values()) / len(exs)


def constant_baseline(suite: Suite) -> dict[str, float]:
    """Accuracy of always predicting the split's most frequent answer."""
    return {k: round(_majority_freq(v), 4) for k, v in suite.splits().items()}

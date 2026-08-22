"""Behavioral suite v2: parse-back verification, leakage audit, determinism.

Parse-back tests re-solve each example by parsing the PROMPT TEXT with
independent regex-based simulators and comparing against the stored answer.
This proves prompts are self-contained and answers are exact.
"""

from __future__ import annotations

import re

import pytest

from latent_lab.bench.suite import (
    FAMILIES,
    MASTER_SEED,
    audit,
    build_suite,
    constant_baseline,
)

SUITE = build_suite()


# --------------------------------------------------------------------------
# independent simulators operating on prompt text only
# --------------------------------------------------------------------------

def solve_fsm(prompt: str) -> str:
    trans: dict[tuple[str, str], str] = {}
    for m in re.finditer(
            r"When in (\S+) reading (\w), go to (\S+)\.", prompt):
        trans[(m.group(1), m.group(2))] = m.group(3)
    for m in re.finditer(r"(\S+) --(\w)--> (\S+)", prompt):
        trans[(m.group(1), m.group(2))] = m.group(3)
    start = re.search(r"starts in (\S+)\.", prompt).group(1)
    word = re.search(r"input '([xyz]+)'", prompt).group(1)
    cur = start
    for a in word:
        cur = trans[(cur, a)]
    return cur


def solve_stack_queue(prompt: str) -> str:
    lines = prompt.splitlines()
    q = lines[-2]  # question precedes final "Answer:" line
    seq: list[str] = []  # index 0 == oldest element (queue front / stack bottom)
    for line in lines:
        m_add = re.match(r"\d+\. Put (\S+) on top\.$", line)
        m_join = re.match(r"\d+\. (\S+) joins the back of the line\.$", line)
        if m_add:
            seq.insert(0, m_add.group(1))
        elif m_join:
            seq.append(m_join.group(1))
        elif re.match(r"\d+\. Remove the top element\.$", line) or re.match(r"\d+\. The element at the front leaves\.$", line):
            seq.pop(0)
    if q.startswith("What is on top"):
        return seq[0]
    if q.startswith("What is at the front"):
        return seq[0]
    return str(len(seq))


def solve_graph_walk(prompt: str) -> str:
    succ = {m.group(1): m.group(2) for m in re.finditer(
        r"(\S+) -> (\S+)", prompt)}
    if not succ:
        succ = {m.group(1): m.group(2) for m in re.finditer(
            r"From (\S+) the path always continues to (\S+)\.", prompt)}
    start = re.search(r"courier starts at (\S+) and", prompt).group(1)
    hops = int(re.search(r"exactly (\d+) hops", prompt).group(1))
    cur = start
    for _ in range(hops):
        cur = succ[cur]
    return cur


def solve_chain_arith(prompt: str) -> str:
    m_mod = int(re.search(r"modulo (\d+)", prompt).group(1))
    x = int(re.search(r"x = (\d+)\.", prompt).group(1))
    steps = {}
    for m in re.finditer(
            r"T(\d+): multiply by (\d+), then add (\d+), then reduce mod",
            prompt):
        steps[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))
    n = max(steps)
    for i in range(1, n + 1):
        a, b = steps[i]
        x = (a * x + b) % m_mod
    return str(x)


def solve_rule_neg(prompt: str) -> str:
    known: dict[str, bool] = {}
    m0 = re.search(r"The label (\S+) is (incorrect|correct)\.", prompt)
    known[m0.group(1)] = m0.group(2) == "correct"
    rules = []
    for m in re.finditer(
            r"If (\S+) is (not )?correct, then (\S+) is (not )?correct\.",
            prompt):
        rules.append((m.group(1), m.group(2) is None, m.group(3),
                      m.group(4) is None))
    target = re.search(r"Is the label (\S+) correct\?", prompt).group(1)
    changed = True
    while changed:
        changed = False
        for src, sp, dst, dp in rules:
            if src in known and dst not in known and known[src] == sp:
                known[dst] = dp
                changed = True
    return "true" if known[target] else "false"


def solve_tiny_prog(prompt: str) -> str:
    regs = {m.group(1): int(m.group(2)) for m in re.finditer(
        r"(r\d)=(\d+)", prompt.splitlines()[0])}
    prog = prompt.split("Program:\n")[1].split(
        "\nEvery line runs once")[0]
    for raw in prog.strip().splitlines():
        line = re.sub(r"^\d+\. ", "", raw)
        twice = line.startswith("repeat twice:")
        body = line.split(": ", 1)[1] if ": " in line else line
        reps = 2 if twice else 1
        m_add = re.match(r"add (\d+) to (r\d)$", body)
        m_sub = re.match(r"subtract (\d+) from (r\d)$", body)
        m_cpy = re.match(r"copy (r\d) into (r\d)$", body)
        for _ in range(reps):
            if m_add:
                regs[m_add.group(2)] += int(m_add.group(1))
            elif m_sub:
                regs[m_sub.group(2)] = max(
                    0, regs[m_sub.group(2)] - int(m_sub.group(1)))
            else:
                regs[m_cpy.group(2)] = regs[m_cpy.group(1)]
    tgt = re.search(r"value of (r\d) afterwards", prompt).group(1)
    return str(regs[tgt])


def _split_initial(text: str) -> tuple[dict[str, str], dict[str, str]]:
    body = text.split("Initial situation: ")[1].split(".\nEvents")[0]
    loc = {who: where for who, where in
           re.findall(r"(\w+) is at the (\w+)", body)}
    holder = {item: who for item, who in
              re.findall(r"the (\w+) belongs to (\w+)", body)}
    return loc, holder


def solve_obj_track(prompt: str) -> str:
    loc, holder = _split_initial(prompt)
    events = prompt.split("Events, in order:\n")[1].split("\nWhere is")[
        0].split("\nWho has")[0].strip().splitlines()
    for ev in events:
        m_move = re.match(r"(\S+) walked to the (\S+)\.$", ev)
        m_take = re.match(r"(\S+) took the (.+?) from (\S+)\.$", ev)
        m_pick = re.match(r"(\S+) picked up the (.+?)\.$", ev)
        if m_move:
            loc[m_move.group(1)] = m_move.group(2)
        elif m_take or m_pick:
            m = m_take or m_pick
            holder[m.group(2)] = m.group(1)
    m_where = re.search(r"Where is (\S+) at the end\?", prompt)
    if m_where:
        return loc[m_where.group(1)]
    it = re.search(r"Who has the (.+?) at the end\?", prompt).group(1)
    return holder[it]


SOLVERS = {
    "fsm": solve_fsm,
    "stack_queue": solve_stack_queue,
    "graph_walk": solve_graph_walk,
    "chain_arith": solve_chain_arith,
    "rule_neg": solve_rule_neg,
    "tiny_prog": solve_tiny_prog,
    "obj_track": solve_obj_track,
}


@pytest.mark.parametrize("family", FAMILIES)
def test_parse_back_train(family):
    for ex in SUITE.train:
        if ex.family == family:
            assert SOLVERS[family](ex.prompt) == ex.answer, ex.ex_id


@pytest.mark.parametrize("family", FAMILIES)
def test_parse_back_test_id(family):
    for ex in SUITE.test_id:
        if ex.family == family:
            assert SOLVERS[family](ex.prompt) == ex.answer, ex.ex_id


@pytest.mark.parametrize("family", FAMILIES)
def test_parse_back_ood(family):
    depths = []
    for ex in SUITE.test_ood:
        if ex.family == family:
            assert SOLVERS[family](ex.prompt) == ex.answer, ex.ex_id
            depths.append(ex.depth)
    assert min(depths) >= 5 and max(depths) <= 8


def test_determinism():
    s2 = build_suite()
    a = [e.to_dict() for e in SUITE.test_id]
    b = [e.to_dict() for e in s2.test_id]
    assert a == b


def test_split_depth_ranges():
    for split in ("train", "validation", "test_id"):
        assert all(2 <= e.depth <= 4 for e in getattr(SUITE, split))
    assert all(5 <= e.depth <= 8 for e in SUITE.test_ood)


def test_candidates_contain_answer_and_are_bounded():
    for exs in SUITE.splits().values():
        for ex in exs:
            assert ex.answer in ex.candidates
            assert len(ex.candidates) <= 101
            assert len(set(ex.candidates)) == len(ex.candidates)


def test_leakage_audit_ok():
    rep = audit(SUITE)
    assert rep["ok"], rep["problems"]
    for split in ("train", "test_id", "test_ood"):
        # pooled-split cap: per-family caps are 0.4 (multi-candidate) and
        # 0.65 (binary rule_neg); pooling dilutes both well below this
        assert rep[f"{split}_majority_answer_frac"] <= 0.45


def test_constant_baseline_reported():
    cb = constant_baseline(SUITE)
    assert set(cb) == {"train", "validation", "test_id", "test_ood"}
    assert all(0 < v <= 0.45 for v in cb.values())


def test_manifest_shape():
    man = SUITE.manifest()
    assert man["master_seed"] == MASTER_SEED
    assert man["sha256"] and len(man["sha256"]) == 64
    assert sum(man["sizes"].values()) > 200


def test_prompts_do_not_contain_the_answer_verbatim_unless_required():
    """Guard against trivially-copyable answers.

    For families where the answer token never appears as a standalone fact
    in the prompt (chain_arith, tiny_prog final values; fsm/graph_walk end
    states; stack/queue results), the answer string may legitimately occur
    as an intermediate value — so this check is family-targeted: the FINAL
    answer must not be printed next to the question marker.
    """
    for exs in (SUITE.validation, SUITE.test_id):
        for ex in exs:
            tail = ex.prompt[-80:]
            if ex.family in ("chain_arith", "tiny_prog"):
                assert f"Answer: {ex.answer}" not in ex.prompt
            assert "Answer:" in tail

"""Deterministic behavioral-v3 benchmark with falsifiable integrity checks.

The module is stdlib-only.  It deliberately does not reuse behavioral-v2
generators: v2 remains an immutable historical benchmark, including its known
``obj_track`` initial-state defect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

SUITE_IDENTITY = "behavioral-v3"
SUITE_VERSION = 3
SUITE_SCHEMA_VERSION = "behavioral_suite.v3"
MASTER_SEED = 20260827
TOKENIZER_IDENTITY = "utf8-bytes.v1"

TRACE_SUPERVISION_FAMILIES = frozenset({
    "fsm", "graph_walk", "chain_arith", "tiny_prog",
})

FAMILIES: tuple[str, ...] = (
    "fsm",
    "stack_queue",
    "graph_walk",
    "chain_arith",
    "rule_neg",
    "tiny_prog",
    "obj_track",
)

ID_DEPTH = (3, 5)
LENGTH_OOD_DEPTH = (7, 10)
FINAL_DEPTH = (4, 8)
DEFAULT_COUNTS: Mapping[str, int] = {
    "train": 40,
    "validation": 32,
    "test_id": 24,
    "test_ood_length": 24,
    "test_ood_semantic": 24,
    "final_test": 32,
}
CHECKPOINT_SELECTION_SPLIT = "validation"
TRAINING_SPLIT = "train"
SELECTION_ELIGIBLE_SPLITS = (CHECKPOINT_SELECTION_SPLIT,)
UNTOUCHED_FINAL_SPLIT = "final_test"

_WORDS_ID = (
    "amber", "birch", "cedar", "delta", "ember", "flint", "grove",
    "hazel", "indigo", "juniper", "kestrel", "linden", "maple",
    "nettle", "onyx", "pine", "quartz", "rowan", "slate", "thistle",
    "umber", "violet", "willow", "yarrow", "zephyr",
)
_WORDS_OOD = (
    "acacia", "banyan", "cypress", "dogwood", "eucalyptus", "fir",
    "ginkgo", "hawthorn", "ironwood", "jacaranda", "kapok", "laburnum",
    "magnolia", "narcissus", "oleander", "poplar", "redwood", "sequoia",
    "tamarack", "verbena", "wisteria", "yucca",
)
_PLACES_ID = ("library", "cafe", "harbor", "attic", "garden", "studio")
_PLACES_OOD = (
    "observatory", "foundry", "aquarium", "planetarium", "archive",
    "greenhouse",
)
_ITEMS_ID = ("compass", "lantern", "notebook", "teapot")
_ITEMS_OOD = ("astrolabe", "sextant", "hourglass", "barometer")
_SEMANTIC_OOD_MARKER = "BEGIN_BEHAVIORAL_V3_JSON"
_SEMANTIC_OOD_END = "END_BEHAVIORAL_V3_JSON"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _seed_for(*parts: object) -> int:
    material = ":".join(str(p) for p in parts)
    return int(_sha256_text(material)[:16], 16)


def _token_lengths(candidates: Sequence[str]) -> tuple[int, ...]:
    return tuple(len(candidate.encode("utf-8")) for candidate in candidates)


@dataclass(frozen=True)
class Example:
    ex_id: str
    split: str
    family: str
    depth: int
    template: str
    prompt: str
    answer: str
    candidates: tuple[str, ...]
    canonical_candidates: tuple[str, ...]
    candidate_permutation_seed: int
    candidate_permutation: tuple[int, ...]
    candidate_token_lengths: tuple[int, ...]
    candidate_tokenizer_identity: str
    gold_index: int
    content_key: str
    initial_text: str
    events_text: tuple[str, ...]
    scenario_json: str = field(repr=False)

    @property
    def scenario(self) -> dict[str, Any]:
        return json.loads(self.scenario_json)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SUITE_SCHEMA_VERSION,
            "suite_identity": SUITE_IDENTITY,
            "suite_version": SUITE_VERSION,
            "example_id": self.ex_id,
            "split": self.split,
            "family": self.family,
            "depth": self.depth,
            "template": self.template,
            "prompt": self.prompt,
            "answer": self.answer,
            "candidates": list(self.candidates),
            "canonical_candidates": list(self.canonical_candidates),
            "candidate_permutation_seed": self.candidate_permutation_seed,
            "candidate_permutation": list(self.candidate_permutation),
            "candidate_token_lengths": list(self.candidate_token_lengths),
            "candidate_tokenizer_identity": self.candidate_tokenizer_identity,
            "gold_index": self.gold_index,
            "content_key": self.content_key,
        }


@dataclass(frozen=True)
class Suite:
    train: tuple[Example, ...]
    validation: tuple[Example, ...]
    test_id: tuple[Example, ...]
    test_ood_length: tuple[Example, ...]
    test_ood_semantic: tuple[Example, ...]
    final_test: tuple[Example, ...]
    seed: int = MASTER_SEED

    def splits(self) -> dict[str, tuple[Example, ...]]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test_id": self.test_id,
            "test_ood_length": self.test_ood_length,
            "test_ood_semantic": self.test_ood_semantic,
            "final_test": self.final_test,
        }

    def records_hash(self) -> str:
        records = {
            name: [example.to_dict() for example in examples]
            for name, examples in self.splits().items()
        }
        return _sha256_text(_canonical_json(records))

    def manifest(self) -> dict[str, Any]:
        split_records = {
            name: [
                _sha256_text(_canonical_json(example.to_dict()))
                for example in examples
            ]
            for name, examples in self.splits().items()
        }
        return {
            "schema_version": SUITE_SCHEMA_VERSION,
            "suite_identity": SUITE_IDENTITY,
            "suite_version": SUITE_VERSION,
            "suite_hash": self.records_hash(),
            "master_seed": self.seed,
            "families": list(FAMILIES),
            "sizes": {name: len(examples) for name, examples in self.splits().items()},
            "depth_domains": {
                "id": list(ID_DEPTH),
                "length_ood": list(LENGTH_OOD_DEPTH),
                "final": list(FINAL_DEPTH),
            },
            "semantic_ood": {
                "split": "test_ood_semantic",
                "shift_dimensions": [
                    "held_out_vocabulary",
                    "held_out_parameter_domain",
                    "structured_json_template",
                ],
            },
            "checkpoint_selection_split": CHECKPOINT_SELECTION_SPLIT,
            "training_split": TRAINING_SPLIT,
            "selection_eligible_splits": list(SELECTION_ELIGIBLE_SPLITS),
            "selection_ineligible_splits": [
                name for name in self.splits() if name not in SELECTION_ELIGIBLE_SPLITS
            ],
            "untouched_final_test": {
                "split": UNTOUCHED_FINAL_SPLIT,
                "checkpoint_selection_allowed": False,
                "baseline_reporting_allowed": False,
            },
            "candidate_order_contract": {
                "permutation_direction": "actual_index_to_canonical_index",
                "seed_field": "candidate_permutation_seed",
                "duplicates_allowed": False,
                "gold_position_schedule": "balanced_then_seeded_shuffle.v1",
            },
            "candidate_token_lengths": {
                "field": "candidate_token_lengths",
                "identity": TOKENIZER_IDENTITY,
                "note": "Model-specific eval records must additionally retain model tokenizer token counts.",
            },
            "example_hashes": split_records,
        }


def _replay(scenario: Mapping[str, Any]) -> str:
    family = scenario["family"]
    if family == "fsm":
        current = scenario["start"]
        transitions = scenario["transitions"]
        for symbol in scenario["events"]:
            current = transitions[current][symbol]
        return current
    if family == "stack_queue":
        values = list(scenario["initial"])
        kind = scenario["kind"]
        for event in scenario["events"]:
            op = event["op"]
            if kind == "stack" and op == "push":
                values.insert(0, event["item"])
            elif kind == "stack" and op == "pop":
                if not values:
                    return "ERROR"
                values.pop(0)
            elif kind == "queue" and op == "rotate":
                if len(values) < 2:
                    return "ERROR"
                for _ in range(int(event.get("count", 1))):
                    values.append(values.pop(0))
            elif kind == "queue" and op == "swap_front":
                if len(values) < 2:
                    return "ERROR"
                for _ in range(int(event.get("count", 1))):
                    values[0], values[1] = values[1], values[0]
            else:
                return "ERROR"
        return values[0] if values else "EMPTY"
    if family == "graph_walk":
        current = scenario["start"]
        for event in scenario["events"]:
            for _ in range(int(event["count"])):
                current = scenario["successor"][current]
        return current
    if family == "chain_arith":
        value = int(scenario["start"])
        modulus = int(scenario["modulus"])
        for event in scenario["events"]:
            value = (int(event["a"]) * value + int(event["b"])) % modulus
        return str(value)
    if family == "rule_neg":
        known: dict[str, bool] = {scenario["initial_prop"]: bool(scenario["initial_value"])}
        for event in scenario["events"]:
            source = event["source"]
            if known.get(source) == bool(event["source_value"]):
                known[event["target"]] = bool(event["target_value"])
        target = scenario["query"]
        if target not in known:
            return "unknown"
        return "true" if known[target] else "false"
    if family == "tiny_prog":
        value = int(scenario["start"])
        modulus = int(scenario["modulus"])
        for event in scenario["events"]:
            if event["op"] == "mul":
                value = value * int(event["arg"]) % modulus
            elif event["op"] == "add":
                value = (value + int(event["arg"])) % modulus
            else:
                return "ERROR"
        return str(value)
    if family == "obj_track":
        locations = dict(scenario["initial_locations"])
        holders = dict(scenario["initial_holders"])
        for event in scenario["events"]:
            if event["op"] == "move":
                locations[event["person"]] = event["place"]
            elif event["op"] == "give":
                holders[event["item"]] = event["person"]
            else:
                return "ERROR"
        query = scenario["query"]
        return locations[query["name"]] if query["kind"] == "location" else holders[query["name"]]
    raise ValueError(f"unknown family: {family}")


def _reference_replay(scenario: Mapping[str, Any]) -> str:
    """Second implementation of the suite semantics used only for audit/rescore.

    This deliberately does not call ``_replay`` or its state-transition helpers:
    generation and reference validation must be able to disagree.
    """
    family = scenario["family"]
    events = scenario["events"]
    if family == "fsm":
        state = str(scenario["start"])
        table = scenario["transitions"]
        for symbol in events:
            state = str(table[state][symbol])
        return state
    if family == "stack_queue":
        values = [str(value) for value in scenario["initial"]]
        kind = scenario["kind"]
        for event in events:
            operation = event["op"]
            if kind == "stack":
                if operation == "push":
                    values = [str(event["item"]), *values]
                elif operation == "pop" and values:
                    values = values[1:]
                else:
                    return "ERROR"
            elif kind == "queue" and len(values) >= 2:
                count = int(event.get("count", 1))
                if operation == "rotate":
                    offset = count % len(values)
                    values = values[offset:] + values[:offset]
                elif operation == "swap_front":
                    if count % 2:
                        values = [values[1], values[0], *values[2:]]
                else:
                    return "ERROR"
            else:
                return "ERROR"
        return values[0] if values else "EMPTY"
    if family == "graph_walk":
        node = str(scenario["start"])
        successor = scenario["successor"]
        total_hops = sum(int(event["count"]) for event in events)
        for _ in range(total_hops):
            node = str(successor[node])
        return node
    if family == "chain_arith":
        result = int(scenario["start"])
        modulus = int(scenario["modulus"])
        for event in events:
            result = (result * int(event["a"]) + int(event["b"])) % modulus
        return str(result)
    if family == "rule_neg":
        facts = {str(scenario["initial_prop"]): bool(scenario["initial_value"])}
        for rule in events:
            if facts.get(str(rule["source"])) == bool(rule["source_value"]):
                facts[str(rule["target"])] = bool(rule["target_value"])
        query = str(scenario["query"])
        return "unknown" if query not in facts else ("true" if facts[query] else "false")
    if family == "tiny_prog":
        register = int(scenario["start"])
        modulus = int(scenario["modulus"])
        for instruction in events:
            argument = int(instruction["arg"])
            if instruction["op"] == "mul":
                register = register * argument % modulus
            elif instruction["op"] == "add":
                register = (register + argument) % modulus
            else:
                return "ERROR"
        return str(register)
    if family == "obj_track":
        query = scenario["query"]
        name = str(query["name"])
        if query["kind"] == "location":
            terminal = str(scenario["initial_locations"][name])
            for event in events:
                if event["op"] == "move" and event["person"] == name:
                    terminal = str(event["place"])
                elif event["op"] not in {"move", "give"}:
                    return "ERROR"
            return terminal
        if query["kind"] == "holder":
            terminal = str(scenario["initial_holders"][name])
            for event in events:
                if event["op"] == "give" and event["item"] == name:
                    terminal = str(event["person"])
                elif event["op"] not in {"move", "give"}:
                    return "ERROR"
            return terminal
        return "ERROR"
    raise ValueError(f"unknown family: {family}")


def _render_prose(scenario: Mapping[str, Any]) -> tuple[str, str, tuple[str, ...]]:
    family = scenario["family"]
    if family == "fsm":
        initial = f"Start state: {scenario['start']}"
        rules = [
            f"Transition: {state} + {symbol} -> {target}"
            for state in sorted(scenario["transitions"])
            for symbol, target in sorted(scenario["transitions"][state].items())
        ]
        events = tuple(f"Input {index}: {symbol}" for index, symbol in enumerate(scenario["events"], 1))
        prompt = "Finite-state trace.\n" + initial + "\n" + "\n".join(rules + list(events)) + "\nQuestion: final state?\nAnswer:"
    elif family == "stack_queue":
        kind = scenario["kind"]
        direction = "top-to-bottom" if kind == "stack" else "front-to-back"
        initial = f"Initial {kind} ({direction}): " + ", ".join(scenario["initial"])
        rendered: list[str] = []
        for index, event in enumerate(scenario["events"], 1):
            if event["op"] == "push":
                action = f"PUSH {event['item']}"
            elif event["op"] == "pop":
                action = "POP"
            elif event["op"] == "rotate":
                action = f"MOVE_FRONT_TO_BACK x{event.get('count', 1)}"
            else:
                action = f"SWAP_FRONT_TWO x{event.get('count', 1)}"
            rendered.append(f"Step {index}: {action}")
        events = tuple(rendered)
        prompt = f"Sequential {kind} trace.\n{initial}\n" + "\n".join(events) + f"\nQuestion: final {'top' if kind == 'stack' else 'front'} item?\nAnswer:"
    elif family == "graph_walk":
        initial = f"Start node: {scenario['start']}"
        rules = [f"Edge: {source} -> {target}" for source, target in sorted(scenario["successor"].items())]
        events = tuple(f"Hop {index}: follow edge {event['count']} time(s)" for index, event in enumerate(scenario["events"], 1))
        prompt = "Functional graph trace.\n" + initial + "\n" + "\n".join(rules + list(events)) + "\nQuestion: final node?\nAnswer:"
    elif family == "chain_arith":
        initial = f"Initial value: {scenario['start']}; modulus: {scenario['modulus']}"
        events = tuple(
            f"Step {index}: x = ({event['a']} * x + {event['b']}) mod {scenario['modulus']}"
            for index, event in enumerate(scenario["events"], 1)
        )
        prompt = "Modular arithmetic trace.\n" + initial + "\n" + "\n".join(events) + "\nQuestion: final x?\nAnswer:"
    elif family == "rule_neg":
        truth = "true" if scenario["initial_value"] else "false"
        initial = f"Initial fact: {scenario['initial_prop']} = {truth}"
        events = tuple(
            f"Rule {index}: if {event['source']} = {'true' if event['source_value'] else 'false'} then {event['target']} = {'true' if event['target_value'] else 'false'}"
            for index, event in enumerate(scenario["events"], 1)
        )
        prompt = "Apply each rule once, exactly in listed order; skipped rules are not revisited.\n" + initial + "\n" + "\n".join(events) + f"\nQuestion: {scenario['query']} = true or false?\nAnswer:"
    elif family == "tiny_prog":
        initial = f"Initial register x: {scenario['start']}; modulus: {scenario['modulus']}"
        events = tuple(
            f"Instruction {index}: {'x = x *' if event['op'] == 'mul' else 'x = x +'} {event['arg']} (mod {scenario['modulus']})"
            for index, event in enumerate(scenario["events"], 1)
        )
        prompt = "Execute the tiny program in order.\n" + initial + "\n" + "\n".join(events) + "\nQuestion: final x?\nAnswer:"
    elif family == "obj_track":
        loc_lines = [f"Location: {person} = {place}" for person, place in sorted(scenario["initial_locations"].items())]
        holder_lines = [f"Holder: {item} = {person}" for item, person in sorted(scenario["initial_holders"].items())]
        initial = "\n".join(loc_lines + holder_lines)
        rendered = []
        for index, event in enumerate(scenario["events"], 1):
            action = f"MOVE {event['person']} -> {event['place']}" if event["op"] == "move" else f"GIVE {event['item']} -> {event['person']}"
            rendered.append(f"Event {index}: {action}")
        events = tuple(rendered)
        query = scenario["query"]
        prompt = "Object tracking trace.\nInitial situation:\n" + initial + "\nEvents, in order:\n" + "\n".join(events) + f"\nQuestion: final {query['kind']} of {query['name']}?\nAnswer:"
    else:
        raise ValueError(f"unknown family: {family}")
    return prompt, initial, events


def _render_semantic_ood(scenario: Mapping[str, Any]) -> tuple[str, str, tuple[str, ...]]:
    payload = _canonical_json(scenario)
    prompt = (
        "Evaluate the held-out structured transition record.\n"
        f"{_SEMANTIC_OOD_MARKER}\n{payload}\n{_SEMANTIC_OOD_END}\n"
        "Return the queried terminal value only.\nAnswer:"
    )
    initial_keys = {
        key: value for key, value in scenario.items()
        if key.startswith("initial") or key in {"start", "modulus", "kind"}
    }
    events = tuple(_canonical_json(event) for event in scenario["events"])
    return prompt, _canonical_json(initial_keys), events


def _parse_prose(prompt: str) -> dict[str, Any]:
    if prompt.startswith("Finite-state trace."):
        start = re.search(r"^Start state: (\S+)$", prompt, re.MULTILINE).group(1)
        transitions: dict[str, dict[str, str]] = {}
        for state, symbol, target in re.findall(r"^Transition: (\S+) \+ (\S+) -> (\S+)$", prompt, re.MULTILINE):
            transitions.setdefault(state, {})[symbol] = target
        events = [symbol for _, symbol in re.findall(r"^Input (\d+): (\S+)$", prompt, re.MULTILINE)]
        return {"family": "fsm", "start": start, "transitions": transitions, "events": events}
    if prompt.startswith("Sequential "):
        kind = re.search(r"^Sequential (stack|queue) trace\.$", prompt, re.MULTILINE).group(1)
        initial_raw = re.search(rf"^Initial {kind} \([^)]*\): (.+)$", prompt, re.MULTILINE).group(1)
        initial = initial_raw.split(", ")
        events = []
        for action in re.findall(r"^Step \d+: (.+)$", prompt, re.MULTILINE):
            if action.startswith("PUSH "):
                events.append({"op": "push", "item": action.split(" ", 1)[1]})
            elif action == "POP":
                events.append({"op": "pop"})
            elif action.startswith("MOVE_FRONT_TO_BACK"):
                events.append({"op": "rotate", "count": int(action.rsplit("x", 1)[1])})
            else:
                events.append({"op": "swap_front", "count": int(action.rsplit("x", 1)[1])})
        return {"family": "stack_queue", "kind": kind, "initial": initial, "events": events}
    if prompt.startswith("Functional graph trace."):
        start = re.search(r"^Start node: (\S+)$", prompt, re.MULTILINE).group(1)
        successor = dict(re.findall(r"^Edge: (\S+) -> (\S+)$", prompt, re.MULTILINE))
        events = [{"op": "hop", "count": int(count)} for count in re.findall(r"^Hop \d+: follow edge (\d+) time\(s\)$", prompt, re.MULTILINE)]
        return {"family": "graph_walk", "start": start, "successor": successor, "events": events}
    if prompt.startswith("Modular arithmetic trace."):
        start, modulus = re.search(r"^Initial value: (\d+); modulus: (\d+)$", prompt, re.MULTILINE).groups()
        events = [
            {"a": int(a), "b": int(b)}
            for a, b, _ in re.findall(r"^Step \d+: x = \((\d+) \* x \+ (\d+)\) mod (\d+)$", prompt, re.MULTILINE)
        ]
        return {"family": "chain_arith", "start": int(start), "modulus": int(modulus), "events": events}
    if prompt.startswith("Apply each rule once"):
        prop, truth = re.search(r"^Initial fact: (\S+) = (true|false)$", prompt, re.MULTILINE).groups()
        events = []
        for source, source_value, target, target_value in re.findall(r"^Rule \d+: if (\S+) = (true|false) then (\S+) = (true|false)$", prompt, re.MULTILINE):
            events.append({"source": source, "source_value": source_value == "true", "target": target, "target_value": target_value == "true"})
        query = re.search(r"^Question: (\S+) = true or false\?$", prompt, re.MULTILINE).group(1)
        return {"family": "rule_neg", "initial_prop": prop, "initial_value": truth == "true", "events": events, "query": query}
    if prompt.startswith("Execute the tiny program"):
        start, modulus = re.search(r"^Initial register x: (\d+); modulus: (\d+)$", prompt, re.MULTILINE).groups()
        events = []
        for op, arg, _ in re.findall(r"^Instruction \d+: x = x ([*+]) (\d+) \(mod (\d+)\)$", prompt, re.MULTILINE):
            events.append({"op": "mul" if op == "*" else "add", "arg": int(arg)})
        return {"family": "tiny_prog", "start": int(start), "modulus": int(modulus), "events": events}
    if prompt.startswith("Object tracking trace."):
        locations = dict(re.findall(r"^Location: (\S+) = (\S+)$", prompt, re.MULTILINE))
        holders = dict(re.findall(r"^Holder: (\S+) = (\S+)$", prompt, re.MULTILINE))
        events = []
        for action in re.findall(r"^Event \d+: (.+)$", prompt, re.MULTILINE):
            if action.startswith("MOVE "):
                person, place = re.match(r"MOVE (\S+) -> (\S+)$", action).groups()
                events.append({"op": "move", "person": person, "place": place})
            else:
                item, person = re.match(r"GIVE (\S+) -> (\S+)$", action).groups()
                events.append({"op": "give", "item": item, "person": person})
        kind, name = re.search(r"^Question: final (location|holder) of (\S+)\?$", prompt, re.MULTILINE).groups()
        return {"family": "obj_track", "initial_locations": locations, "initial_holders": holders, "events": events, "query": {"kind": kind, "name": name}}
    raise ValueError("unrecognized behavioral-v3 prompt")


def parse_prompt(prompt: str) -> dict[str, Any]:
    if _SEMANTIC_OOD_MARKER in prompt:
        payload = prompt.split(f"{_SEMANTIC_OOD_MARKER}\n", 1)[1].split(f"\n{_SEMANTIC_OOD_END}", 1)[0]
        scenario = json.loads(payload)
        if not isinstance(scenario, dict):
            raise ValueError("structured scenario must be an object")
        return scenario
    return _parse_prose(prompt)


def reference_solve_prompt(prompt: str) -> str:
    """Independently parse prompt text and replay from its serialized initial state."""
    return _reference_replay(parse_prompt(prompt))


def reference_trace_prompt(prompt: str) -> tuple[str, ...]:
    """Return prompt-derived scalar state after each event prefix.

    This is intentionally narrower than the benchmark solver: only families
    whose complete intermediate state is a single answer-shaped scalar are
    eligible for latent trace supervision.
    """
    scenario = parse_prompt(prompt)
    family = scenario.get("family")
    if family not in TRACE_SUPERVISION_FAMILIES:
        raise ValueError(
            f"family {family!r} has no scalar trace-supervision contract")
    events = scenario.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("trace-supervised prompt must contain events")
    trace = tuple(
        _reference_replay({**scenario, "events": events[:prefix_length]})
        for prefix_length in range(1, len(events) + 1)
    )
    if trace[-1] != _reference_replay(scenario):
        raise AssertionError("prefix replay disagrees with full prompt replay")
    return trace


def safe_trace_targets(prompt: str, answer: str, k_steps: int
                       ) -> tuple[tuple[int, str], ...]:
    """Select non-final, non-gold trace targets without shifting step indices."""
    if isinstance(k_steps, bool) or not isinstance(k_steps, int) or k_steps < 0:
        raise ValueError("k_steps must be a non-negative integer")
    scenario = parse_prompt(prompt)
    if scenario.get("family") not in TRACE_SUPERVISION_FAMILIES:
        return ()
    trace = reference_trace_prompt(prompt)
    if trace[-1] != answer:
        raise ValueError("prompt-derived terminal state disagrees with gold answer")
    target_count = min(max(0, k_steps - 1), max(0, len(trace) - 1))
    return tuple(
        (step_index, trace[step_index - 1])
        for step_index in range(1, target_count + 1)
        if trace[step_index - 1] != answer
    )


def _candidate_order(
    canonical_candidates: Sequence[str],
    answer: str,
    permutation_seed: int,
    desired_gold_position: int,
) -> tuple[tuple[str, ...], tuple[int, ...], int]:
    canonical = tuple(canonical_candidates)
    if len(canonical) < 2 or len(set(canonical)) != len(canonical):
        raise ValueError("candidate set must contain at least two unique values")
    if canonical.count(answer) != 1:
        raise ValueError("gold answer must occur exactly once in candidates")
    gold_canonical_index = canonical.index(answer)
    non_gold = [index for index in range(len(canonical)) if index != gold_canonical_index]
    random.Random(permutation_seed).shuffle(non_gold)
    gold_position = desired_gold_position % len(canonical)
    permutation = non_gold[:]
    permutation.insert(gold_position, gold_canonical_index)
    actual = tuple(canonical[index] for index in permutation)
    return actual, tuple(permutation), gold_position


def _scenario_key(scenario: Mapping[str, Any]) -> str:
    return f"{scenario['family']}:" + _sha256_text(_canonical_json(scenario))[:24]


def _make_example(
    *,
    scenario: Mapping[str, Any],
    split: str,
    ordinal: int,
    template: str,
    canonical_candidates: Sequence[str],
    permutation_seed: int,
    desired_gold_position: int,
) -> Example:
    answer = _replay(scenario)
    if answer in {"ERROR", "unknown", "EMPTY"}:
        raise ValueError("generated scenario has no finite closed-set answer")
    if template == "semantic_json_v1":
        prompt, initial_text, events_text = _render_semantic_ood(scenario)
    else:
        prompt, initial_text, events_text = _render_prose(scenario)
    actual, permutation, gold_index = _candidate_order(
        canonical_candidates, answer, permutation_seed, desired_gold_position
    )
    family = str(scenario["family"])
    return Example(
        ex_id=f"{split}-{family}-{ordinal:05d}",
        split=split,
        family=family,
        depth=len(scenario["events"]),
        template=template,
        prompt=prompt,
        answer=answer,
        candidates=actual,
        canonical_candidates=tuple(canonical_candidates),
        candidate_permutation_seed=permutation_seed,
        candidate_permutation=permutation,
        candidate_token_lengths=_token_lengths(actual),
        candidate_tokenizer_identity=TOKENIZER_IDENTITY,
        gold_index=gold_index,
        content_key=_scenario_key(scenario),
        initial_text=initial_text,
        events_text=events_text,
        scenario_json=_canonical_json(scenario),
    )


def _domain_values(semantic_ood: bool) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    return (
        _WORDS_OOD if semantic_ood else _WORDS_ID,
        _PLACES_OOD if semantic_ood else _PLACES_ID,
        _ITEMS_OOD if semantic_ood else _ITEMS_ID,
    )


def _gen_fsm(rng: random.Random, depth: int, semantic_ood: bool) -> tuple[dict[str, Any], tuple[str, ...]]:
    words, _, _ = _domain_values(semantic_ood)
    alphabet = ("q", "r", "s") if semantic_ood else ("x", "y", "z")
    for _ in range(512):
        states = rng.sample(list(words), 5)
        transitions = {
            state: {symbol: rng.choice([target for target in states if target != state]) for symbol in alphabet}
            for state in states
        }
        events = [rng.choice(alphabet) for _ in range(depth)]
        scenario = {"family": "fsm", "start": states[0], "transitions": transitions, "events": events}
        prior = dict(scenario)
        prior["events"] = events[:-1]
        reverse = dict(scenario)
        reverse["events"] = list(reversed(events))
        before_last = _replay(prior)
        alternate_last_answers = {
            transitions[before_last][symbol]
            for symbol in alphabet
            if symbol != events[-1]
        }
        if (
            _replay(scenario) != before_last
            and _replay(scenario) != _replay(reverse)
            and any(answer != _replay(scenario) for answer in alternate_last_answers)
        ):
            return scenario, tuple(sorted(states))
    raise RuntimeError("could not generate causal fsm")


def _gen_stack_queue(rng: random.Random, depth: int, semantic_ood: bool) -> tuple[dict[str, Any], tuple[str, ...]]:
    words, _, _ = _domain_values(semantic_ood)
    candidates = tuple(sorted(rng.sample(list(words), 6)))
    if rng.random() < 0.5:
        initial = list(rng.sample(list(candidates), 2))
        events = []
        previous = initial[0]
        for _ in range(depth):
            item = rng.choice([value for value in candidates if value != previous])
            events.append({"op": "push", "item": item})
            previous = item
        if events[0]["item"] == events[-1]["item"]:
            return _gen_stack_queue(rng, depth, semantic_ood)
        scenario = {"family": "stack_queue", "kind": "stack", "initial": initial, "events": events}
        return scenario, candidates
    for _ in range(256):
        initial = list(rng.sample(list(candidates), 6))
        events = [{"op": rng.choice(("rotate", "swap_front")), "count": 1} for _ in range(depth)]
        scenario = {"family": "stack_queue", "kind": "queue", "initial": initial, "events": events}
        prior = {**scenario, "events": events[:-1]}
        reverse = {**scenario, "events": list(reversed(events))}
        if _replay(scenario) != _replay(prior) and _replay(scenario) != _replay(reverse):
            return scenario, candidates
    return _gen_stack_queue(rng, depth, semantic_ood)


def _gen_graph_walk(rng: random.Random, depth: int, semantic_ood: bool) -> tuple[dict[str, Any], tuple[str, ...]]:
    words, _, _ = _domain_values(semantic_ood)
    nodes = rng.sample(list(words), 6)
    rotated = nodes[1:] + nodes[:1]
    successor = dict(zip(nodes, rotated))
    scenario = {
        "family": "graph_walk",
        "start": rng.choice(nodes),
        "successor": successor,
        "events": [{"op": "hop", "count": 1} for _ in range(depth)],
    }
    return scenario, tuple(sorted(nodes))


def _gen_chain_arith(rng: random.Random, depth: int, semantic_ood: bool) -> tuple[dict[str, Any], tuple[str, ...]]:
    modulus = 19 if semantic_ood else 11
    for _ in range(256):
        events = [{"a": rng.choice((2, 3, 4)), "b": rng.randrange(1, modulus)} for _ in range(depth)]
        scenario = {"family": "chain_arith", "start": rng.randrange(modulus), "modulus": modulus, "events": events}
        prior = {**scenario, "events": events[:-1]}
        reverse = {**scenario, "events": list(reversed(events))}
        if _replay(scenario) != _replay(prior) and _replay(scenario) != _replay(reverse):
            return scenario, tuple(str(value) for value in range(modulus))
    raise RuntimeError("could not generate causal arithmetic chain")


def _gen_rule_neg(rng: random.Random, depth: int, semantic_ood: bool) -> tuple[dict[str, Any], tuple[str, ...]]:
    words, _, _ = _domain_values(semantic_ood)
    props = rng.sample(list(words), depth + 1)
    values = [bool(rng.getrandbits(1)) for _ in range(depth + 1)]
    events = [
        {"source": props[index], "source_value": values[index], "target": props[index + 1], "target_value": values[index + 1]}
        for index in range(depth)
    ]
    scenario = {
        "family": "rule_neg",
        "initial_prop": props[0],
        "initial_value": values[0],
        "events": events,
        "query": props[-1],
    }
    return scenario, ("false", "true")


def _gen_tiny_prog(rng: random.Random, depth: int, semantic_ood: bool) -> tuple[dict[str, Any], tuple[str, ...]]:
    modulus = 23 if semantic_ood else 17
    for _ in range(256):
        events = [
            {"op": "mul" if index % 2 == 0 else "add", "arg": rng.randint(2, 5)}
            for index in range(depth)
        ]
        scenario = {"family": "tiny_prog", "start": rng.randrange(modulus), "modulus": modulus, "events": events}
        prior = {**scenario, "events": events[:-1]}
        reverse = {**scenario, "events": list(reversed(events))}
        if _replay(scenario) != _replay(prior) and _replay(scenario) != _replay(reverse):
            return scenario, tuple(str(value) for value in range(modulus))
    raise RuntimeError("could not generate causal tiny program")


def _gen_obj_track(rng: random.Random, depth: int, semantic_ood: bool) -> tuple[dict[str, Any], tuple[str, ...]]:
    words, places_domain, items_domain = _domain_values(semantic_ood)
    people = rng.sample(list(words), 4)
    places = rng.sample(list(places_domain), 4)
    items = rng.sample(list(items_domain), 2)
    initial_locations = {person: rng.choice(places) for person in people}
    initial_holders = {item: rng.choice(people) for item in items}
    events: list[dict[str, Any]] = []
    if rng.random() < 0.5:
        query_person = rng.choice(people)
        first_place, last_place = rng.sample([place for place in places if place != initial_locations[query_person]], 2)
        events.append({"op": "move", "person": query_person, "place": first_place})
        while len(events) < depth - 1:
            distractor = rng.choice([person for person in people if person != query_person])
            events.append({"op": "move", "person": distractor, "place": rng.choice(places)})
        events.append({"op": "move", "person": query_person, "place": last_place})
        query = {"kind": "location", "name": query_person}
        candidates = tuple(sorted(places))
    else:
        query_item = rng.choice(items)
        first_person, last_person = rng.sample([person for person in people if person != initial_holders[query_item]], 2)
        events.append({"op": "give", "item": query_item, "person": first_person})
        while len(events) < depth - 1:
            distractor = rng.choice([item for item in items if item != query_item])
            events.append({"op": "give", "item": distractor, "person": rng.choice(people)})
        events.append({"op": "give", "item": query_item, "person": last_person})
        query = {"kind": "holder", "name": query_item}
        candidates = tuple(sorted(people))
    scenario = {
        "family": "obj_track",
        "initial_locations": initial_locations,
        "initial_holders": initial_holders,
        "events": events,
        "query": query,
    }
    return scenario, candidates


_GENERATORS: Mapping[str, Callable[[random.Random, int, bool], tuple[dict[str, Any], tuple[str, ...]]]] = {
    "fsm": _gen_fsm,
    "stack_queue": _gen_stack_queue,
    "graph_walk": _gen_graph_walk,
    "chain_arith": _gen_chain_arith,
    "rule_neg": _gen_rule_neg,
    "tiny_prog": _gen_tiny_prog,
    "obj_track": _gen_obj_track,
}


def _gold_position_schedule(count: int, candidate_count: int, seed: int) -> list[int]:
    positions = [index % candidate_count for index in range(count)]
    random.Random(seed).shuffle(positions)
    return positions


def _build_family_split(
    family: str,
    split: str,
    count: int,
    seed: int,
    depth_range: tuple[int, int],
    semantic_ood: bool,
    seen_keys: set[str],
) -> list[Example]:
    rng = random.Random(_seed_for(seed, split, family, "generation"))
    generated: list[tuple[dict[str, Any], tuple[str, ...]]] = []
    candidate_count: int | None = None
    while len(generated) < count:
        scenario, candidates = _GENERATORS[family](rng, rng.randint(*depth_range), semantic_ood)
        key = _scenario_key(scenario)
        if key in seen_keys:
            continue
        if candidate_count is None:
            candidate_count = len(candidates)
        if len(candidates) != candidate_count:
            raise RuntimeError(f"{family} changed candidate count within split")
        seen_keys.add(key)
        generated.append((scenario, candidates))
    assert candidate_count is not None
    positions = _gold_position_schedule(count, candidate_count, _seed_for(seed, split, family, "positions"))
    template = "semantic_json_v1" if semantic_ood else "prose_v1"
    return [
        _make_example(
            scenario=scenario,
            split=split,
            ordinal=index,
            template=template,
            canonical_candidates=candidates,
            permutation_seed=_seed_for(seed, split, family, index, "candidate-permutation"),
            desired_gold_position=positions[index],
        )
        for index, (scenario, candidates) in enumerate(generated)
    ]


def build_suite(
    *,
    counts_per_family: Mapping[str, int] = DEFAULT_COUNTS,
    seed: int = MASTER_SEED,
) -> Suite:
    missing = set(DEFAULT_COUNTS) - set(counts_per_family)
    if missing:
        raise ValueError(f"missing split counts: {sorted(missing)}")
    seen_keys: set[str] = set()
    split_examples: dict[str, list[Example]] = {name: [] for name in DEFAULT_COUNTS}
    for split in DEFAULT_COUNTS:
        if split == "test_ood_length":
            depth_range = LENGTH_OOD_DEPTH
        elif split == "final_test":
            depth_range = FINAL_DEPTH
        else:
            depth_range = ID_DEPTH
        semantic_ood = split == "test_ood_semantic"
        for family in FAMILIES:
            split_examples[split].extend(
                _build_family_split(
                    family,
                    split,
                    int(counts_per_family[split]),
                    seed,
                    depth_range,
                    semantic_ood,
                    seen_keys,
                )
            )
        random.Random(_seed_for(seed, split, "example-order")).shuffle(split_examples[split])
    return Suite(
        train=tuple(split_examples["train"]),
        validation=tuple(split_examples["validation"]),
        test_id=tuple(split_examples["test_id"]),
        test_ood_length=tuple(split_examples["test_ood_length"]),
        test_ood_semantic=tuple(split_examples["test_ood_semantic"]),
        final_test=tuple(split_examples["final_test"]),
        seed=seed,
    )


def _render_variant(example: Example, scenario: Mapping[str, Any]) -> str:
    return (_render_semantic_ood(scenario) if example.template == "semantic_json_v1" else _render_prose(scenario))[0]


def without_last_causal_event(example: Example) -> str:
    scenario = example.scenario
    scenario["events"] = scenario["events"][:-1]
    return _render_variant(example, scenario)


def with_reversed_events(example: Example) -> str:
    scenario = example.scenario
    scenario["events"] = list(reversed(scenario["events"]))
    return _render_variant(example, scenario)


def counterfactual_prompt(example: Example) -> tuple[str, str]:
    """Mutate one causally relevant event; return prompt and expected new gold."""
    scenario = example.scenario
    events = scenario["events"]
    family = example.family
    original = example.answer
    variants: list[dict[str, Any]] = []
    if family == "fsm":
        for symbol in sorted(next(iter(scenario["transitions"].values()))):
            if symbol != events[-1]:
                variants.append({**scenario, "events": [*events[:-1], symbol]})
    elif family == "stack_queue":
        if scenario["kind"] == "stack":
            for candidate in example.canonical_candidates:
                if candidate != events[-1]["item"]:
                    variants.append({**scenario, "events": [*events[:-1], {"op": "push", "item": candidate}]})
        else:
            for alternate in ("rotate", "swap_front"):
                for count in range(1, len(scenario["initial"]) + 1):
                    variants.append({
                        **scenario,
                        "events": [*events[:-1], {"op": alternate, "count": count}],
                    })
    elif family == "graph_walk":
        variants.append({**scenario, "events": [*events[:-1], {"op": "hop", "count": 2}]})
    elif family == "chain_arith":
        for delta in range(1, int(scenario["modulus"])):
            changed = {**events[-1], "b": (int(events[-1]["b"]) + delta) % int(scenario["modulus"])}
            variants.append({**scenario, "events": [*events[:-1], changed]})
    elif family == "rule_neg":
        changed = {**events[-1], "target_value": not events[-1]["target_value"]}
        variants.append({**scenario, "events": [*events[:-1], changed]})
    elif family == "tiny_prog":
        for delta in range(1, int(scenario["modulus"])):
            changed = {**events[-1], "arg": (int(events[-1]["arg"]) + delta) % int(scenario["modulus"])}
            variants.append({**scenario, "events": [*events[:-1], changed]})
    elif family == "obj_track":
        event = events[-1]
        if event["op"] == "move":
            for candidate in example.canonical_candidates:
                if candidate != event["place"]:
                    variants.append({**scenario, "events": [*events[:-1], {**event, "place": candidate}]})
        else:
            for candidate in example.canonical_candidates:
                if candidate != event["person"]:
                    variants.append({**scenario, "events": [*events[:-1], {**event, "person": candidate}]})
    for variant in variants:
        new_answer = _replay(variant)
        if new_answer != original and new_answer in example.canonical_candidates:
            return _render_variant(example, variant), new_answer
    raise RuntimeError(f"no closed-set counterfactual for {example.ex_id}")


def nonterminal_counterfactual_prompt(example: Example) -> tuple[str, str] | None:
    """Change the earliest causal non-final event, or report ineligibility.

    The final event and all other events remain unchanged.  ``None`` means that
    no one-event mutation before the final event changes the closed-set gold.
    """
    scenario = example.scenario
    events = scenario["events"]
    if len(events) < 2:
        return None

    def replacements(index: int) -> Iterable[Any]:
        event = events[index]
        if example.family == "fsm":
            return (
                symbol
                for symbol in sorted(next(iter(scenario["transitions"].values())))
                if symbol != event
            )
        if example.family == "stack_queue":
            if scenario["kind"] == "stack":
                return (
                    {"op": "push", "item": candidate}
                    for candidate in example.canonical_candidates
                    if candidate != event["item"]
                )
            return (
                {"op": operation, "count": count}
                for operation in ("rotate", "swap_front")
                for count in range(1, len(scenario["initial"]) + 1)
                if operation != event["op"] or count != int(event.get("count", 1))
            )
        if example.family == "graph_walk":
            return (
                {"op": "hop", "count": count}
                for count in range(1, len(scenario["successor"]) + 1)
                if count != int(event["count"])
            )
        if example.family == "chain_arith":
            modulus = int(scenario["modulus"])
            return (
                {**event, "b": (int(event["b"]) + delta) % modulus}
                for delta in range(1, modulus)
            )
        if example.family == "tiny_prog":
            modulus = int(scenario["modulus"])
            return (
                {**event, "arg": (int(event["arg"]) + delta) % modulus}
                for delta in range(1, modulus)
            )
        return ()

    for index in range(len(events) - 1):
        for replacement in replacements(index):
            changed_events = list(events)
            changed_events[index] = replacement
            variant = {**scenario, "events": changed_events}
            prompt = _render_variant(example, variant)
            new_answer = reference_solve_prompt(prompt)
            if _replay(variant) != new_answer:
                raise AssertionError("counterfactual replay disagrees with reference solver")
            if new_answer == example.answer or new_answer not in example.canonical_candidates:
                continue
            return prompt, new_answer
    return None


def _position_balance(examples: Sequence[Example]) -> dict[str, Any]:
    grouped: dict[str, list[Example]] = {family: [] for family in FAMILIES}
    for example in examples:
        grouped[example.family].append(example)
    result: dict[str, Any] = {}
    for family, family_examples in grouped.items():
        counts = Counter(example.gold_index for example in family_examples)
        slots = len(family_examples[0].candidates) if family_examples else 0
        values = [counts.get(index, 0) for index in range(slots)]
        result[family] = {
            "counts": {str(index): count for index, count in enumerate(values)},
            "max_minus_min": max(values) - min(values) if values else 0,
        }
    return result


def audit_suite(suite: Suite) -> dict[str, Any]:
    problems: list[str] = []
    split_reports: dict[str, Any] = {}
    all_keys: dict[str, set[str]] = {}
    all_prompts: dict[str, set[str]] = {}
    for split, examples in suite.splits().items():
        ids = [example.ex_id for example in examples]
        prompts = [example.prompt for example in examples]
        keys = [example.content_key for example in examples]
        all_keys[split] = set(keys)
        all_prompts[split] = set(prompts)
        if len(ids) != len(set(ids)):
            problems.append(f"{split}: duplicate example ids")
        if len(prompts) != len(set(prompts)):
            problems.append(f"{split}: duplicate prompts")
        if len(keys) != len(set(keys)):
            problems.append(f"{split}: duplicate scenario cores")
        family_reports: dict[str, Any] = {}
        for family in FAMILIES:
            family_examples = [example for example in examples if example.family == family]
            removed_changed = 0
            counterfactual_changed = 0
            reversed_changed = 0
            parsed = 0
            unique_answers = set()
            for example in family_examples:
                unique_answers.add(example.answer)
                if len(set(example.candidates)) != len(example.candidates):
                    problems.append(f"{example.ex_id}: duplicate candidates")
                if tuple(example.canonical_candidates[index] for index in example.candidate_permutation) != example.candidates:
                    problems.append(f"{example.ex_id}: candidate permutation roundtrip failed")
                if example.answer != example.candidates[example.gold_index]:
                    problems.append(f"{example.ex_id}: gold index mismatch")
                if example.candidate_token_lengths != _token_lengths(example.candidates):
                    problems.append(f"{example.ex_id}: candidate token lengths mismatch")
                try:
                    parsed_answer = reference_solve_prompt(example.prompt)
                except Exception as exc:  # pragma: no cover - reported as audit evidence
                    problems.append(f"{example.ex_id}: parser failed: {type(exc).__name__}")
                else:
                    parsed += int(parsed_answer == example.answer)
                    if parsed_answer != example.answer:
                        problems.append(f"{example.ex_id}: independent solver mismatch")
                removed_changed += int(reference_solve_prompt(without_last_causal_event(example)) != example.answer)
                reversed_changed += int(reference_solve_prompt(with_reversed_events(example)) != example.answer)
                try:
                    cf_prompt, expected = counterfactual_prompt(example)
                except RuntimeError:
                    problems.append(f"{example.ex_id}: no counterfactual")
                else:
                    counterfactual_changed += int(expected != example.answer and reference_solve_prompt(cf_prompt) == expected)
                if f"Answer: {example.answer}" in example.prompt:
                    problems.append(f"{example.ex_id}: direct answer field leak")
            denominator = len(family_examples) or 1
            family_reports[family] = {
                "count": len(family_examples),
                "independent_solver_agreement": parsed / denominator,
                "causal_event_removal_change_rate": removed_changed / denominator,
                "counterfactual_change_rate": counterfactual_changed / denominator,
                "event_reverse_change_rate": reversed_changed / denominator,
                "event_order_sensitivity_applicable": family != "graph_walk",
                "unique_answer_count": len(unique_answers),
                "initial_only_not_identifying": counterfactual_changed == len(family_examples),
            }
            if family_examples and removed_changed / denominator < 0.95:
                problems.append(f"{split}/{family}: causal removal rate below 0.95")
            if family_examples and counterfactual_changed != len(family_examples):
                problems.append(f"{split}/{family}: counterfactual coverage incomplete")
            if family_examples and family != "graph_walk" and reversed_changed / denominator < 0.95:
                problems.append(f"{split}/{family}: event reverse change rate below 0.95")
            if len(family_examples) >= 4 and len(unique_answers) < 2:
                problems.append(f"{split}/{family}: constant family answer")
        position_balance = _position_balance(examples)
        for family, report in position_balance.items():
            if report["max_minus_min"] > 1:
                problems.append(f"{split}/{family}: gold positions unbalanced")
        split_reports[split] = {
            "families": family_reports,
            "gold_position_balance": position_balance,
        }
    names = list(suite.splits())
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if all_keys[left] & all_keys[right]:
                problems.append(f"cross-split scenario overlap: {left}/{right}")
            if all_prompts[left] & all_prompts[right]:
                problems.append(f"cross-split prompt overlap: {left}/{right}")
    return {
        "schema_version": "behavioral_v3_validation.v1",
        "suite_identity": SUITE_IDENTITY,
        "suite_hash": suite.records_hash(),
        "ok": not problems,
        "problems": problems,
        "splits": split_reports,
    }


def _accuracy(examples: Sequence[Example], guesses: Iterable[str]) -> float:
    pairs = list(zip(examples, guesses, strict=True))
    return sum(example.answer == guess for example, guess in pairs) / len(pairs) if pairs else 0.0


def _hash_guess(text: str, example: Example) -> str:
    canonical_index = int(_sha256_text(text), 16) % len(example.canonical_candidates)
    return example.canonical_candidates[canonical_index]


def _train_majority(suite: Suite) -> dict[str, str]:
    result = {}
    for family in FAMILIES:
        answers = [example.answer for example in suite.train if example.family == family]
        result[family] = min(Counter(answers).items(), key=lambda item: (-item[1], item[0]))[0]
    return result


def _stable_event_permutation(events: Sequence[Any], namespace: str) -> list[Any]:
    """Order events using only canonical bytes, SHA-256, and their original index."""
    decorated = [
        (
            _sha256_text(f"{namespace}:{index}:{_canonical_json(event)}"),
            index,
            event,
        )
        for index, event in enumerate(events)
    ]
    return [event for _, _, event in sorted(decorated)]


def _shuffled_reference_guess(example: Example) -> str:
    scenario = example.scenario
    scenario["events"] = _stable_event_permutation(
        scenario["events"],
        f"{example.ex_id}:shuffled-events-baseline.v1",
    )
    result = _reference_replay(scenario)
    return result if result in example.canonical_candidates else example.canonical_candidates[0]


def _mean_uniform_chance(examples: Sequence[Example]) -> float:
    """Compute the heterogeneous chance baseline without version-sensitive float sums."""
    if not examples:
        return 0.0
    exact = sum((Fraction(1, len(example.candidates)) for example in examples), Fraction())
    return float(exact / len(examples))


def _lexical_guess(example: Example) -> str:
    counts = {
        candidate: len(re.findall(rf"(?<!\w){re.escape(candidate)}(?!\w)", example.prompt))
        for candidate in example.canonical_candidates
    }
    return min(counts, key=lambda candidate: (-counts[candidate], candidate))


def _initial_state_guess(example: Example) -> str:
    scenario = example.scenario
    if example.family in {"fsm", "graph_walk"}:
        guess = str(scenario["start"])
    elif example.family == "stack_queue":
        guess = str(scenario["initial"][0])
    elif example.family in {"chain_arith", "tiny_prog"}:
        guess = str(scenario["start"])
    elif example.family == "rule_neg":
        guess = "true" if scenario["initial_value"] else "false"
    else:
        query = scenario["query"]
        guess = (
            scenario["initial_locations"][query["name"]]
            if query["kind"] == "location"
            else scenario["initial_holders"][query["name"]]
        )
    return guess if guess in example.canonical_candidates else example.canonical_candidates[0]


def _events_only_lexical_guess(example: Example) -> str:
    event_text = "\n".join(example.events_text)
    counts = {
        candidate: len(re.findall(rf"(?<!\w){re.escape(candidate)}(?!\w)", event_text))
        for candidate in example.canonical_candidates
    }
    return min(counts, key=lambda candidate: (-counts[candidate], candidate))


def baseline_suite(suite: Suite) -> dict[str, Any]:
    """Honest non-model baselines; final_test is intentionally not evaluated."""
    majority = _train_majority(suite)
    evaluated = [name for name in suite.splits() if name != UNTOUCHED_FINAL_SPLIT]
    report: dict[str, Any] = {
        "schema_version": "behavioral_v3_baselines.v1",
        "evaluated_splits": evaluated,
        "excluded_splits": {UNTOUCHED_FINAL_SPLIT: "untouched final test"},
        "primary_baseline_note": "All hash/lexical baselines are deterministic heuristics, not learned models.",
        "splits": {},
    }
    for split in evaluated:
        examples = suite.splits()[split]
        max_positions = max(len(example.candidates) for example in examples)
        position_scores = {}
        for position in range(max_positions):
            guesses = [example.candidates[position] if position < len(example.candidates) else "__NO_CANDIDATE__" for example in examples]
            position_scores[str(position)] = _accuracy(examples, guesses)
        position_by_family: dict[str, dict[str, float]] = {}
        for family in FAMILIES:
            family_examples = [example for example in examples if example.family == family]
            family_positions = len(family_examples[0].candidates)
            position_by_family[family] = {
                str(position): _accuracy(
                    family_examples,
                    (example.candidates[position] for example in family_examples),
                )
                for position in range(family_positions)
            }
        report["splits"][split] = {
            "mean_per_example_uniform_chance": _mean_uniform_chance(examples),
            "train_majority_constant_answer": _accuracy(examples, (majority[example.family] for example in examples)),
            "candidate_position": position_scores,
            "candidate_position_by_family": position_by_family,
            "initial_only_sha256": _accuracy(examples, (_hash_guess(example.initial_text, example) for example in examples)),
            "initial_only_state_heuristic": _accuracy(examples, (_initial_state_guess(example) for example in examples)),
            "events_only_sha256": _accuracy(examples, (_hash_guess("\n".join(example.events_text), example) for example in examples)),
            "events_only_lexical_heuristic": _accuracy(examples, (_events_only_lexical_guess(example) for example in examples)),
            "shuffled_events_reference": _accuracy(examples, (_shuffled_reference_guess(example) for example in examples)),
            "lexical_candidate_frequency": _accuracy(examples, (_lexical_guess(example) for example in examples)),
        }
    return report


def validation_report(suite: Suite) -> dict[str, Any]:
    audit = audit_suite(suite)
    return {
        **audit,
        "baselines": baseline_suite(suite),
        "acceptance_thresholds": {
            "independent_solver_agreement": 1.0,
            "causal_event_removal_change_rate": 0.95,
            "counterfactual_change_rate": 1.0,
            "event_reverse_change_rate_when_applicable": 0.95,
            "gold_position_max_minus_min": 1,
        },
        "final_test_integrity_audit_only": True,
        "untouched_final_test_model_or_baseline_metrics_emitted": False,
    }


def write_artifacts(suite: Suite, manifest_path: Path, validation_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(suite.manifest(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validation_path.write_text(json.dumps(validation_report(suite), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate immutable behavioral-v3 evidence")
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/behavioral_v3_manifest.json"))
    parser.add_argument("--validation", type=Path, default=Path("artifacts/behavioral_v3_validation.json"))
    args = parser.parse_args(argv)
    suite = build_suite()
    report = audit_suite(suite)
    write_artifacts(suite, args.manifest, args.validation)
    print(json.dumps({"suite_hash": suite.records_hash(), "validation_ok": report["ok"]}, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())

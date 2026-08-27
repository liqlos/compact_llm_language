"""Behavioral-v3 textual baselines with independently rescorable evidence.

This module owns a generative evidence format, ``text_generation_eval.v1``.
It intentionally does not implement candidate log-probability ranking: textual
baselines are scored only by parsing their retained, complete generations.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import re
import resource
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_MODEL_ID = "Qwen/Qwen3.5-2B"
DEFAULT_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
DEFAULT_SUITE_HASH = (
    "5cf5cbf397510ba597b59f7ccf0839cf344e6fb795a5cb29d031f39dac218254"
)

TEXT_RECORD_SCHEMA = "text_generation_eval.v1"
TEXT_REPORT_SCHEMA = "text_generation_eval.report.v1"
PARSER_VERSION = "answer_marker_or_exact_last_line.v1"
PARSER_IMPLEMENTATION = "latent_lab.bench.text_baselines:derive_generation_verdict"

SYSTEM_DIRECT = (
    "Answer the question directly and exactly. End your reply with a final "
    "line 'Answer: X' where X is the answer only."
)
SYSTEM_THINK = (
    "You are a precise solver. Work out the answer step by step, then finish "
    "with a final line 'Answer: X' where X is the answer only."
)
SYSTEM_SCRATCH = (
    "Use a concise visible scratch calculation, then finish with a final line "
    "'Answer: X' where X is the answer only. Keep the entire reply within the "
    "small generation budget."
)
MODE_DESCRIPTIONS = {
    "A": "direct chat completion with native thinking disabled",
    "B": "native thinking chat completion",
    "C": "concise visible scratch with native thinking disabled",
}
DEFAULT_MAX_NEW_TOKENS = {"A": 64, "B": 512, "C": 64}
EVALUATION_SPLITS = (
    "validation",
    "test_id",
    "test_ood_length",
    "test_ood_semantic",
    "final_test",
)

ANSWER_RE = re.compile(r"^Answer:\s*(.+?)\s*$", re.IGNORECASE)
TERMINATION_TOKENS = ("<|im_end|>", "<|endoftext|>")
STRIPPED_SPECIAL_TOKENS = TERMINATION_TOKENS + ("<|im_start|>",)


class TextEvidenceError(ValueError):
    """A text-generation record cannot be independently verified."""


def _canonical_json(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise TextEvidenceError(f"value is not canonical JSON: {error}") from error
    return encoded.encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value))


def canonical(text: str) -> str:
    return text.strip().strip(".,;:!?\"'`").lower()


def strip_special(text: str) -> str:
    """Cut everything from the first decoded special token onward."""
    for token in STRIPPED_SPECIAL_TOKENS:
        index = text.find(token)
        if index != -1:
            text = text[:index]
    return text


def parse_answer(generated: str) -> tuple[str | None, str]:
    """Return the final explicit ``Answer:`` value and parser status."""
    cleaned = strip_special(generated)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return None, "no_answer_marker"
    match = ANSWER_RE.fullmatch(lines[-1])
    if match is None:
        return None, "no_answer_marker"
    return canonical(match.group(1)), "ok"


def _termination_from_raw(
    generated_text: str,
    generated_token_ids: Sequence[int],
    termination_token_ids: Sequence[int],
) -> bool:
    # The full decoded generation is the independently rescorable raw output.
    # Token ids remain retained provenance, but cannot assert termination
    # without reloading the pinned tokenizer. A decoded terminal marker must be
    # present at the end; appending a forged EOS id cannot change the verdict.
    del generated_token_ids, termination_token_ids
    return any(
        (index := generated_text.find(token)) >= 0
        and not generated_text[index + len(token):].strip()
        for token in TERMINATION_TOKENS
    )


def derive_generation_verdict(
    *,
    generated_text: str,
    generated_token_ids: Sequence[int],
    termination_token_ids: Sequence[int],
    candidates: Sequence[str],
    gold_answer: str,
    error_status: str | None,
) -> dict[str, Any]:
    """Derive the complete textual verdict from retained raw fields only."""
    terminated = _termination_from_raw(
        generated_text, generated_token_ids, termination_token_ids
    )
    if error_status is not None:
        return {
            "terminated": terminated,
            "termination_status": "GENERATION_ERROR",
            "parser_status": "generation_error",
            "parsed_answer": None,
            "correctness": False,
        }

    parsed, parser_status = parse_answer(generated_text)
    if parsed is None:
        cleaned = strip_special(generated_text)
        tail = [line for line in cleaned.strip().splitlines() if line.strip()]
        canonical_candidates = [canonical(candidate) for candidate in candidates]
        if tail and canonical(tail[-1]) in canonical_candidates:
            parsed = canonical(tail[-1])
            parser_status = "last_line"

    correctness = bool(
        terminated
        and parsed is not None
        and parsed == canonical(gold_answer)
    )
    return {
        "terminated": terminated,
        "termination_status": "TERMINATED" if terminated else "NON_TERMINATION",
        "parser_status": parser_status,
        "parsed_answer": parsed,
        "correctness": correctness,
    }


@lru_cache(maxsize=1)
def parser_identity() -> dict[str, str]:
    source = "\n".join(
        inspect.getsource(function)
        for function in (
            canonical,
            strip_special,
            parse_answer,
            _termination_from_raw,
            derive_generation_verdict,
        )
    )
    implementation_material = {
        "answer_pattern": ANSWER_RE.pattern,
        "implementation": PARSER_IMPLEMENTATION,
        "source": source,
        "stripped_special_tokens": list(STRIPPED_SPECIAL_TOKENS),
        "termination_tokens": list(TERMINATION_TOKENS),
        "version": PARSER_VERSION,
    }
    return {
        "implementation": PARSER_IMPLEMENTATION,
        "version": PARSER_VERSION,
        "sha256": _sha256_json(implementation_material),
    }


@lru_cache(maxsize=1)
def producer_identity() -> dict[str, str]:
    return {
        "implementation": "latent_lab.bench.text_baselines",
        "version": "text_generation_producer.v1",
        "source_sha256": _sha256_bytes(Path(__file__).read_bytes()),
    }


def _execution_context_from_env() -> dict[str, Any] | None:
    names = {
        "plan_hash": "RCC_R1_ACTIVE_PLAN_HASH",
        "command_index": "RCC_R1_ACTIVE_COMMAND_INDEX",
        "driver_source_sha256": "RCC_R1_DRIVER_SOURCE_SHA256",
        "producer_source_sha256": "RCC_R1_TEXT_PRODUCER_SOURCE_SHA256",
        "suite_hash": "RCC_R1_DRIVER_SUITE_HASH",
    }
    values = {field: os.environ.get(name) for field, name in names.items()}
    if all(value is None for value in values.values()):
        return None
    if any(value is None for value in values.values()):
        raise TextEvidenceError("partial R1 driver execution context")
    try:
        command_index = int(values["command_index"])
    except (TypeError, ValueError) as error:
        raise TextEvidenceError("invalid R1 driver command index") from error
    return {
        "plan_hash": values["plan_hash"],
        "command_index": command_index,
        "driver_source_sha256": values["driver_source_sha256"],
        "producer_source_sha256": values["producer_source_sha256"],
        "suite_hash": values["suite_hash"],
    }


def _valid_execution_context(
    context: Any, *, suite_hash: str,
) -> bool:
    return bool(
        isinstance(context, Mapping)
        and set(context) == {
            "plan_hash", "command_index", "driver_source_sha256",
            "producer_source_sha256", "suite_hash",
        }
        and isinstance(context.get("plan_hash"), str)
        and re.fullmatch(r"[0-9a-f]{64}", context["plan_hash"]) is not None
        and isinstance(context.get("driver_source_sha256"), str)
        and re.fullmatch(
            r"[0-9a-f]{64}", context["driver_source_sha256"]
        ) is not None
        and context.get("producer_source_sha256") == producer_identity()[
            "source_sha256"]
        and isinstance(context.get("command_index"), int)
        and not isinstance(context["command_index"], bool)
        and context["command_index"] >= 0
        and context.get("suite_hash") == suite_hash
    )


def score_example(generated: str, ex: Any) -> dict[str, Any]:
    """Compatibility wrapper used by parser regressions, never evidence I/O."""
    verdict = derive_generation_verdict(
        generated_text=generated,
        generated_token_ids=(),
        termination_token_ids=(),
        candidates=ex.candidates,
        gold_answer=ex.answer,
        error_status=None,
    )
    return {
        "status": verdict["termination_status"]
        if not verdict["terminated"]
        else verdict["parser_status"],
        "correct": float(verdict["correctness"]),
    }


def _mode_prompt_config(mode: str) -> tuple[str, bool]:
    if mode == "A":
        return SYSTEM_DIRECT, False
    elif mode == "B":
        return SYSTEM_THINK, True
    elif mode == "C":
        return SYSTEM_SCRATCH, False
    raise ValueError(f"unknown text baseline mode: {mode!r}")


def prompt_messages(ex: Any, mode: str) -> list[dict[str, str]]:
    system, _thinking = _mode_prompt_config(mode)
    user = ex.prompt.rsplit("Answer:", 1)[0].strip()
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_prompt(tok: Any, ex: Any, mode: str):
    """Return model input ids while preserving the historical helper API."""
    messages = prompt_messages(ex, mode)
    _system, thinking = _mode_prompt_config(mode)
    rendered = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=thinking,
    )
    return rendered.input_ids


def _eos_ids(model: Any, tok: Any) -> tuple[int, ...]:
    value = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if value is None:
        value = getattr(tok, "eos_token_id", None)
    if value is None:
        return ()
    if isinstance(value, int) and not isinstance(value, bool):
        return (value,)
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        return tuple(value)
    raise TextEvidenceError(f"invalid eos_token_id: {value!r}")


def run_batched(model: Any, tok: Any, prompts_ids: Sequence[Any], max_new: int,
                device: str) -> list[dict[str, Any]]:
    """Left-padded greedy generation retaining every generated token id."""
    import torch

    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = tok.eos_token_id
    if pad_id is None:
        raise TextEvidenceError("tokenizer has neither pad_token_id nor eos_token_id")
    eos_ids = _eos_ids(model, tok)
    max_length = max(value.shape[1] for value in prompts_ids)
    input_rows, masks = [], []
    for value in prompts_ids:
        padding = max_length - value.shape[1]
        input_rows.append(torch.cat([
            torch.full((1, padding), pad_id, dtype=value.dtype), value,
        ], dim=1))
        masks.append(torch.cat([
            torch.zeros((1, padding), dtype=torch.long),
            torch.ones((1, value.shape[1]), dtype=torch.long),
        ], dim=1))
    input_ids = torch.cat(input_rows).to(device)
    attention_mask = torch.cat(masks).to(device)
    with torch.no_grad():
        sequences = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=pad_id,
        )

    outputs = []
    eos_set = set(eos_ids)
    for row in sequences[:, input_ids.shape[1]:]:
        token_ids = [int(token) for token in row.tolist()]
        first_eos = next(
            (index for index, token in enumerate(token_ids) if token in eos_set),
            None,
        )
        if first_eos is not None:
            token_ids = token_ids[:first_eos + 1]
        elif pad_id not in eos_set:
            while token_ids and token_ids[-1] == pad_id:
                token_ids.pop()
        outputs.append({
            "generated_text": tok.decode(token_ids, skip_special_tokens=False),
            "generated_token_ids": token_ids,
        })
    return outputs


def _int_list(value: Sequence[int], field: str) -> list[int]:
    result = list(value)
    if not all(isinstance(item, int) and not isinstance(item, bool)
               and item >= 0 for item in result):
        raise TextEvidenceError(f"{field} must contain non-negative integers")
    return result


def derive_gold_index(candidates: Sequence[str], gold_answer: str) -> int:
    matches = [index for index, candidate in enumerate(candidates)
               if candidate == gold_answer]
    if len(matches) != 1:
        raise TextEvidenceError(
            "gold answer must occur exactly once in actual candidate order"
        )
    return matches[0]


@lru_cache(maxsize=1)
def _canonical_suite() -> Any:
    from latent_lab.bench.suite_v3 import build_suite

    return build_suite()


def _validate_suite_binding(record: Mapping[str, Any], suite: Any) -> Any:
    manifest = suite.manifest()
    for field, expected in (
        ("suite_identity", manifest["suite_identity"]),
        ("suite_version", manifest["suite_version"]),
        ("suite_hash", manifest["suite_hash"]),
    ):
        if record.get(field) != expected:
            raise TextEvidenceError(f"record {field} is not canonical behavioral-v3")
    split = record.get("split")
    if split not in EVALUATION_SPLITS:
        raise TextEvidenceError("record split is not an evaluation split")
    examples = list(getattr(suite, split))
    matches = [example for example in examples
               if example.ex_id == record.get("example_id")]
    if len(matches) != 1:
        raise TextEvidenceError("example_id is not unique in the declared suite split")
    example = matches[0]
    expected_fields = {
        "family": example.family,
        "depth": example.depth,
        "template": example.template,
        "benchmark_prompt": example.prompt,
        "candidates_in_actual_order": list(example.candidates),
        "canonical_candidates": list(example.canonical_candidates),
        "candidate_permutation_seed": example.candidate_permutation_seed,
        "candidate_permutation": list(example.candidate_permutation),
        "gold_answer": example.answer,
        "derived_gold_index": example.gold_index,
    }
    for field, expected in expected_fields.items():
        if record.get(field) != expected:
            raise TextEvidenceError(
                f"record {field} disagrees with canonical suite example"
            )
    return example


def _record_hash(record: Mapping[str, Any]) -> str:
    payload = dict(record)
    payload.pop("record_hash", None)
    return _sha256_json(payload)


def make_text_record(
    *,
    run_id: str,
    suite_manifest: Mapping[str, Any],
    ex: Any,
    mode: str,
    seed: int,
    model_identity: Mapping[str, Any],
    tokenizer_identity: Mapping[str, Any],
    messages: Sequence[Mapping[str, str]],
    input_ids: Sequence[int],
    candidate_token_ids: Sequence[Sequence[int]],
    generated_text: str,
    generated_token_ids: Sequence[int],
    max_new_tokens: int,
    error_status: str | None,
    compute: Mapping[str, Any],
    execution_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    candidates = list(ex.candidates)
    input_id_list = _int_list(input_ids, "input_ids")
    generated_id_list = _int_list(generated_token_ids, "generated_token_ids")
    candidate_id_lists = [
        _int_list(ids, "candidate_token_ids") for ids in candidate_token_ids
    ]
    if len(candidate_id_lists) != len(candidates):
        raise TextEvidenceError("candidate token ids must align with candidates")
    gold_index = derive_gold_index(candidates, ex.answer)
    if gold_index != ex.gold_index:
        raise TextEvidenceError("suite gold_index disagrees with actual candidate order")
    eos_ids = _int_list(tokenizer_identity.get("eos_token_ids", ()),
                        "tokenizer_identity.eos_token_ids")
    verdict = derive_generation_verdict(
        generated_text=generated_text,
        generated_token_ids=generated_id_list,
        termination_token_ids=eos_ids,
        candidates=candidates,
        gold_answer=ex.answer,
        error_status=error_status,
    )
    prompt_payload = [dict(message) for message in messages]
    _system_prompt, enable_thinking = _mode_prompt_config(mode)
    record = {
        "schema_version": TEXT_RECORD_SCHEMA,
        "run_id": run_id,
        "suite_identity": suite_manifest["suite_identity"],
        "suite_version": suite_manifest["suite_version"],
        "suite_hash": suite_manifest["suite_hash"],
        "example_id": ex.ex_id,
        "split": ex.split,
        "family": ex.family,
        "depth": ex.depth,
        "template": ex.template,
        "mode": mode,
        "mode_description": MODE_DESCRIPTIONS[mode],
        "generation_config": {
            "do_sample": False,
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            "enable_thinking": enable_thinking,
        },
        "seed": seed,
        "model_identity": dict(model_identity),
        "tokenizer_identity": dict(tokenizer_identity),
        "parser": parser_identity(),
        "producer": producer_identity(),
        "execution_context": (
            dict(execution_context) if execution_context is not None else None
        ),
        "benchmark_prompt": ex.prompt,
        "benchmark_prompt_sha256": _sha256_text(ex.prompt),
        "prompt_messages": prompt_payload,
        "prompt_sha256": _sha256_json(prompt_payload),
        "input_ids": input_id_list,
        "input_ids_sha256": _sha256_json(input_id_list),
        "input_token_count": len(input_id_list),
        "candidates_in_actual_order": candidates,
        "canonical_candidates": list(ex.canonical_candidates),
        "candidate_permutation_seed": ex.candidate_permutation_seed,
        "candidate_permutation": list(ex.candidate_permutation),
        "candidate_token_ids": candidate_id_lists,
        "candidate_token_counts": [len(ids) for ids in candidate_id_lists],
        "gold_answer": ex.answer,
        "derived_gold_index": gold_index,
        "generated_text": generated_text,
        "generated_text_sha256": _sha256_text(generated_text),
        "generated_token_ids": generated_id_list,
        "generated_token_ids_sha256": _sha256_json(generated_id_list),
        "generated_token_count": len(generated_id_list),
        "termination_evidence_source": "decoded_text_terminal_marker_v1",
        "error_status": error_status,
        "terminated": verdict["terminated"],
        "termination_status": verdict["termination_status"],
        "parser_status": verdict["parser_status"],
        "parsed_answer": verdict["parsed_answer"],
        "correctness": verdict["correctness"],
        "compute": dict(compute),
    }
    record["record_hash"] = _record_hash(record)
    return record


_RAW_REQUIRED_FIELDS = frozenset({
    "benchmark_prompt",
    "benchmark_prompt_sha256",
    "generated_text",
    "generated_text_sha256",
    "generated_token_ids",
    "generated_token_ids_sha256",
    "generated_token_count",
    "input_ids",
    "input_ids_sha256",
    "input_token_count",
    "prompt_messages",
    "prompt_sha256",
})


def offline_rescore_record(
    record: Mapping[str, Any], *, suite: Any | None = None,
) -> dict[str, Any]:
    """Validate raw evidence and reproduce the stored verdict exactly."""
    if not isinstance(record, Mapping):
        raise TextEvidenceError("record must be an object")
    missing = sorted(_RAW_REQUIRED_FIELDS - set(record))
    if missing:
        raise TextEvidenceError(f"missing raw evidence fields: {missing}")
    if record.get("schema_version") != TEXT_RECORD_SCHEMA:
        raise TextEvidenceError("unsupported text evidence schema")
    if record.get("parser") != parser_identity():
        raise TextEvidenceError("parser identity/hash does not match implementation")
    if record.get("producer") != producer_identity():
        raise TextEvidenceError("producer source identity does not match implementation")
    if record.get("record_hash") != _record_hash(record):
        raise TextEvidenceError("record hash mismatch")
    suite = _canonical_suite() if suite is None else suite
    example = _validate_suite_binding(record, suite)
    execution_context = record.get("execution_context")
    if execution_context is not None and not _valid_execution_context(
            execution_context, suite_hash=record["suite_hash"]):
        raise TextEvidenceError("invalid R1 driver execution context")
    model_identity = record.get("model_identity")
    tokenizer_identity = record.get("tokenizer_identity")
    if not isinstance(model_identity, Mapping) or not isinstance(
            tokenizer_identity, Mapping):
        raise TextEvidenceError("model/tokenizer identity must be objects")
    for identity, field in (
        (model_identity, "revision"),
        (tokenizer_identity, "revision"),
    ):
        value = identity.get(field)
        if (not isinstance(value, str) or len(value) != 40
                or re.fullmatch(r"[0-9a-f]{40}", value) is None):
            raise TextEvidenceError("model/tokenizer revision must be a pinned commit")
    if model_identity["revision"] != tokenizer_identity["revision"]:
        raise TextEvidenceError("model/tokenizer revisions disagree")
    resolved_commit = tokenizer_identity.get("resolved_commit")
    if resolved_commit is not None and resolved_commit != tokenizer_identity[
            "revision"]:
        raise TextEvidenceError("tokenizer resolved commit disagrees with revision")
    chat_template_hash = tokenizer_identity.get("chat_template_sha256")
    if (not isinstance(chat_template_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", chat_template_hash) is None):
        raise TextEvidenceError("tokenizer chat-template hash is invalid")
    if not isinstance(model_identity.get("model_id"), str) \
            or not model_identity["model_id"]:
        raise TextEvidenceError("model identity is missing model_id")
    mode = record.get("mode")
    if mode not in MODE_DESCRIPTIONS:
        raise TextEvidenceError("unknown text baseline mode")
    if record.get("mode_description") != MODE_DESCRIPTIONS[mode]:
        raise TextEvidenceError("mode description disagrees with mode")
    _system_prompt, enable_thinking = _mode_prompt_config(mode)
    generation_config = record.get("generation_config")
    if generation_config != {
        "do_sample": False,
        "temperature": 0,
        "max_new_tokens": generation_config.get("max_new_tokens")
        if isinstance(generation_config, Mapping) else None,
        "enable_thinking": enable_thinking,
    }:
        raise TextEvidenceError("generation config disagrees with text mode")
    max_new_tokens = generation_config["max_new_tokens"]
    if (not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool)
            or max_new_tokens <= 0):
        raise TextEvidenceError("max_new_tokens must be a positive integer")

    if not isinstance(record["benchmark_prompt"], str):
        raise TextEvidenceError("benchmark_prompt must be a string")
    if record["benchmark_prompt_sha256"] != _sha256_text(
            record["benchmark_prompt"]):
        raise TextEvidenceError("benchmark prompt hash mismatch")
    prompt_messages_value = record["prompt_messages"]
    if (not isinstance(prompt_messages_value, list)
            or not all(isinstance(message, dict) for message
                       in prompt_messages_value)):
        raise TextEvidenceError("prompt_messages must be a list of objects")
    if record["prompt_sha256"] != _sha256_json(prompt_messages_value):
        raise TextEvidenceError("prompt messages hash mismatch")
    if prompt_messages_value != prompt_messages(example, mode):
        raise TextEvidenceError("prompt messages disagree with suite example/mode")

    input_ids = _int_list(record["input_ids"], "input_ids")
    generated_ids = _int_list(record["generated_token_ids"],
                              "generated_token_ids")
    if record["input_ids_sha256"] != _sha256_json(input_ids):
        raise TextEvidenceError("input_ids hash mismatch")
    if record["input_token_count"] != len(input_ids):
        raise TextEvidenceError("input token count mismatch")
    if not isinstance(record["generated_text"], str):
        raise TextEvidenceError("generated_text must be a string")
    if record["generated_text_sha256"] != _sha256_text(record["generated_text"]):
        raise TextEvidenceError("generated_text hash mismatch")
    if record["generated_token_ids_sha256"] != _sha256_json(generated_ids):
        raise TextEvidenceError("generated_token_ids hash mismatch")
    if record["generated_token_count"] != len(generated_ids):
        raise TextEvidenceError("generated token count mismatch")
    if record.get("termination_evidence_source") != (
            "decoded_text_terminal_marker_v1"):
        raise TextEvidenceError("unsupported termination evidence source")
    if len(generated_ids) > max_new_tokens:
        raise TextEvidenceError("generated token count exceeds generation budget")

    candidates = record.get("candidates_in_actual_order")
    if (not isinstance(candidates, list) or not candidates
            or not all(isinstance(candidate, str) for candidate in candidates)
            or len(set(candidates)) != len(candidates)):
        raise TextEvidenceError("actual candidates must be unique strings")
    gold_answer = record.get("gold_answer")
    if not isinstance(gold_answer, str):
        raise TextEvidenceError("gold_answer must be a string")
    if record.get("derived_gold_index") != derive_gold_index(candidates, gold_answer):
        raise TextEvidenceError("derived gold index mismatch")
    permutation = record.get("candidate_permutation")
    if (not isinstance(permutation, list)
            or sorted(permutation) != list(range(len(candidates)))):
        raise TextEvidenceError("candidate permutation is corrupt")
    if not isinstance(record.get("candidate_permutation_seed"), int):
        raise TextEvidenceError("candidate permutation seed must be an integer")
    canonical_candidates = record.get("canonical_candidates")
    if (not isinstance(canonical_candidates, list)
            or len(canonical_candidates) != len(candidates)
            or not all(isinstance(candidate, str)
                       for candidate in canonical_candidates)):
        raise TextEvidenceError("canonical candidates are corrupt")
    if [canonical_candidates[index] for index in permutation] != candidates:
        raise TextEvidenceError("candidate order disagrees with permutation")

    candidate_token_ids = record.get("candidate_token_ids")
    if not isinstance(candidate_token_ids, list):
        raise TextEvidenceError("candidate token ids must be a list")
    candidate_ids = [
        _int_list(ids, "candidate_token_ids") for ids in candidate_token_ids
    ]
    if len(candidate_ids) != len(candidates):
        raise TextEvidenceError("candidate token ids do not align")
    if record.get("candidate_token_counts") != [len(ids) for ids in candidate_ids]:
        raise TextEvidenceError("candidate token counts mismatch")

    eos_ids = _int_list(tokenizer_identity.get("eos_token_ids", ()),
                        "tokenizer_identity.eos_token_ids")
    if not eos_ids:
        raise TextEvidenceError(
            "current text evidence requires tokenizer termination token ids"
        )
    error_status = record.get("error_status")
    if error_status is not None and (
        not isinstance(error_status, str) or not error_status
    ):
        raise TextEvidenceError("error_status must be null or a non-empty string")
    expected = derive_generation_verdict(
        generated_text=record["generated_text"],
        generated_token_ids=generated_ids,
        termination_token_ids=eos_ids,
        candidates=candidates,
        gold_answer=gold_answer,
        error_status=error_status,
    )
    stored = {key: record.get(key) for key in expected}
    if stored != expected:
        raise TextEvidenceError(
            f"stored verdict disagrees with raw rescore: {stored!r} != {expected!r}"
        )
    compute = record.get("compute")
    if not isinstance(compute, Mapping):
        raise TextEvidenceError("compute must be an object")
    for field in (
        "batch_generate_wall_seconds", "allocated_generate_wall_seconds",
    ):
        value = compute.get(field)
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or value < 0):
            raise TextEvidenceError(
                f"compute {field} must be finite and non-negative"
            )
    for field, minimum in (
        ("batch_index", 0), ("batch_size", 1),
        ("prefill_tokens", 0), ("generated_tokens", 0),
        ("candidate_tokenizer_calls", 0), ("prompt_tokenizer_calls", 0),
        ("decode_calls", 0),
    ):
        value = compute.get(field)
        if (not isinstance(value, int) or isinstance(value, bool)
                or value < minimum):
            raise TextEvidenceError(
                f"compute {field} must be an integer >= {minimum}"
            )
    if not math.isclose(
        compute["allocated_generate_wall_seconds"] * compute["batch_size"],
        compute["batch_generate_wall_seconds"],
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise TextEvidenceError("allocated and batch generation time disagree")
    if compute["prefill_tokens"] != len(input_ids):
        raise TextEvidenceError("compute prefill token count mismatch")
    if compute["generated_tokens"] != len(generated_ids):
        raise TextEvidenceError("compute generated token count mismatch")
    if compute["candidate_tokenizer_calls"] != len(candidates):
        raise TextEvidenceError("compute candidate tokenizer call count mismatch")
    if compute["prompt_tokenizer_calls"] != 1 or compute["decode_calls"] != 1:
        raise TextEvidenceError("text record requires one prompt tokenize/decode call")
    if compute.get("successful_task") is not (error_status is None):
        raise TextEvidenceError("compute successful_task disagrees with error status")
    return expected


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(records)
    correct = sum(bool(record["correctness"]) for record in records)
    nontermination = sum(
        record["termination_status"] == "NON_TERMINATION" for record in records
    )
    errors = sum(record["error_status"] is not None for record in records)
    successful = count - errors
    return {
        "record_count": count,
        "exact_match_count": correct,
        "exact_match_rate": correct / count if count else 0.0,
        "nontermination_count": nontermination,
        "nontermination_rate": nontermination / count if count else 0.0,
        "error_count": errors,
        "error_rate": errors / count if count else 0.0,
        "successful_task_count": successful,
        "mean_generated_tokens": (
            sum(record["generated_token_count"] for record in records) / count
            if count else 0.0
        ),
    }


def summarize_by_family(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for family in sorted({str(record["family"]) for record in records}):
        subset = [record for record in records if record["family"] == family]
        result[family] = summarize_records(subset)
    return result


def _report_hash(report: Mapping[str, Any]) -> str:
    payload = dict(report)
    payload.pop("report_hash", None)
    return _sha256_json(payload)


def _coverage_for_records(
    records: Sequence[Mapping[str, Any]], split: str, suite: Any,
) -> dict[str, Any]:
    expected_ids = [example.ex_id for example in getattr(suite, split)]
    observed_ids = [record.get("example_id") for record in records]
    if observed_ids == expected_ids:
        status = "FULL_SPLIT"
    elif observed_ids and observed_ids == expected_ids[:len(observed_ids)]:
        status = "SMOKE_PREFIX_ONLY"
    else:
        raise TextEvidenceError(
            "records must be the complete split or a deterministic smoke prefix"
        )
    return {
        "status": status,
        "record_count": len(observed_ids),
        "full_split_record_count": len(expected_ids),
        "ordered_example_ids_sha256": _sha256_json(observed_ids),
        "full_split_example_ids_sha256": _sha256_json(expected_ids),
    }


def _r1_report_eligible(
    *, coverage: Mapping[str, Any], mode: str, max_new_tokens: int,
    model_identity: Mapping[str, Any], tokenizer_identity: Mapping[str, Any],
    seed: int,
    execution_context: Mapping[str, Any] | None,
    suite_hash: str,
) -> bool:
    return bool(
        coverage.get("status") == "FULL_SPLIT"
        and mode in {"A", "C"}
        and max_new_tokens == 64
        and model_identity.get("model_id") == DEFAULT_MODEL_ID
        and model_identity.get("revision") == DEFAULT_REVISION
        and model_identity.get("device") == "cuda"
        and model_identity.get("dtype") == "torch.bfloat16"
        and tokenizer_identity.get("tokenizer_id") == DEFAULT_MODEL_ID
        and tokenizer_identity.get("revision") == DEFAULT_REVISION
        and tokenizer_identity.get("resolved_commit") == DEFAULT_REVISION
        and isinstance(tokenizer_identity.get("chat_template_sha256"), str)
        and re.fullmatch(
            r"[0-9a-f]{64}", tokenizer_identity["chat_template_sha256"]
        ) is not None
        and seed in {0, 1, 2}
        and suite_hash == DEFAULT_SUITE_HASH
        and _valid_execution_context(
            execution_context, suite_hash=suite_hash
        )
    )


def make_text_report(
    *,
    run_id: str,
    suite_manifest: Mapping[str, Any],
    mode: str,
    split: str,
    seed: int,
    model_identity: Mapping[str, Any],
    tokenizer_identity: Mapping[str, Any],
    max_new_tokens: int,
    records: Sequence[Mapping[str, Any]],
    wall_seconds: float,
    peak_memory_bytes: int,
    model_generate_calls: int,
    execution_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record_list = [dict(record) for record in records]
    suite = _canonical_suite()
    coverage = _coverage_for_records(record_list, split, suite)
    r1_structure_valid = _r1_report_eligible(
        coverage=coverage,
        mode=mode,
        max_new_tokens=max_new_tokens,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        seed=seed,
        execution_context=execution_context,
        suite_hash=suite_manifest["suite_hash"],
    )
    report = {
        "schema_version": TEXT_REPORT_SCHEMA,
        "record_schema_version": TEXT_RECORD_SCHEMA,
        "run_id": run_id,
        "suite_identity": suite_manifest["suite_identity"],
        "suite_version": suite_manifest["suite_version"],
        "suite_hash": suite_manifest["suite_hash"],
        "split": split,
        "selection_eligible": False,
        "validation_role": (
            "comparison_only_no_checkpoint_selection"
            if split == "validation" else None
        ),
        "coverage": coverage,
        "r1_preregistered_structure_valid": r1_structure_valid,
        "r1_preregistered_evidence_eligible": False,
        "requires_driver_receipt_attestation": True,
        "mode": mode,
        "mode_description": MODE_DESCRIPTIONS[mode],
        "seed": seed,
        "model_identity": dict(model_identity),
        "tokenizer_identity": dict(tokenizer_identity),
        "parser": parser_identity(),
        "producer": producer_identity(),
        "execution_context": (
            dict(execution_context) if execution_context is not None else None
        ),
        "sampling": {
            "do_sample": False,
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
        },
        "platform": platform.platform(),
        "derived_metrics": summarize_records(record_list),
        "by_family": summarize_by_family(record_list),
        "compute": {
            "wall_seconds": wall_seconds,
            "peak_memory_bytes": peak_memory_bytes,
            "model_generate_calls": model_generate_calls,
            "successful_task_denominator": sum(
                record["error_status"] is None for record in record_list
            ),
        },
        "records": record_list,
    }
    report["report_hash"] = _report_hash(report)
    return report


def offline_rescore_report(report: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(report, Mapping):
        raise TextEvidenceError("report must be an object")
    if report.get("schema_version") != TEXT_REPORT_SCHEMA:
        raise TextEvidenceError("unsupported text report schema")
    if report.get("record_schema_version") != TEXT_RECORD_SCHEMA:
        raise TextEvidenceError("report record schema mismatch")
    if report.get("report_hash") != _report_hash(report):
        raise TextEvidenceError("report hash mismatch")
    if report.get("producer") != producer_identity():
        raise TextEvidenceError("report producer source identity mismatch")
    execution_context = report.get("execution_context")
    if execution_context is not None and not _valid_execution_context(
            execution_context, suite_hash=report.get("suite_hash")):
        raise TextEvidenceError("report execution context is invalid")
    if report.get("selection_eligible") is not False:
        raise TextEvidenceError(
            "fixed text baselines are never checkpoint-selection eligible"
        )
    expected_validation_role = (
        "comparison_only_no_checkpoint_selection"
        if report.get("split") == "validation" else None
    )
    if report.get("validation_role") != expected_validation_role:
        raise TextEvidenceError("text baseline validation role is misleading")
    records = report.get("records")
    if not isinstance(records, list) or not records:
        raise TextEvidenceError("report records must be a non-empty list")
    suite = _canonical_suite()
    for record in records:
        offline_rescore_record(record, suite=suite)
        for field in (
            "run_id", "suite_identity", "suite_version", "suite_hash",
            "split", "mode", "seed", "model_identity", "tokenizer_identity",
            "parser",
            "producer", "execution_context",
        ):
            if record.get(field) != report.get(field):
                raise TextEvidenceError(f"record/report {field} mismatch")
    coverage = _coverage_for_records(records, report.get("split"), suite)
    if report.get("coverage") != coverage:
        raise TextEvidenceError("report coverage disagrees with canonical split")
    sampling = report.get("sampling")
    if (not isinstance(sampling, Mapping)
            or sampling.get("do_sample") is not False
            or sampling.get("temperature") != 0
            or not isinstance(sampling.get("max_new_tokens"), int)
            or isinstance(sampling.get("max_new_tokens"), bool)
            or sampling["max_new_tokens"] <= 0):
        raise TextEvidenceError("report sampling config is invalid")
    for record in records:
        if record.get("generation_config", {}).get("max_new_tokens") != sampling[
                "max_new_tokens"]:
            raise TextEvidenceError("record/report generation budget mismatch")
    model_identity = report.get("model_identity")
    tokenizer_identity = report.get("tokenizer_identity")
    if not isinstance(model_identity, Mapping) or not isinstance(
            tokenizer_identity, Mapping):
        raise TextEvidenceError("report model/tokenizer identity is missing")
    r1_structure_valid = _r1_report_eligible(
        coverage=coverage,
        mode=report.get("mode"),
        max_new_tokens=sampling["max_new_tokens"],
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        seed=report.get("seed"),
        execution_context=execution_context,
        suite_hash=report.get("suite_hash"),
    )
    if report.get("r1_preregistered_structure_valid") is not r1_structure_valid:
        raise TextEvidenceError("R1 structural eligibility claim is incorrect")
    if report.get("r1_preregistered_evidence_eligible") is not False:
        raise TextEvidenceError(
            "standalone text reports cannot claim receipt-attested eligibility"
        )
    if report.get("requires_driver_receipt_attestation") is not True:
        raise TextEvidenceError("driver receipt requirement is missing")
    metrics = summarize_records(records)
    by_family = summarize_by_family(records)
    if report.get("derived_metrics") != metrics:
        raise TextEvidenceError("report metrics disagree with raw rescore")
    if report.get("by_family") != by_family:
        raise TextEvidenceError("family metrics disagree with raw rescore")
    compute = report.get("compute")
    if not isinstance(compute, Mapping):
        raise TextEvidenceError("report compute must be an object")
    wall = compute.get("wall_seconds")
    peak = compute.get("peak_memory_bytes")
    calls = compute.get("model_generate_calls")
    if (not isinstance(wall, (int, float)) or isinstance(wall, bool)
            or not math.isfinite(wall) or wall < 0):
        raise TextEvidenceError("report wall time must be finite and non-negative")
    if not isinstance(peak, int) or isinstance(peak, bool) or peak < 0:
        raise TextEvidenceError("report peak memory must be non-negative integer")
    if not isinstance(calls, int) or isinstance(calls, bool) or calls < 1:
        raise TextEvidenceError("report model_generate_calls must be positive integer")
    if compute.get("successful_task_denominator") != metrics[
            "successful_task_count"]:
        raise TextEvidenceError("report successful-task denominator mismatch")
    batch_indexes = [record["compute"]["batch_index"] for record in records]
    if sorted(set(batch_indexes)) != list(range(calls)):
        raise TextEvidenceError("report batch indexes disagree with generate calls")
    allocated = sum(
        record["compute"]["allocated_generate_wall_seconds"]
        for record in records
    )
    if allocated > wall + 1e-9:
        raise TextEvidenceError("allocated generation time exceeds report wall time")
    return {
        "schema_version": "text_generation_rescore.v1",
        "run_id": report.get("run_id"),
        "mode": report.get("mode"),
        "seed": report.get("seed"),
        "split": report.get("split"),
        "report_hash": report.get("report_hash"),
        "derived_metrics": metrics,
        "by_family": by_family,
        "coverage": coverage,
        "r1_preregistered_structure_valid": r1_structure_valid,
        "r1_preregistered_evidence_eligible": False,
        "requires_driver_receipt_attestation": True,
        "execution_context": execution_context,
        "status": (
            "VALID_FULL_RAW_EVIDENCE_PENDING_DRIVER_RECEIPT"
            if r1_structure_valid
            else "VALID_SMOKE_ONLY_NOT_HEADLINE_ELIGIBLE"
            if coverage["status"] == "SMOKE_PREFIX_ONLY"
            else "VALID_NON_PREREGISTERED_NOT_HEADLINE_ELIGIBLE"
        ),
    }


def _strict_json_loads(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise TextEvidenceError(f"non-finite JSON constant: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise TextEvidenceError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except json.JSONDecodeError as error:
        raise TextEvidenceError(f"invalid JSON: {error}") from error


def load_and_rescore(path: Path) -> dict[str, Any]:
    try:
        payload = _strict_json_loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise TextEvidenceError(f"cannot read text evidence {path}: {error}") from error
    return offline_rescore_report(payload)


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(_canonical_json(report) + b"\n")
    os.replace(temporary, path)


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _tokenizer_identity(tok: Any, model: Any, revision: str) -> dict[str, Any]:
    model_max_length = getattr(tok, "model_max_length", None)
    if not isinstance(model_max_length, int) or isinstance(model_max_length, bool):
        model_max_length = None
    return {
        "tokenizer_id": str(getattr(tok, "name_or_path", "unknown")),
        "tokenizer_class": type(tok).__name__,
        "revision": revision,
        "resolved_commit": getattr(tok, "init_kwargs", {}).get("_commit_hash"),
        "vocab_size": len(tok),
        "model_max_length": model_max_length,
        "pad_token_id": getattr(tok, "pad_token_id", None),
        "eos_token_ids": list(_eos_ids(model, tok)),
        "chat_template_sha256": _sha256_text(str(getattr(tok, "chat_template", ""))),
    }


def _model_identity(model: Any, model_id: str, revision: str,
                    device: str) -> dict[str, Any]:
    try:
        dtype = str(next(model.parameters()).dtype)
    except (StopIteration, AttributeError):
        dtype = "unknown"
    return {
        "model_id": model_id,
        "revision": revision,
        "model_class": type(model).__name__,
        "dtype": dtype,
        "device": device,
    }


def _candidate_ids(tok: Any, candidates: Sequence[str]) -> list[list[int]]:
    return [
        _int_list(tok.encode(candidate, add_special_tokens=False),
                  "candidate_token_ids")
        for candidate in candidates
    ]


def _main_generate(args: argparse.Namespace) -> int:
    if not args.mode or not args.out or not args.run_id:
        raise TextEvidenceError("generation requires --mode, --out, and --run-id")
    if args.limit is not None and args.limit <= 0:
        raise TextEvidenceError("--limit must be positive")
    if args.batch <= 0:
        raise TextEvidenceError("--batch must be positive")
    max_new = (DEFAULT_MAX_NEW_TOKENS[args.mode]
               if args.max_new_tokens is None else args.max_new_tokens)
    if max_new <= 0:
        raise TextEvidenceError("--max-new-tokens must be positive")

    from latent_lab.train.checkpointing import require_pinned_revision

    args.revision = require_pinned_revision(args.revision)
    import torch
    import transformers
    from latent_lab.bench.suite_v3 import build_suite

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model, revision=args.revision
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16
    ).eval()
    model.to(args.device)

    suite = build_suite()
    suite_manifest = suite.manifest()
    examples = list(getattr(suite, args.split))
    if args.limit is not None:
        examples = examples[:args.limit]
    tokenizer_identity = _tokenizer_identity(tokenizer, model, args.revision)
    model_identity = _model_identity(model, args.model, args.revision, args.device)
    execution_context = _execution_context_from_env()

    records = []
    run_start = time.perf_counter()
    model_generate_calls = 0
    for start in range(0, len(examples), args.batch):
        chunk = examples[start:start + args.batch]
        messages = [prompt_messages(example, args.mode) for example in chunk]
        prompt_ids = [build_prompt(tokenizer, example, args.mode) for example in chunk]
        input_id_lists = [
            _int_list(value[0].tolist(), "input_ids") for value in prompt_ids
        ]
        candidate_ids = [
            _candidate_ids(tokenizer, example.candidates) for example in chunk
        ]
        batch_start = time.perf_counter()
        outputs = run_batched(model, tokenizer, prompt_ids, max_new, args.device)
        batch_wall = time.perf_counter() - batch_start
        model_generate_calls += 1
        allocated_wall = batch_wall / len(chunk)
        for index, (example, raw_output) in enumerate(zip(chunk, outputs)):
            compute = {
                "batch_index": model_generate_calls - 1,
                "batch_size": len(chunk),
                "batch_generate_wall_seconds": batch_wall,
                "allocated_generate_wall_seconds": allocated_wall,
                "prefill_tokens": len(input_id_lists[index]),
                "generated_tokens": len(raw_output["generated_token_ids"]),
                "candidate_tokenizer_calls": len(example.candidates),
                "prompt_tokenizer_calls": 1,
                "decode_calls": 1,
                "successful_task": True,
            }
            records.append(make_text_record(
                run_id=args.run_id,
                suite_manifest=suite_manifest,
                ex=example,
                mode=args.mode,
                seed=args.seed,
                model_identity=model_identity,
                tokenizer_identity=tokenizer_identity,
                messages=messages[index],
                input_ids=input_id_lists[index],
                candidate_token_ids=candidate_ids[index],
                generated_text=raw_output["generated_text"],
                generated_token_ids=raw_output["generated_token_ids"],
                max_new_tokens=max_new,
                error_status=None,
                compute=compute,
                execution_context=execution_context,
            ))
        print(
            f"[{args.mode}/{args.split}] {len(records)}/{len(examples)} raw records",
            flush=True,
        )

    report = make_text_report(
        run_id=args.run_id,
        suite_manifest=suite_manifest,
        mode=args.mode,
        split=args.split,
        seed=args.seed,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        max_new_tokens=max_new,
        records=records,
        wall_seconds=time.perf_counter() - run_start,
        peak_memory_bytes=peak_rss_bytes(),
        model_generate_calls=model_generate_calls,
        execution_context=execution_context,
    )
    offline_rescore_report(report)
    _write_report(Path(args.out), report)
    metrics = report["derived_metrics"]
    print(
        f"== {args.mode}/{args.split}: exact_match_rate="
        f"{metrics['exact_match_rate']:.6f} nontermination="
        f"{metrics['nontermination_count']} -> {args.out}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rescore", type=Path)
    parser.add_argument("--require-r1-preregistered", action="store_true")
    parser.add_argument("--expect-producer-command-index", type=int)
    parser.add_argument("--expect-mode", choices=sorted(MODE_DESCRIPTIONS))
    parser.add_argument("--expect-seed", type=int)
    parser.add_argument("--expect-split", choices=EVALUATION_SPLITS)
    parser.add_argument("--expect-run-id")
    parser.add_argument("--mode", choices=sorted(MODE_DESCRIPTIONS))
    parser.add_argument("--split", default="validation", choices=EVALUATION_SPLITS)
    parser.add_argument("--out")
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args(argv)
    if args.rescore is not None:
        if args.mode or args.run_id or args.limit is not None:
            raise TextEvidenceError(
                "--rescore cannot be combined with generation-only arguments"
            )
        expected_values = (
            args.expect_producer_command_index, args.expect_mode,
            args.expect_seed, args.expect_split, args.expect_run_id,
        )
        if not args.require_r1_preregistered and any(
                value is not None for value in expected_values):
            raise TextEvidenceError(
                "--expect-* arguments require --require-r1-preregistered"
            )
        result = load_and_rescore(args.rescore)
        if (args.require_r1_preregistered
                and not result["r1_preregistered_structure_valid"]):
            raise TextEvidenceError(
                "report lacks full preregistered R1 structure/plan binding"
            )
        if args.require_r1_preregistered:
            active_context = _execution_context_from_env()
            stored_context = result.get("execution_context")
            if active_context is None or stored_context is None or any(
                active_context[field] != stored_context[field]
                for field in (
                    "plan_hash", "driver_source_sha256",
                    "producer_source_sha256", "suite_hash",
                )
            ):
                raise TextEvidenceError(
                    "report is not bound to the active driver plan/source"
                )
            expectations = {
                "command_index": args.expect_producer_command_index,
                "mode": args.expect_mode,
                "seed": args.expect_seed,
                "split": args.expect_split,
                "run_id": args.expect_run_id,
            }
            if any(value is None for value in expectations.values()):
                raise TextEvidenceError(
                    "R1 rescore requires all expected producer metadata"
                )
            actual = {
                "command_index": stored_context["command_index"],
                "mode": result["mode"],
                "seed": result["seed"],
                "split": result["split"],
                "run_id": result["run_id"],
            }
            if actual != expectations:
                raise TextEvidenceError(
                    f"report producer metadata mismatch: {actual!r} != "
                    f"{expectations!r}"
                )
        rendered = json.dumps(result, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":"), allow_nan=False)
        if args.out:
            Path(args.out).write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return 0
    if args.require_r1_preregistered:
        raise TextEvidenceError(
            "--require-r1-preregistered is only valid with --rescore"
        )
    if any(value is not None for value in (
        args.expect_producer_command_index, args.expect_mode, args.expect_seed,
        args.expect_split, args.expect_run_id,
    )):
        raise TextEvidenceError("--expect-* arguments require --rescore")
    return _main_generate(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TextEvidenceError as error:
        raise SystemExit(f"text evidence error: {error}") from error

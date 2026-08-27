#!/usr/bin/env python3
"""Generate the immutable code-first R1 baseline audit.

This helper intentionally hashes Markdown without opening it.  Narrative claims
are audited only after ``artifacts/audit_before.json`` has been emitted.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GATE_ROOT = ROOT / ".rcc_work" / "audit_baseline_gate_full"
DRY_GATE_ROOT = ROOT / ".rcc_work" / "audit_baseline_gate_dry"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _markdown_hashes() -> list[dict[str, object]]:
    names = _git("ls-files", "*.md").splitlines()
    return [
        {
            "path": name,
            "sha256": _sha256(ROOT / name),
            "content_read_during_phase_a": False,
        }
        for name in sorted(names)
    ]


def _artifact_hashes(inventory: dict) -> list[dict[str, object]]:
    retained_kinds = {"adapter_checkpoint", "eval_json", "metrics_json"}
    return [
        {
            "id": row["id"],
            "kind": row["kind"],
            "sha256": row["sha256"],
            "size_bytes": row["size_bytes"],
        }
        for row in inventory["artifacts"]
        if row["kind"] in retained_kinds
    ]


def build() -> dict:
    full_gate_path = GATE_ROOT / "gate_verdict.json"
    dry_gate_path = DRY_GATE_ROOT / "gate_verdict.json"
    inventory_path = GATE_ROOT / "artifact_inventory.json"
    full_gate = _load_json(full_gate_path)
    dry_gate = _load_json(dry_gate_path)
    inventory = _load_json(inventory_path)
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    return {
        "schema_version": "r1_audit.v1",
        "phase": "before",
        "audit_method": {
            "code_first": True,
            "markdown_content_read": False,
            "paid_or_remote_operations": False,
            "large_model_downloads": False,
            "historical_artifacts_mutated": False,
        },
        "repository": {
            "baseline_git_sha": "aec85a9c8e60a9297b295ceba94cef8e6fb93ef3",
            "baseline_branch": "main",
            "baseline_dirty": False,
            "current_generation_commit": _git("rev-parse", "HEAD"),
        },
        "runtime": {
            "os": "Darwin 25.4.0 arm64",
            "system_python": "3.14.6",
            "project_venv_python": "3.14.0",
            "clean_contract_probe_python": "3.11.15",
            "uv": "0.11.28",
            "git": "2.46.0",
            "generator_python": platform.python_version(),
        },
        "dependency_groups": project.get("dependency-groups", {}),
        "baseline_commands": [
            {
                "command": ".venv/bin/python -m compileall -q rcc bench evals latent_lab tests",
                "exit_code": 0,
                "result": "PASS",
            },
            {
                "command": ".venv/bin/pytest -o addopts='' -q",
                "exit_code": 0,
                "result": "529 passed, 1 skipped in 30.04s",
            },
            {
                "command": ".venv/bin/pytest -o addopts='' -q tests/test_dictenc.py tests/test_gate.py tests/test_journal.py tests/test_router.py tests/test_scratch.py tests/test_security.py tests/test_session.py tests/test_store.py tests/test_tokens.py tests/test_provenance_links.py tests/test_text_parsing.py tests/test_telemetry_isolation.py",
                "exit_code": 0,
                "result": "98 passed in 0.57s",
                "selection": "explicit RCC core-only baseline",
            },
            {
                "command": "/private/tmp/rcc-r1-dev-baseline/bin/python -m compileall -q rcc bench evals latent_lab tests",
                "exit_code": 1,
                "result": "Python 3.11 SyntaxError in latent_lab/bench/no_spend_gate.py:1508",
            },
            {
                "command": ".venv/bin/python -m latent_lab.bench.no_spend_gate --dry-run --no-telemetry --out .rcc_work/audit_baseline_gate_dry",
                "exit_code": 1,
                "result": "NOT_READY: 6 blockers",
            },
            {
                "command": ".venv/bin/python -m latent_lab.bench.no_spend_gate --no-telemetry --out .rcc_work/audit_baseline_gate_full",
                "exit_code": 1,
                "result": "NOT_READY: 12 blockers",
            },
        ],
        "no_spend_gate": {
            "dry": {
                "verdict": dry_gate["verdict"],
                "blocker_codes": [row["code"] for row in dry_gate["blockers"]],
                "counts": dry_gate["counts"],
            },
            "full": {
                "verdict": full_gate["verdict"],
                "blocker_codes": [row["code"] for row in full_gate["blockers"]],
                "counts": full_gate["counts"],
                "prerequisites": full_gate["prerequisites"],
                "source_fingerprints": full_gate["inputs"]["source_fingerprints"],
            },
        },
        "artifact_inventory": {
            "source_path": ".rcc_work/audit_baseline_gate_full/artifact_inventory.json",
            "source_sha256": _sha256(inventory_path),
            "n_all_files": inventory["n_artifacts"],
            "checkpoint_and_eval_files": _artifact_hashes(inventory),
            "rescorability": {
                "historical_latent_eval_files": 50,
                "independently_rescorable_latent_eval_files": 0,
                "non_rescorable_missing_raw_prediction": 50,
                "historical_textual_files_with_preview_only": 6,
                "historical_2b_checkpoints_selected_by_legacy_scorer": 13,
            },
        },
        "audit_leads": [
            {"id": 1, "status": "CONFIRMED", "evidence": "pre-37725e1 scorer ranked candidate index 0 instead of ex.answer"},
            {"id": 2, "status": "CONFIRMED", "evidence": "50/50 latent eval files contain derived fields only and no raw candidate scores"},
            {"id": 3, "status": "CONFIRMED", "evidence": "13/13 2B best checkpoints were selected from histories produced by the legacy scorer; histories lack raw records"},
            {"id": 4, "status": "CONFIRMED", "evidence": "behavioral-v2 obj_track renders mutated final state as Initial situation"},
            {"id": 5, "status": "PARTIALLY_CONFIRMED", "evidence": "candidate counts vary from 2 through 100 and runtime ranks raw summed logprob; historical token lengths were not retained"},
            {"id": 6, "status": "CONFIRMED", "evidence": "latent_run/artifacts and corrected_scoring/no_spend accept incompatible raw fields and exact-tie policies"},
            {"id": 7, "status": "CONFIRMED_WITH_NONRESCORABLE_CAVEAT", "evidence": "stored 4B direct/no-thinking is 97/112 ID and 84/112 OOD versus native-thinking 1/112 and 0/112; full generations are absent"},
            {"id": 8, "status": "CONFIRMED", "evidence": "historical K runs trained separate adapters; same-adapter coverage omits K0 and LoRA affects prefill"},
            {"id": 9, "status": "CONFIRMED", "evidence": "full interval sends answer tokens after the first directly from embeddings to norm/lm_head without decoder layers"},
            {"id": 10, "status": "CONFIRMED", "evidence": "cache gradients detach while grad_checkpoint=true is persisted without an executed checkpointing path"},
            {"id": 11, "status": "CONFIRMED", "evidence": "VocabGuard misses tokenizer __call__, batch_decode, apply_chat_template, generate and alternate vocab paths"},
            {"id": 12, "status": "CONFIRMED", "evidence": "delimiter codec fails arbitrary delimiter keys, newlines, literal backslash-n and nested structures"},
            {"id": 13, "status": "CONFIRMED", "evidence": "numeric provenance accepts 142 from 1420 and 2 from 20.1 by substring matching"},
            {"id": 14, "status": "CONFIRMED", "evidence": "tampered persisted top-level run_id loads and compiles a foreign StoredRef"},
            {"id": 15, "status": "CONFIRMED", "evidence": "Python 3.11 compile fails and clean dev dependencies do not provide the lab test contract"},
            {"id": 16, "status": "REFUTED", "evidence": "built wheel contains latent_lab/bench; repository-root bench is a separate package not declared for installation"},
            {"id": 17, "status": "CONFIRMED", "evidence": "all 8 invalid 4B live runs are byte-identical to rejected copies; live reports are non-finite"},
            {"id": 18, "status": "DEFERRED_UNTIL_PHASE_B", "evidence": "Markdown content intentionally not read before this audit"},
        ],
        "additional_confirmed_contract_failures": [
            "RawStore.put trusts an existing content-addressed path without verifying bytes",
            "session load silently assigns zero tokens when raw content is missing and does not bind tokenizer identity",
            "Scratch accepts NaN confidence, unknown nonnumeric provenance and prompt-structural control text",
            "generic LatentBackend is mock-only scaffold, not the proven LocalizedRecurrence runtime ABI",
            "runtime compute counters omit candidate tails and are discarded by evaluation",
        ],
        "markdown_claims_pending": _markdown_hashes(),
        "baseline_verdict": {
            "status": "R1_NOT_READY",
            "paid_spend_authorized": False,
            "reason": "benchmark, scorer, runtime, RCC, reproducibility and artifact-truth contracts have executable blockers",
        },
    }


def main() -> int:
    output = ROOT / "artifacts" / "audit_before.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(output.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

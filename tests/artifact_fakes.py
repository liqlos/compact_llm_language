"""Builders for small fake-but-structurally-valid evidence artifacts.

Used by negative/positive contract tests: every builder produces
artifacts that pass the full strict validators, so tests can mutate one
field at a time and pin the exact rejection.
"""

from functools import lru_cache

import torch

from latent_lab.bench.latent_run import recipe_from_config
from latent_lab.train.checkpointing import (
    CHECKPOINT_FILE,
    RUN_MANIFEST_FILE,
    TRAIN_REPORT_FILE,
    adapter_state_sha256,
    atomic_write_json,
    save_adapter_bundle,
    sha256_file,
    write_run_status,
    write_train_generation,
)

REV_OK = "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b"
SUITE_SHA = "ab" * 32
MODEL = "Qwen/Qwen3.5-2B"
DEPS = {"python": "3.14.0", "torch": "2.13.0",
        "transformers": "5.15.1", "huggingface_hub": "1.28.0"}

BASE_CFG = {
    "mode": "E-localized", "interval": [12, 18],
    "k": 4, "max_k": 16, "lora_r": 8, "lora_alpha": 16.0,
    "lr": 1e-4, "steps": 800, "seed": 0,
    "optimizer": "adamw", "weight_decay": 0.01,
    "lr_schedule": "constant", "warmup": 50, "clip": 0.5,
    "detach_z0": False,
    "model": MODEL, "revision": REV_OK,
    "label": "E4_k4_s0", "device": "cpu",
    "train_examples": 10,
}


@lru_cache(maxsize=1)
def _current_suite():
    """Build behavioral-v3 once for current-evidence fixture identities."""
    from latent_lab.bench.suite_v3 import build_suite

    return build_suite()


def _current_suite_sha256() -> str:
    return _current_suite().records_hash()


def cfg(**over) -> dict:
    return {**BASE_CFG, **over}


def run_contract(config: dict | None = None, **over) -> dict:
    """The FULL canonical run expectation (every required key present).

    Built from the same config the artifact claims; ``over`` lets a test
    deliberately corrupt single fields (e.g. an unrelated config digest).
    """
    from latent_lab.bench.latent_run import recipe_from_config

    c = dict(config if config is not None else BASE_CFG)
    suite = c.get("suite_sha256", _current_suite_sha256())
    recipe = recipe_from_config(c, suite)
    out = {
        "model_id": c["model"],
        "revision": c["revision"],
        "suite_sha256": suite,
        "seed": c["seed"],
        "label": c.get("label"),
        "k": c["k"],
        "steps": c["steps"],
        "config_sha256": recipe["config_sha256"],
    }
    out.update(over)
    return out


def _state() -> dict:
    g = torch.Generator().manual_seed(5)
    return {
        "lora.0.A": torch.randn(8, 4, generator=g) * 0.01,
        "lora.0.B": torch.randn(4, 8, generator=g) * 0.01,
    }


def build_verified_run(out_dir, *, config=None) -> dict:
    """A complete, fully verifiable train generation (status/manifest/
    report/bundle all coherent under the strict schema)."""
    c = dict(config if config is not None else BASE_CFG)
    write_run_status(out_dir, "complete")
    ckpt_path = out_dir / CHECKPOINT_FILE
    state = _state()
    suite_sha256 = c.get("suite_sha256", _current_suite_sha256())
    bundle = save_adapter_bundle(
        ckpt_path, state, model_id=c["model"], revision=c["revision"],
        recipe=recipe_from_config(c, suite_sha256),
        metrics={"best_val_acc": 0.5})
    recipe = bundle["recipe"]
    state_sha256 = adapter_state_sha256(state)
    from latent_lab.bench.eval_v3 import build_eval_record, canonical_sha256
    from latent_lab.bench.latent_run import canonical_v3_history_entry

    records = []
    current_suite = _current_suite()
    current_examples = current_suite.validation[:2]
    use_current_examples = suite_sha256 == current_suite.records_hash()
    for index in range(2):
        if use_current_examples:
            example = current_examples[index]
            candidates = tuple(example.candidates)
            gold_answer = example.answer
            winner = (next(candidate for candidate in candidates
                           if candidate != gold_answer)
                      if index == 0 else gold_answer)
            rows = tuple(
                [-0.1] if candidate == winner else [-2.0 - candidate_index]
                for candidate_index, candidate in enumerate(candidates)
            )
            example_id = example.ex_id
            family = example.family
            prompt = example.prompt
            permutation_seed = example.candidate_permutation_seed
            permutation = tuple(example.candidate_permutation)
        else:
            candidates = ("yes", "no")
            gold_answer = "yes"
            rows = (([-0.1], [-1.0]), ([-2.0], [-0.2]))[index]
            example_id = f"validation-{index}"
            family = "fsm"
            prompt = f"state {index}"
            permutation_seed = index
            permutation = (0, 1)
        compute = {
            "prefill_layers": 12,
            "recurrence_interval_applications": 24,
            "k_loops": 4,
            "candidate_tail_layers": len(candidates) * 6,
            "lm_head_calls": len(candidates),
            "tokenizer_calls": len(candidates) + 1,
            "decode_calls": 0,
            "wall_seconds": 0.01,
            "peak_memory_bytes": None,
            "successful_task": True,
        }
        records.append(build_eval_record(
            run_id="verified-test-run", recipe_hash=canonical_sha256(recipe),
            model_id=c["model"], model_revision=c["revision"],
            adapter_id=str(out_dir), checkpoint_id="step-100",
            checkpoint_content_hash=state_sha256,
            suite_id="behavioral-v3", suite_version=3,
            suite_hash=suite_sha256,
            example_id=example_id, split="validation",
            family=family, prompt=prompt,
            candidates=candidates,
            candidate_permutation_seed=permutation_seed,
            candidate_permutation=permutation, gold_answer=gold_answer,
            per_token_logprobs=rows, k=c["k"],
            recurrence_config={"interval": c["interval"],
                               "trained_k": c["k"]},
            compute=compute))
    report = {
        "run_id": "verified-test-run",
        "adapter_id": str(out_dir),
        "config": c,
        "model": c["model"], "revision": c["revision"],
        "suite_sha256": suite_sha256,
        "recipe": recipe,
        "selection_provenance":
            "latent_eval.v3_recomputed_from_raw_validation_records",
        "selected_adapter_state_sha256": state_sha256,
        "checkpoint_content_digest": bundle["content_digest"],
        "checkpoint_sha256": sha256_file(ckpt_path),
        "best_val_acc": 0.5, "best_step": 100,
        "val_history": [canonical_v3_history_entry(100, records)],
        "final_train_loss": 0.1,
    }
    manifest = {
        "kind": "latent_lab.train_generation", "status": "complete",
        "argv": ["python", "-m", "latent_lab.bench.latent_run", "train",
                 "--label", str(c.get("label"))],
        "command": "python -m latent_lab.bench.latent_run train",
        "dependencies": dict(DEPS),
        "precision": {"backbone_dtype": "torch.float32"},
        "seed": c["seed"],
        "label": c.get("label"),
        "identity": {"model_id": c["model"], "revision": c["revision"]},
        "recipe": recipe,
        "suite_sha256": suite_sha256,
        "selected_adapter_state_sha256": state_sha256,
        "checkpoint_content_digest": bundle["content_digest"],
        "checkpoint_sha256": sha256_file(ckpt_path),
        "wall_seconds": 1.0,
    }
    write_train_generation(out_dir, manifest=manifest, report=report)
    return manifest


def eval_record(ex_id: str = "e0", scores=(2.0, 1.0, 0.5)) -> dict:
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    gold_idx = 0
    rank = order.index(gold_idx)
    return {
        "ex_id": ex_id, "family": "fsm", "depth": 3,
        "candidates": [f"c{i}" for i in range(len(scores))],
        "answer": "c0", "gold_candidate_index": gold_idx,
        "scores_raw": [float(s) for s in scores],
        "score_order": list(order),
        "rank_of_gold": rank,
        "correct": 1.0 if rank == 0 else 0.0,
        "n_candidates": len(scores),
    }


EVAL_IDENTITY = {
    "model_id": MODEL, "revision": REV_OK, "suite_sha256": SUITE_SHA,
    "checkpoint_content_digest": "cd" * 32,
    "split": "test_id", "ablation": "clean", "k_steps": 4, "seed": 0,
    "tokenizer_class": "PreTrainedTokenizerFast",
    "interval": [12, 18], "max_k": 16,
}


def build_eval_payload(**over) -> dict:
    """A complete eval payload with lossless recomputable raw records.

    ``identity=...`` REPLACES the default identity wholesale (so tests
    can drop or mutate single fields exactly)."""
    ident = over.pop("identity", None) or dict(EVAL_IDENTITY)
    k_steps = over.pop("k_steps", ident.get("k_steps", 4))
    records = over.pop("records", None) or [eval_record()]
    correct = sum(r["correct"] for r in records)
    split = ident.get("split")
    ablation = ident.get("ablation", "clean")
    res = {
        "tag": f"E-localized|{split}|clean|K={k_steps}",
        "ablate": {}, "k_steps": k_steps,
        "n": len(records), "accuracy": round(correct / len(records), 4),
        "by_depth": {}, "by_family": {}, "seconds": 1.0,
        "records": records,
    }
    d = {
        "status": "complete",
        "adapter": ".rcc_work/E4_k4_s0",
        "split": split, "config": {},
        "model": ident.get("model_id"), "revision": ident.get("revision"),
        "identity": {**ident, "k_steps": k_steps},
        "suite_sha256": ident.get("suite_sha256"),
        "device": "cpu", "seed": ident.get("seed"),
        "results": {ablation: res},
        "peak_rss_mib": 1.0, "platform": "test",
    }
    d.update(over)
    return d


def read_manifest(run_dir) -> dict:
    import json
    return json.loads((run_dir / RUN_MANIFEST_FILE).read_text())


def read_report(run_dir) -> dict:
    import json
    return json.loads((run_dir / TRAIN_REPORT_FILE).read_text())


def rewrite_json(path, mutation) -> None:
    import json
    d = json.loads(path.read_text())
    mutation(d)
    atomic_write_json(path, d)

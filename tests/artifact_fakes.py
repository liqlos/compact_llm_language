"""Builders for small fake-but-structurally-valid evidence artifacts.

Used by negative/positive contract tests: every builder produces
artifacts that pass the full strict validators, so tests can mutate one
field at a time and pin the exact rejection.
"""

import torch

from latent_lab.bench.latent_run import recipe_from_config
from latent_lab.train.checkpointing import (
    CHECKPOINT_FILE,
    RUN_MANIFEST_FILE,
    TRAIN_REPORT_FILE,
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
    "detach_z0": False, "grad_checkpoint": True,
    "model": MODEL, "revision": REV_OK,
    "label": "E4_k4_s0", "device": "cpu",
    "train_examples": 10,
}


def cfg(**over) -> dict:
    return {**BASE_CFG, **over}


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
    bundle = save_adapter_bundle(
        ckpt_path, _state(), model_id=c["model"], revision=c["revision"],
        recipe=recipe_from_config(c, c.get("suite_sha256", SUITE_SHA)),
        metrics={"best_val_acc": 0.5})
    recipe = bundle["recipe"]
    report = {
        "config": c,
        "model": c["model"], "revision": c["revision"],
        "suite_sha256": c.get("suite_sha256", SUITE_SHA),
        "recipe": recipe,
        "checkpoint_content_digest": bundle["content_digest"],
        "checkpoint_sha256": sha256_file(ckpt_path),
        "best_val_acc": 0.5, "best_step": 100,
        "val_history": [], "final_train_loss": 0.1,
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
        "suite_sha256": c.get("suite_sha256", SUITE_SHA),
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

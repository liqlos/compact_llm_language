"""Focused tests for the bounded no-spend integrity gate (fail-closed v2).

Covers the acceptance properties that must hold BEFORE any GPU spend:
corrected gold-aware scorer invariances, deterministic corrected checkpoint
selection, artifact discovery/hash/duplicate detection, train-report schema
blockers (missing fields AND schema violations are blockers, never
defaults), safe checkpoint classification through the project loader with
ACTUAL fp32 payload enforcement, single-lossless-representation rescoring
with explicit INVALID_* statuses (ties/duplicates/conflicts/non-finite),
the per-artifact relational join (report <-> fp32 identity-bound
checkpoint <-> raw evals on BOTH mandatory splits), orphan/duplicate/
quarantine completeness blockers, output/input overlap rejection,
source-fingerprint mutation detection, canonical byte-stability across
reruns, and driver exit semantics (0 READY / 1 NOT_READY / 2 execution
error).

Every case from the independent READY fail-open audit is reproduced here as
a focused negative test: an invalid evidence set must yield NOT_READY (or
an execution error), never READY.
"""

from __future__ import annotations

import json
import os
import re
import shutil

import pytest

from latent_lab.bench import no_spend_gate as g
from latent_lab.bench.corrected_scoring import (
    FLAG_DUPLICATE_CANDIDATES,
    FLAG_NONFINITE_SCORES,
    FLAG_NORMALIZED_MATCH,
    INVALID_AMBIGUOUS_TOP_TIE,
    INVALID_CONFLICTING_REPRESENTATIONS,
    INVALID_DUPLICATE_EXAMPLE_RECORDS,
    INVALID_MALFORMED_CANDIDATE_SCORES,
    INVALID_MALFORMED_RANKED_CANDIDATES,
    INVALID_UNKNOWN_EXAMPLE,
    MISSING_RAW_PREDICTION,
    corrected_score,
    normalize_answer,
    rescore_records,
    select_best_checkpoint,
)

ISO_TS = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")


def _strict_loads(blob: bytes) -> object:
    """Strict JSON parse: rejects NaN/Infinity tokens entirely."""
    return json.loads(
        blob,
        parse_constant=lambda c: pytest.fail(f"non-strict JSON constant {c}"),
    )


# ---------------------------------------------------------------------------
# corrected scorer invariances
# ---------------------------------------------------------------------------

class TestCorrectedScorer:
    def test_gold_not_at_index_zero_top_rank_is_correct(self):
        cands = ("a", "b", "gold", "c")
        cs = corrected_score(cands, "gold", order=[2, 0, 3, 1])
        assert cs.correct and cs.rank_of_gold == 0

    def test_gold_not_at_index_zero_other_ranked_first_incorrect(self):
        cands = ("a", "b", "gold", "c")
        cs = corrected_score(cands, "gold", order=[0, 2, 3, 1])
        assert not cs.correct and cs.rank_of_gold == 1

    def test_permutation_invariance_with_scores_attached_to_content(self):
        cands = ("a", "b", "gold", "c")
        scores = [0.1, 0.4, 0.9, -0.3]
        base = corrected_score(cands, "gold", scores=scores)
        perm = [3, 1, 0, 2]
        pc = [cands[i] for i in perm]
        ps = [scores[i] for i in perm]
        moved = corrected_score(pc, "gold", scores=ps)
        assert base.correct == moved.correct
        assert base.rank_of_gold == moved.rank_of_gold == 0
        # and a losing permutation of the same mapping stays losing
        scores2 = [0.9, 0.4, 0.1, -0.3]
        lost = corrected_score(cands, "gold", scores=scores2)
        assert not lost.correct and lost.rank_of_gold == 2

    def test_duplicate_candidates_flagged_not_silently_scored(self):
        cands = ("gold", "b", "gold", "c")
        cs = corrected_score(cands, "gold", order=[2, 0, 3, 1])
        assert FLAG_DUPLICATE_CANDIDATES in cs.flags
        assert cs.correct  # best-ranked gold occurrence is rank 0 here
        cs2 = corrected_score(cands, "gold", order=[1, 2, 3, 0])
        assert FLAG_DUPLICATE_CANDIDATES in cs2.flags
        assert cs2.rank_of_gold == 1

    def test_whitespace_parser_normalization(self):
        assert normalize_answer(" 42 ") == normalize_answer("42")
        assert normalize_answer("new\t york") == "new york"
        cs = corrected_score((" 42", "43"), "42", order=[0, 1])
        assert cs.correct and cs.rank_of_gold == 0
        # internal-whitespace parser difference still matches the gold and
        # is explicitly flagged rather than silently absorbed
        cs2 = corrected_score(("new  york", "boston"), "new\tyork",
                              order=[1, 0])
        assert not cs2.correct
        assert FLAG_NORMALIZED_MATCH in cs2.flags

    def test_gold_absent_from_candidates(self):
        cs = corrected_score(("a", "b"), "zzz", order=[0, 1])
        assert not cs.correct and cs.rank_of_gold == -1

    def test_nonfinite_scores_sink_and_flag(self):
        nan = float("nan")
        cs = corrected_score(("a", "gold"), "gold", scores=[nan, 0.5])
        assert FLAG_NONFINITE_SCORES in cs.flags and cs.correct

    def test_order_must_be_permutation(self):
        with pytest.raises(ValueError):
            corrected_score(("a", "b"), "a", order=[0, 0])


class TestSelectBestCheckpoint:
    def test_ignores_poisoned_stored_best_and_recomputes(self):
        hist = [{"step": 100, "accuracy": 0.2},
                {"step": 200, "accuracy": 0.9},
                {"step": 300, "accuracy": 0.9}]
        sel = select_best_checkpoint(hist)
        assert sel.step == 200          # earliest among tied best
        assert sel.metric == 0.9
        assert sel.provenance == "recomputed_from_history"

    def test_nan_metrics_rejected_not_skipped_silently(self):
        nan = float("nan")
        hist = [{"step": 100, "accuracy": nan},
                {"step": 200, "accuracy": 0.4}]
        sel = select_best_checkpoint(hist)
        assert sel.n_rejected_nonfinite == 1
        assert sel.step == 200
        assert select_best_checkpoint(
            [{"step": 1, "accuracy": nan}, {"step": 2, "accuracy": nan}]
        ) is None

    def test_empty_history(self):
        assert select_best_checkpoint([]) is None


class TestRescoreEligibilityFailClosed:
    EX_A = {"ex_id": "e0", "candidates": ("a", "b", "gold"),
            "answer": "gold"}
    EX_B = {"ex_id": "e1", "candidates": ("x", "gold", "y"),
            "answer": "gold"}

    def test_derived_only_records_are_non_rescorable(self):
        recs = [{"ex_id": "e0", "correct": 1.0, "rank_of_gold": 0,
                 "n_candidates": 3}]
        out = rescore_records(recs, {"e0": self.EX_A})
        assert out.status == MISSING_RAW_PREDICTION

    def test_predicted_answer_alone_is_lossy_never_a_rescore(self):
        out = rescore_records([{"ex_id": "e0", "predicted_answer": "gold"}],
                              {"e0": self.EX_A})
        assert out.status == MISSING_RAW_PREDICTION

    def test_raw_candidate_scores_allow_true_rescore(self):
        recs = [
            {"ex_id": "e0", "candidate_scores": [0.1, 0.2, 0.9]},
            {"ex_id": "e1", "candidate_scores": [0.9, 0.2, 0.1]},
        ]
        out = rescore_records(recs, {"e0": self.EX_A, "e1": self.EX_B})
        assert out.status == "RESCORED_CORRECTED"
        assert out.corrected_accuracy == pytest.approx(0.5)

    def test_ranked_candidates_full_permutation_rescores(self):
        out = rescore_records(
            [{"ex_id": "e0", "ranked_candidates": ["b", "gold", "a"]}],
            {"e0": self.EX_A})
        assert out.status == "RESCORED_CORRECTED"
        assert out.corrected_accuracy == pytest.approx(0.0)  # gold rank 1
        out2 = rescore_records(
            [{"ex_id": "e0", "ranked_candidates": ["gold", "a", "b"]}],
            {"e0": self.EX_A})
        assert out2.corrected_accuracy == pytest.approx(1.0)
        assert out.flags == ()

    def test_missing_example_is_invalid_status_not_crash(self):
        out = rescore_records([{"ex_id": "nope",
                                "candidate_scores": [1, 2, 3]}],
                              {"e0": self.EX_A})
        assert out.status == INVALID_UNKNOWN_EXAMPLE

    def test_duplicate_example_records_are_invalid(self):
        recs = [{"ex_id": "e0", "candidate_scores": [0.1, 0.2, 0.9]},
                {"ex_id": "e0", "candidate_scores": [0.9, 0.2, 0.1]}]
        out = rescore_records(recs, {"e0": self.EX_A})
        assert out.status == INVALID_DUPLICATE_EXAMPLE_RECORDS

    def test_conflicting_raw_representations_are_invalid(self):
        recs = [{"ex_id": "e0", "candidate_scores": [0.1, 0.2, 0.9],
                 "ranked_candidates": ["b", "gold", "a"]}]
        out = rescore_records(recs, {"e0": self.EX_A})
        assert out.status == INVALID_CONFLICTING_REPRESENTATIONS

    def test_malformed_scores_are_invalid_status_not_exception(self):
        for bad in ([0.1, 0.2],                      # length mismatch
                    ["0.9", 0.1, 0.2],               # non-numeric entry
                    [["0.9"], 0.1, 0.2]):            # nested junk
            out = rescore_records(
                [{"ex_id": "e0", "candidate_scores": bad}], {"e0": self.EX_A})
            assert out.status == INVALID_MALFORMED_CANDIDATE_SCORES, bad

    def test_string_nan_scores_are_invalid_not_sunk(self):
        out = rescore_records(
            [{"ex_id": "e0", "candidate_scores": ["NaN", 0.1, 0.2]}],
            {"e0": self.EX_A})
        assert out.status == INVALID_MALFORMED_CANDIDATE_SCORES

    def test_partial_ranking_is_invalid_status_not_exception(self):
        out = rescore_records(
            [{"ex_id": "e0", "ranked_candidates": ["gold"]}],
            {"e0": self.EX_A})
        assert out.status == INVALID_MALFORMED_RANKED_CANDIDATES

    def test_unknown_or_repeated_rank_entries_are_invalid(self):
        out = rescore_records(
            [{"ex_id": "e0",
              "ranked_candidates": ["gold", "zzz", "a"]}], {"e0": self.EX_A})
        assert out.status == INVALID_MALFORMED_RANKED_CANDIDATES
        out2 = rescore_records(
            [{"ex_id": "e0",
              "ranked_candidates": ["gold", "gold", "a"]}], {"e0": self.EX_A})
        assert out2.status == INVALID_MALFORMED_RANKED_CANDIDATES


# ---------------------------------------------------------------------------
# synthetic corpus + gate end-to-end
# ---------------------------------------------------------------------------

PINNED_REV_2B = "15852e8c16360a2fea060d615a32b45270f8a8fc"


def _valid_report(**over) -> dict:
    rep = {
        "config": {
            "mode": "E-localized", "interval": [12, 18], "k": 4,
            "lora_r": 8, "lr": 0.0001, "steps": 800, "seed": 0,
            "max_k": 16, "detach_z0": False, "device": "cuda",
            "train_examples": 490, "grad_checkpoint": True,
            "scorer": "corrected-gold-aware-v1",
        },
        "model": "Qwen/Qwen3.5-2B",
        "revision": PINNED_REV_2B,
        "suite_sha256": g.current_suite_sha256(),
        "best_val_acc": 0.5,
        "best_step": 200,
        "val_history": [
            {"tag": "val", "ablate": {}, "k_steps": 4, "n": 28,
             "accuracy": 0.25, "by_depth": {}, "by_family": {},
             "seconds": 1.0, "step": 100},
            {"tag": "val", "ablate": {}, "k_steps": 4, "n": 28,
             "accuracy": 0.5, "by_depth": {}, "by_family": {},
             "seconds": 1.0, "step": 200},
        ],
        "final_train_loss": 1.23,
        "gpu_mem": {},
        "wall_seconds": 10.0,
        "peak_rss_mib": 1.0,
        "platform": "test",
        "trainable_precision": "fp32",
    }
    rep.update(over)
    return rep


def _suite_ex(prefix="ti"):
    from latent_lab.bench.no_spend_gate import suite_examples_by_id

    for ex_id, ex in suite_examples_by_id().items():
        if ex_id.startswith(f"{prefix}-"):
            return ex
    raise AssertionError(f"no {prefix}- example")


def _raw_record(ex, *, gold_wins=True, tie=False) -> dict:
    n = len(ex.candidates)
    scores = [0.01 * i for i in range(n)]
    gold_pos = list(ex.candidates).index(ex.answer)
    scores[gold_pos] = 9.0 if gold_wins else -9.0
    if tie and n > 1:
        # duplicate the maximum at gold's position AND another one so the
        # top of the ranking is ambiguous regardless of where gold sits
        scores[gold_pos] = 9.0
        scores[(gold_pos + 1) % n] = 9.0
    return {"ex_id": ex.ex_id, "family": ex.family, "depth": ex.depth,
            "n_candidates": n, "correct": 1.0 if gold_wins else 0.0,
            "rank_of_gold": 0 if gold_wins else 1,
            "candidate_scores": scores}


def _raw_eval(adapter="runs/E_k4_s0", split="test_id", **over) -> dict:
    prefix = "ti" if split == "test_id" else "to"
    ev = {
        "adapter": adapter, "split": split,
        "model": "Qwen/Qwen3.5-2B", "revision": PINNED_REV_2B,
        "suite_sha256": g.current_suite_sha256(),
        "results": {"clean": {
            "tag": "t", "ablate": {}, "k_steps": 4, "n": 1,
            "accuracy": 1.0, "by_depth": {}, "by_family": {},
            "seconds": 1.0,
            "records": [_raw_record(_suite_ex(prefix))],
        }},
    }
    ev.update(over)
    return ev


def _derived_only_eval(adapter="runs/E_k4_s0") -> dict:
    ex = _suite_ex()
    return {
        "adapter": adapter, "split": "test_id",
        "model": "Qwen/Qwen3.5-2B", "revision": PINNED_REV_2B,
        "suite_sha256": g.current_suite_sha256(),
        "results": {"clean": {
            "tag": "t", "ablate": {}, "k_steps": 4, "n": 1,
            "accuracy": 1.0, "by_depth": {}, "by_family": {},
            "seconds": 1.0,
            "records": [
                {"ex_id": ex.ex_id, "family": ex.family,
                 "depth": ex.depth, "correct": 1.0,
                 "rank_of_gold": 0,
                 "n_candidates": len(ex.candidates)},
            ],
        }},
    }


@pytest.fixture()
def corpus(tmp_path):
    """Synthetic retained-evidence tree mirroring the historical shape."""
    r2b = tmp_path / "remote_results"
    (r2b / "runs" / "E_k4_s0").mkdir(parents=True)
    (r2b / "runs" / "legacy_run").mkdir(parents=True)

    # valid-schema report (still missing trainable_precision -> blocker)
    legacy_report = _valid_report()
    del legacy_report["config"]["scorer"]
    del legacy_report["trainable_precision"]
    (r2b / "runs" / "E_k4_s0" / "train_report.json").write_text(
        json.dumps(legacy_report))
    # plain-dict legacy checkpoint (bf16 lora + fp32 clock analog)
    import torch

    torch.save({"lora.0.A": torch.zeros(2, 4, dtype=torch.bfloat16),
                "clock.weight": torch.zeros(3, 4)},
               r2b / "runs" / "E_k4_s0" / "best_params.pt")

    # schema-violating report (revision not pinned)
    bad = _valid_report()
    bad["revision"] = "main"
    (r2b / "runs" / "legacy_run" / "train_report.json").write_text(
        json.dumps(bad))
    torch.save({"w": torch.zeros(2)}, r2b / "runs" / "legacy_run" /
               "best_params.pt")

    (r2b / "results").mkdir()
    (r2b / "results" / "ev_E_k4_s0_test_id_clean.json").write_text(
        json.dumps(_derived_only_eval()))

    # 4B: quarantined NaN batch fully duplicated into the live tree
    r4b = tmp_path / "remote_results_4b"
    for top in ("runs", "_rejected_nan_batch"):
        d = r4b / top / "E4_k1_s0"
        d.mkdir(parents=True)
        rep = _valid_report(model="Qwen/Qwen3.5-4B",
                            revision="8" * 40)
        rep["final_train_loss"] = float("nan")
        rep["best_val_acc"] = 1.0
        (d / "train_report.json").write_text(json.dumps(rep))
        torch.save({"w": torch.tensor([float("inf")])}, d / "best_params.pt")
    (r4b / "_rejected_nan_batch" / "REJECTED.md").write_text(
        "# REJECTED\nfinal_train_loss=NaN; fake best 1.0\n")
    return r2b, r4b


PROOF_OK = {"all_passed": True, "returncode": 0,
            "nodes": list(g.PROOF_TEST_NODES)}


class TestGateEndToEnd:
    def run_gate(self, corpus, **kw):
        r2b, r4b = corpus
        kw.setdefault("skip_proof_tests", True)
        return g.run_gate(r2b, r4b, repo_root=None, **kw)

    def test_discovery_counts_without_hand_list(self, corpus):
        res = self.run_gate(corpus)
        c = res.gate_verdict["counts"]
        assert c["runs_2b"] == 2
        assert c["runs_4b_live"] == 1
        assert c["runs_4b_rejected"] == 1
        assert c["files_scanned"] == 10
        assert c["checkpoints_total"] == 4

    def test_inventory_hashes_duplicates_hardlinks(self, tmp_path, corpus):
        r2b, r4b = corpus
        src = r2b / "runs" / "E_k4_s0" / "best_params.pt"
        dup_dir = r2b / "runs" / "dupe_run"
        dup_dir.mkdir()
        (dup_dir / "best_params.pt").write_bytes(src.read_bytes())
        os.link(src, r2b / "results" / "linked_copy.pt")

        res = self.run_gate(corpus)
        inv = {a["id"]: a for a in res.inventory["artifacts"]}
        a = inv["2b/runs/E_k4_s0/best_params.pt"]
        b = inv["2b/runs/dupe_run/best_params.pt"]
        assert a["sha256"] == b["sha256"]
        assert b["identical_content_with"] == sorted(
            ["2b/runs/E_k4_s0/best_params.pt", "2b/results/linked_copy.pt"])
        linked = inv["2b/results/linked_copy.pt"]
        assert linked["hardlinked_with"] == ["2b/runs/E_k4_s0/best_params.pt"]

    def test_historical_corpus_is_NOT_READY_with_exact_blockers(self, corpus):
        res = self.run_gate(corpus)
        v = res.gate_verdict
        assert v["verdict"] == "NOT_READY"
        codes = {b["code"] for b in v["blockers"]}
        assert "CKPT_LEGACY_UNBOUND_IDENTITY" in codes
        assert "NON_RESCORABLE_MISSING_RAW_PREDICTION" in codes
        assert "REPORT_SCHEMA_MISSING_FIELDS" in codes      # unpinned rev
        assert "REPORT_TRAINABLE_PRECISION_MISSING" in codes
        assert "TRAINABLES_NOT_FP32_IN_RETAINED_CHECKPOINTS" in codes
        assert "LIVE_4B_DUPLICATES_REJECTED_NAN_BATCH" in codes
        assert "EVAL_COVERAGE_INCOMPLETE" in codes          # join contract
        for b in v["blockers"]:
            assert b["smallest_next_action"].strip()

    def test_corrupt_checkpoint_classified_and_quarantine_analyzed(
            self, corpus):
        res = self.run_gate(corpus)
        runs = {(r["root"], r["run_id"]): r
                for r in res.artifact_verdicts["runs"]}
        q = res.artifact_verdicts["quarantine_4b"]
        assert runs[("4b", "E4_k1_s0")]["checkpoint"][
            "classification"] == "corrupt"
        assert q["quarantine_marker_present"] is True
        assert q["live_tree_duplicates_quarantine"] is True
        assert q["rejected_runs_with_nan_final_loss"] == ["E4_k1_s0"]
        assert q["live_vs_rejected_differing_files"] == []

    def test_strict_bundle_from_project_loader_is_LOADABLE(self, corpus):
        import torch

        from latent_lab.train.checkpointing import save_adapter_bundle

        r2b, _ = corpus
        run = r2b / "runs" / "bound_run"
        run.mkdir()
        state = {"lora.0.A": torch.eye(2, dtype=torch.float32)}
        save_adapter_bundle(run / "best_params.pt", state,
                            model_id="Qwen/Qwen3.5-2B",
                            revision=PINNED_REV_2B,
                            metrics={"best_score": 0.5, "best_step": 200})
        (run / "train_report.json").write_text(json.dumps(_valid_report()))

        res = self.run_gate(corpus)
        rv = [r for r in res.artifact_verdicts["runs"]
              if r["run_id"] == "bound_run"][0]
        ck = rv["checkpoint"]
        assert ck["classification"] == "loadable"
        assert "strict_project_loader_passed" in ck["reasons"]

    def test_identity_conflict_between_bundle_and_report_invalid(
            self, corpus):
        import torch

        from latent_lab.train.checkpointing import save_adapter_bundle

        r2b, _ = corpus
        run = r2b / "runs" / "conflict_run"
        run.mkdir()
        save_adapter_bundle(run / "best_params.pt",
                            {"lora.0.A": torch.eye(2)},
                            model_id="other/model", revision="b" * 40)
        (run / "train_report.json").write_text(json.dumps(_valid_report()))
        res = self.run_gate(corpus)
        rv = [r for r in res.artifact_verdicts["runs"]
              if r["run_id"] == "conflict_run"][0]
        assert rv["checkpoint"]["classification"] == "invalid"

    def test_rescorable_records_get_RESCORED_CORRECTED(self, corpus):
        ex_a, ex_b = _suite_ex("ti"), None
        for ex_id, ex in g.suite_examples_by_id().items():
            if ex_id.startswith("ti-") and ex_id != ex_a.ex_id:
                ex_b = ex
                break
        good = _raw_record(ex_a)
        bad = _raw_record(ex_b, gold_wins=False)
        ev = _derived_only_eval()
        ev["results"]["clean"]["records"] = [good, bad]
        r2b, _ = corpus
        p = r2b / "results" / "ev_raw_test_id_clean.json"
        p.write_text(json.dumps(ev))

        res = self.run_gate(corpus)
        entry = [e for e in res.artifact_verdicts["evaluations"]
                 if e["file"] == "ev_raw_test_id_clean.json"][0]
        assert entry["status"] == "RESCORED_CORRECTED"
        assert entry["corrected_accuracy"] == pytest.approx(0.5)

    def test_READY_positive_control_exit0_path(self, corpus, tmp_path):
        """Honest READY control: every prerequisite IS provable offline —
        proves the gate is neither rigged to permanent NOT_READY nor
        weakened into READY."""
        import torch

        from latent_lab.train.checkpointing import save_adapter_bundle

        r2b, r4b = corpus
        shutil.rmtree(r4b / "runs")           # live tree clean
        shutil.rmtree(r2b / "runs" / "legacy_run")
        (r2b / "results" / "ev_E_k4_s0_test_id_clean.json").unlink()

        run = r2b / "runs" / "E_k4_s0"
        os.remove(run / "best_params.pt")
        save_adapter_bundle(run / "best_params.pt",
                            {"lora.0.A": torch.eye(2, dtype=torch.float32)},
                            model_id="Qwen/Qwen3.5-2B",
                            revision=PINNED_REV_2B,
                            metrics={"best_score": 0.5, "best_step": 200})
        (run / "train_report.json").write_text(json.dumps(_valid_report()))
        for split in ("test_id", "test_ood"):
            p = r2b / "results" / f"ev_E_k4_s0_{split}.json"
            p.write_text(json.dumps(_raw_eval(split=split)))

        res = g.run_gate(r2b, r4b, repo_root=None, skip_proof_tests=True,
                         proof_result=PROOF_OK)
        v = res.gate_verdict
        assert v["verdict"] == "READY", json.dumps(v["blockers"], indent=1)
        statuses = {p["id"]: p["status"] for p in v["prerequisites"]}
        assert all(s == "PROVEN" for s in statuses.values())
        assert v["blockers"] == []
        assert v["counts"]["runs_2b_join_complete"] == 1
        assert g._exit_for("READY") == 0

    def test_canonical_outputs_byte_stable_and_timestamp_free(
            self, corpus, tmp_path):
        r2b, r4b = corpus
        outs = []
        for i in (1, 2):
            res = g.run_gate(r2b, r4b, repo_root=None, skip_proof_tests=True)
            d = tmp_path / f"out{i}"
            d.mkdir()
            g.write_canonical(d / "artifact_inventory.json", res.inventory)
            g.write_canonical(d / "artifact_verdicts.json",
                              res.artifact_verdicts)
            g.write_canonical(d / "gate_verdict.json", res.gate_verdict)
            (d / "GATE_REPORT.md").write_text(res.report_md)
            outs.append(d)

        names = ["artifact_inventory.json", "artifact_verdicts.json",
                 "gate_verdict.json", "GATE_REPORT.md"]
        for name in names:
            assert (outs[0] / name).read_bytes() == \
                (outs[1] / name).read_bytes(), name

        for name in names:
            raw = (outs[0] / name).read_text()
            assert not ISO_TS.search(raw), f"{name} leaks a wall-clock value"
            if name.endswith(".json"):
                doc = _strict_loads((outs[0] / name).read_bytes())
                assert isinstance(doc, dict)

    def test_dry_run_skips_payload_loading_but_still_hash(self, corpus):
        res = self.run_gate(corpus, dry_run=True)
        runs = res.artifact_verdicts["runs"]
        ck = [r["checkpoint"] for r in runs if r.get("checkpoint")]
        assert all(c["classification"] == "unproven" for c in ck)
        assert res.gate_verdict["counts"]["files_scanned"] == 10
        assert res.gate_verdict["counts"]["eval_files_checked"] == 0


# ---------------------------------------------------------------------------
# fail-closed repairs of every audited READY fail-open
# ---------------------------------------------------------------------------

class TestFailClosedRepairs:
    """Each test mirrors one audit repro: invalid evidence must yield
    NOT_READY with the specific blocker — NEVER READY."""

    def run_case(self, tmp_path, mutate=None):
        import torch

        from latent_lab.train.checkpointing import save_adapter_bundle

        r2b = tmp_path / "r2"
        run = r2b / "runs" / "runA"
        run.mkdir(parents=True)
        (run / "train_report.json").write_text(json.dumps(_valid_report()))
        save_adapter_bundle(
            run / "best_params.pt",
            {"lora.A": torch.eye(2, dtype=torch.float32)},
            model_id="Qwen/Qwen3.5-2B", revision=PINNED_REV_2B,
            metrics={"best_score": 0.5, "best_step": 200})
        results = r2b / "results"
        results.mkdir()
        for split in ("test_id", "test_ood"):
            (results / f"ev_runA_{split}.json").write_text(
                json.dumps(_raw_eval(adapter="runs/runA", split=split)))

        r4b = tmp_path / "r4"
        rej = r4b / "_rejected_nan_batch" / "E4_bad"
        rej.mkdir(parents=True)
        bad_rep = _valid_report(model="Qwen/Qwen3.5-4B", revision="8" * 40)
        bad_rep["final_train_loss"] = float("nan")
        bad_rep["best_val_acc"] = 1.0
        (rej / "train_report.json").write_text(json.dumps(bad_rep))
        torch.save({"w": torch.zeros(2)}, rej / "best_params.pt")
        (r4b / "_rejected_nan_batch" / "REJECTED.md").write_text("# R\n")

        if mutate:
            mutate(r2b, r4b)
        res = g.run_gate(r2b, r4b, repo_root=None, skip_proof_tests=True,
                         proof_result=PROOF_OK)
        codes = {b["code"] for b in res.gate_verdict["blockers"]}
        evals = {e["file"]: e.get("status")
                 for e in res.artifact_verdicts["evaluations"]}
        return res, codes, evals

    # -- case 1: bf16 trainables inside an identity-bound bundle ----------

    def test_bf16_trainables_in_bound_bundle_fail_closed(self, tmp_path):
        import torch

        from latent_lab.train.checkpointing import save_adapter_bundle

        def bf16(r2b, _):
            run = r2b / "runs" / "runA"
            rep = json.loads((run / "train_report.json").read_text())
            rep["trainable_precision"] = "bf16"
            (run / "train_report.json").write_text(json.dumps(rep))
            save_adapter_bundle(
                run / "best_params.pt",
                {"lora.A": torch.eye(2, dtype=torch.bfloat16)},
                model_id="Qwen/Qwen3.5-2B", revision=PINNED_REV_2B,
                metrics={"best_score": 0.5, "best_step": 200})

        res, codes, _ = self.run_case(tmp_path, bf16)
        assert res.verdict == "NOT_READY"
        assert "TRAINABLES_NOT_FP32_IN_RETAINED_CHECKPOINTS" in codes
        rv = [r for r in res.artifact_verdicts["runs"]
              if r["run_id"] == "runA"][0]
        assert rv["checkpoint"]["classification"] == "invalid"
        assert any(str(r).startswith("non_fp32_trainables_stored")
                   for r in rv["checkpoint"]["reasons"])
        assert "REPORT_TRAINABLE_PRECISION_NOT_FP32" in codes

    # -- case 2: mismatched suite hash ------------------------------------

    def test_wrong_eval_suite_hash_blocks_join(self, tmp_path):
        def wrong(r2b, _):
            p = r2b / "results" / "ev_runA_test_id.json"
            ev = json.loads(p.read_text())
            ev["suite_sha256"] = "0" * 64
            p.write_text(json.dumps(ev))

        res, codes, evals = self.run_case(tmp_path, wrong)
        assert res.verdict == "NOT_READY"
        assert "EVAL_SUITE_HASH_MISMATCH" in codes
        assert evals["ev_runA_test_id.json"] == "RESCORED_CORRECTED"

    # -- cases 3-6: empty records / malformed / NaN / ties -----------------

    def test_empty_records_block(self, tmp_path):
        def no_rec(r2b, _):
            p = r2b / "results" / "ev_runA_test_id.json"
            ev = json.loads(p.read_text())
            ev["results"] = {}
            p.write_text(json.dumps(ev))

        res, codes, evals = self.run_case(tmp_path, no_rec)
        assert res.verdict == "NOT_READY"
        assert "EVAL_EVIDENCE_INVALID" in codes
        assert evals["ev_runA_test_id.json"] == "NO_RECORDS"

    def test_malformed_json_blocks_without_crash(self, tmp_path):
        def broken(r2b, _):
            (r2b / "results" / "ev_runA_test_id.json").write_text(
                "{broken-json")

        res, codes, evals = self.run_case(tmp_path, broken)
        assert res.verdict == "NOT_READY"
        assert "EVAL_EVIDENCE_INVALID" in codes
        assert evals["ev_runA_test_id.json"] == "unreadable"

    def test_nan_literal_json_is_strictly_rejected(self, tmp_path):
        """json.dumps(float('nan')) emits a bare NaN literal — retained
        artifacts must reject it instead of parsing it silently."""

        def nanlit(r2b, _):
            p = r2b / "results" / "ev_runA_test_id.json"
            ev = json.loads(p.read_text())
            rec = ev["results"]["clean"]["records"][0]
            rec["candidate_scores"] = [
                float("nan")] * len(rec["candidate_scores"])
            p.write_text(json.dumps(ev))     # emits bare NaN literals

        res, codes, evals = self.run_case(tmp_path, nanlit)
        assert res.verdict == "NOT_READY"
        assert "EVAL_EVIDENCE_INVALID" in codes
        assert evals["ev_runA_test_id.json"] == "unreadable"

    def test_top_tie_is_ambiguous_invalid(self, tmp_path):
        def tied(r2b, _):
            for name in ("ev_runA_test_id.json", "ev_runA_test_ood.json"):
                p = r2b / "results" / name
                ev = json.loads(p.read_text())
                rec = ev["results"]["clean"]["records"][0]
                ex = _suite_ex("ti" if "test_id" in name else "to")
                ev["results"]["clean"]["records"] = [
                    _raw_record(ex, tie=True)]
                p.write_text(json.dumps(ev))

        res, codes, evals = self.run_case(tmp_path, tied)
        assert res.verdict == "NOT_READY"
        assert "EVAL_EVIDENCE_INVALID" in codes
        assert evals["ev_runA_test_id.json"] == INVALID_AMBIGUOUS_TOP_TIE
        assert evals["ev_runA_test_ood.json"] == INVALID_AMBIGUOUS_TOP_TIE

    def test_conflicting_representations_block(self, tmp_path):
        def conflict(r2b, _):
            p = r2b / "results" / "ev_runA_test_id.json"
            ev = json.loads(p.read_text())
            rec = ev["results"]["clean"]["records"][0]
            ex = _suite_ex("ti")
            rec["ranked_candidates"] = list(ex.candidates)
            p.write_text(json.dumps(ev))

        res, codes, evals = self.run_case(tmp_path, conflict)
        assert res.verdict == "NOT_READY"
        assert evals["ev_runA_test_id.json"] == \
            INVALID_CONFLICTING_REPRESENTATIONS

    def test_duplicate_example_records_block(self, tmp_path):
        def dupe(r2b, _):
            p = r2b / "results" / "ev_runA_test_id.json"
            ev = json.loads(p.read_text())
            recs = ev["results"]["clean"]["records"]
            ev["results"]["clean"]["records"] = [recs[0], dict(recs[0])]
            p.write_text(json.dumps(ev))

        res, _, evals = self.run_case(tmp_path, dupe)
        assert res.verdict == "NOT_READY"
        assert evals["ev_runA_test_id.json"] == \
            INVALID_DUPLICATE_EXAMPLE_RECORDS

    def test_partial_ranking_is_invalid_not_exception(self, tmp_path):
        def partial(r2b, _):
            p = r2b / "results" / "ev_runA_test_id.json"
            ev = json.loads(p.read_text())
            rec = ev["results"]["clean"]["records"][0]
            del rec["candidate_scores"]
            rec["ranked_candidates"] = ["only-one"]
            p.write_text(json.dumps(ev))

        res, codes, evals = self.run_case(tmp_path, partial)
        assert res.verdict == "NOT_READY"
        assert "EVAL_EVIDENCE_INVALID" in codes      # NOT an execution crash
        assert evals["ev_runA_test_id.json"] == \
            INVALID_MALFORMED_RANKED_CANDIDATES

    # -- cases 7-8: relational join coverage -------------------------------

    def test_second_checkpoint_without_eval_blocks_join(self, tmp_path):
        import torch

        from latent_lab.train.checkpointing import save_adapter_bundle

        def extra_run(r2b, _):
            run = r2b / "runs" / "runB"
            run.mkdir()
            (run / "train_report.json").write_text(
                json.dumps(_valid_report()))
            save_adapter_bundle(
                run / "best_params.pt",
                {"lora.A": torch.eye(2, dtype=torch.float32) * 2},
                model_id="Qwen/Qwen3.5-2B", revision=PINNED_REV_2B,
                metrics={"best_score": 0.5, "best_step": 200})

        res, codes, _ = self.run_case(tmp_path, extra_run)
        assert res.verdict == "NOT_READY"
        assert "EVAL_COVERAGE_INCOMPLETE" in codes
        jv = {r["dir"]: r.get("join") for r in res.artifact_verdicts["runs"]
              if r.get("join")}
        assert jv["runs/runB"]["complete"] is False

    def test_only_one_mandatory_split_covered_blocks_join(self, tmp_path):
        def drop_ood(r2b, _):
            (r2b / "results" / "ev_runA_test_ood.json").unlink()

        res, codes, _ = self.run_case(tmp_path, drop_ood)
        assert res.verdict == "NOT_READY"
        assert "EVAL_COVERAGE_INCOMPLETE" in codes
        rv = [r for r in res.artifact_verdicts["runs"]
              if r["run_id"] == "runA"][0]
        assert rv["join"]["mandatory_splits_missing"] == ["test_ood"]

    # -- case 9: orphans + ambiguous duplicates -----------------------------

    def test_orphan_corrupt_checkpoint_blocks(self, tmp_path):
        import torch

        def orphan(r2b, _):
            d = r2b / "runs" / "orphan"
            d.mkdir()
            torch.save({"w": torch.tensor([float("nan")])},
                       d / "best_params.pt")

        res, codes, _ = self.run_case(tmp_path, orphan)
        assert res.verdict == "NOT_READY"
        assert "ORPHAN_EVIDENCE" in codes
        assert "CKPT_AMBIGUOUS_DUPLICATE_BINDING" not in codes
        assert res.artifact_verdicts["orphans"]["checkpoints"] == \
            ["2b/runs/orphan/best_params.pt"]

    def test_byte_identical_checkpoints_bind_ambiguously(self, tmp_path):
        def dupe_binding(r2b, _):
            src = r2b / "runs" / "runA" / "best_params.pt"
            d = r2b / "runs" / "runA_clone"
            d.mkdir()
            (d / "best_params.pt").write_bytes(src.read_bytes())

        res, codes, _ = self.run_case(tmp_path, dupe_binding)
        assert res.verdict == "NOT_READY"
        assert "CKPT_AMBIGUOUS_DUPLICATE_BINDING" in codes

    # -- case 10: quarantine masking / completeness --------------------------

    def test_one_differing_file_no_longer_masks_live_duplication(
            self, tmp_path):
        import torch

        from latent_lab.train.checkpointing import save_adapter_bundle

        def partial_dup(r2b, r4b):
            rejected = r4b / "_rejected_nan_batch" / "E4_bad"
            torch.save({"w": torch.tensor([float("inf")])},
                       rejected / "best_params.pt")
            live = r4b / "runs" / "E4_bad"
            live.mkdir(parents=True)
            shutil.copy2(rejected / "train_report.json",
                         live / "train_report.json")
            save_adapter_bundle(
                live / "best_params.pt",
                {"lora.A": torch.eye(2, dtype=torch.float32)},
                model_id="Qwen/Qwen3.5-4B", revision="8" * 40,
                metrics={"best_score": 1.0, "best_step": 200})

        res, codes, _ = self.run_case(tmp_path, partial_dup)
        assert res.verdict == "NOT_READY"
        assert "LIVE_4B_DUPLICATES_REJECTED_NAN_BATCH" in codes
        q = res.artifact_verdicts["quarantine_4b"]
        assert q["live_vs_rejected_identical_files"] == \
            ["E4_bad/train_report.json"]
        assert q["live_vs_rejected_differing_files"] == \
            ["E4_bad/best_params.pt"]

    def test_marker_only_quarantine_is_incomplete(self, tmp_path):
        def marker_only(_, r4b):
            shutil.rmtree(r4b / "_rejected_nan_batch" / "E4_bad")

        res, codes, _ = self.run_case(tmp_path, marker_only)
        assert res.verdict == "NOT_READY"
        assert "QUARANTINE_INCOMPLETE" in codes

    def test_missing_quarantine_marker_blocks(self, tmp_path):
        def no_marker(_, r4b):
            (r4b / "_rejected_nan_batch" / "REJECTED.md").unlink()

        res, codes, _ = self.run_case(tmp_path, no_marker)
        assert res.verdict == "NOT_READY"
        assert "QUARANTINE_MARKER_MISSING" in codes

    def test_live_known_invalid_4b_artifacts_block(self, tmp_path):
        def live_nan(r2b, r4b):
            rep = _valid_report(model="Qwen/Qwen3.5-4B", revision="8" * 40)
            rep["final_train_loss"] = float("nan")
            d = r4b / "runs" / "E4_live_bad"
            d.mkdir(parents=True)
            (d / "train_report.json").write_text(json.dumps(rep))

        res, codes, _ = self.run_case(tmp_path, live_nan)
        assert res.verdict == "NOT_READY"
        assert "LIVE_4B_KNOWN_INVALID_ARTIFACTS" in codes

    def test_live_noncanonical_report_blocks(self, tmp_path):
        def live_noncanon(r2b, r4b):
            d = r4b / "runs" / "E4_live_nc"
            d.mkdir(parents=True)
            (d / "train_report.json").write_text(
                '{"final_train_loss": NaN, "best_val_acc": 0.4}')

        res, codes, _ = self.run_case(tmp_path, live_noncanon)
        assert res.verdict == "NOT_READY"
        assert "LIVE_4B_KNOWN_INVALID_ARTIFACTS" in codes

    # -- report schema hardening ---------------------------------------------

    @pytest.mark.parametrize("field,value,problem", [
        ("best_step", 200.5, "best_step:type"),
        ("best_step", True, "best_step:type"),
        ("best_step", -1, "best_step:type"),
        ("trainable_precision", "bf16", "trainable_precision:not_fp32"),
        ("val_history", [{"step": 100, "accuracy": 0.5},
                         {"step": 200.7, "accuracy": 0.6}],
         "val_history[1].step:type"),
        ("val_history", [{"step": 100, "accuracy": 0.5},
                         {"step": True, "accuracy": 0.6}],
         "val_history[1].step:type"),
    ])
    def test_schema_violations_are_blockers_never_defaults(
            self, tmp_path, field, value, problem):
        def badrep(r2b, _):
            p = r2b / "runs" / "runA" / "train_report.json"
            rep = json.loads(p.read_text())
            rep[field] = value
            p.write_text(json.dumps(rep))

        res, codes, _ = self.run_case(tmp_path, badrep)
        assert res.verdict == "NOT_READY"
        rv = [r for r in res.artifact_verdicts["runs"]
              if r["run_id"] == "runA"][0]
        assert problem in rv["report_problems"], rv["report_problems"]
        assert "REPORT_SCHEMA_MISSING_FIELDS" in codes or \
            problem.startswith("trainable_precision")

    def test_nonfinite_report_metric_via_json_literal_is_unreadable(
            self, tmp_path):
        def nanrep(r2b, _):
            p = r2b / "runs" / "runA" / "train_report.json"
            text = p.read_text().replace('"final_train_loss": 1.23',
                                         '"final_train_loss": NaN')
            p.write_text(text)

        res, codes, _ = self.run_case(tmp_path, nanrep)
        assert res.verdict == "NOT_READY"
        assert "REPORT_SCHEMA_MISSING_FIELDS" in codes

    def test_invalid_history_fails_selection_never_skip_and_select(
            self, tmp_path):
        def badhist(r2b, _):
            p = r2b / "runs" / "runA" / "train_report.json"
            rep = json.loads(p.read_text())
            rep["val_history"].append({"step": 300, "junk": True})
            p.write_text(json.dumps(rep))

        res, codes, _ = self.run_case(tmp_path, badhist)
        assert res.verdict == "NOT_READY"
        rv = [r for r in res.artifact_verdicts["runs"]
              if r["run_id"] == "runA"][0]
        assert "val_history[2].accuracy:missing" in rv["report_problems"]
        assert rv["selection_check"]["consistent_with_reported"] is False
        assert "SELECTION_PROVENANCE_NOT_CORRECTED" in codes

    def test_eval_identity_mismatch_blocks_join(self, tmp_path):
        def wrong_model(r2b, _):
            p = r2b / "results" / "ev_runA_test_id.json"
            ev = json.loads(p.read_text())
            ev["model"] = "Qwen/Qwen3.5-4B"
            ev["revision"] = "8" * 40
            p.write_text(json.dumps(ev))

        res, codes, _ = self.run_case(tmp_path, wrong_model)
        assert res.verdict == "NOT_READY"
        assert "EVAL_IDENTITY_MISMATCH" in codes
        assert "EVAL_COVERAGE_INCOMPLETE" in codes

    def test_unbound_eval_file_blocks(self, tmp_path):
        def stray(r2b, _):
            ev = _raw_eval(adapter="runs/ghost_run", split="test_id")
            (r2b / "results" / "ev_ghost_test_id.json").write_text(
                json.dumps(ev))

        res, codes, _ = self.run_case(tmp_path, stray)
        assert res.verdict == "NOT_READY"
        assert "EVAL_UNBOUND_TO_RETAINED_RUN" in codes


# ---------------------------------------------------------------------------
# output/input safety: overlap rejection + source fingerprints
# ---------------------------------------------------------------------------

class TestOutputInputSafety:
    def _build_inputs(self, tmp_path):
        import torch

        from latent_lab.train.checkpointing import save_adapter_bundle

        base = tmp_path / "inputs"
        r2b = base / "r2"
        (r2b / "runs" / "runA").mkdir(parents=True)
        (r2b / "runs" / "runA" / "train_report.json").write_text(
            json.dumps(_valid_report()))
        save_adapter_bundle(
            r2b / "runs" / "runA" / "best_params.pt",
            {"lora.A": torch.eye(2, dtype=torch.float32)},
            model_id="Qwen/Qwen3.5-2B", revision=PINNED_REV_2B,
            metrics={"best_score": 0.5, "best_step": 200})
        (r2b / "results").mkdir()
        for split in ("test_id", "test_ood"):
            (r2b / "results" / f"ev_runA_{split}.json").write_text(
                json.dumps(_raw_eval(adapter="runs/runA", split=split)))
        r4b = base / "r4"
        rej = r4b / "_rejected_nan_batch" / "E4_bad"
        rej.mkdir(parents=True)
        bad_rep = _valid_report(model="Qwen/Qwen3.5-4B", revision="8" * 40)
        bad_rep["final_train_loss"] = float("nan")
        bad_rep["best_val_acc"] = 1.0
        (rej / "train_report.json").write_text(json.dumps(bad_rep))
        torch.save({"w": torch.zeros(2)}, rej / "best_params.pt")
        (r4b / "_rejected_nan_batch" / "REJECTED.md").write_text("# R\n")
        return r2b, r4b

    @pytest.mark.parametrize("out_rel", [
        "inputs/r2/out",       # output inside an input root
        "inputs",              # an input root inside the output dir
    ])
    def test_output_input_overlap_rejected_before_writing(
            self, tmp_path, out_rel, capsys):
        r2b, r4b = self._build_inputs(tmp_path)
        before = g.fingerprint_roots([("2b", r2b), ("4b", r4b)])
        rc = g.main(["--results-2b", str(r2b), "--results-4b", str(r4b),
                     "--out", str(tmp_path / out_rel), "--skip-proof-tests",
                     "--no-telemetry"])
        assert rc == 2
        assert "overlap" in capsys.readouterr().err
        assert g.fingerprint_roots([("2b", r2b), ("4b", r4b)]) == before

    def test_equal_paths_overlap_rejected(self, tmp_path, capsys):
        r2b, r4b = self._build_inputs(tmp_path)
        rc = g.main(["--results-2b", str(r2b), "--results-4b", str(r4b),
                     "--out", str(r2b), "--skip-proof-tests",
                     "--no-telemetry"])
        assert rc == 2
        assert "overlap" in capsys.readouterr().err

    def test_source_mutation_during_scan_is_execution_error(
            self, tmp_path, monkeypatch):
        r2b, r4b = self._build_inputs(tmp_path)
        real = g.evaluate_eval_file

        def mutating(path, examples_by_id):
            out = real(path, examples_by_id)
            # simulate concurrent evidence tampering mid-scan
            victim = next((r2b / "results").glob("ev_runA_*.json"))
            victim.write_text(victim.read_text() + " ")
            return out

        monkeypatch.setattr(g, "evaluate_eval_file", mutating)
        with pytest.raises(g.GateExecutionError, match="fingerprint"):
            g.run_gate(r2b, r4b, repo_root=None, skip_proof_tests=True)

    def test_fingerprint_deterministic_and_location_independent(
            self, tmp_path):
        import uuid

        r2b, r4b = self._build_inputs(tmp_path)
        fp1 = g.fingerprint_roots([("2b", r2b), ("4b", r4b)])
        fp2 = g.fingerprint_roots([("2b", r2b), ("4b", r4b)])
        assert fp1 == fp2
        # same CONTENT under a differently named tree -> same fingerprint
        clone = tmp_path / f"clone-{uuid.uuid4().hex[:8]}"
        shutil.copytree(r2b, clone)
        fp3 = g.fingerprint_roots([("2b", clone), ("4b", r4b)])
        assert fp3 == fp1

    def test_main_twice_disjoint_outputs_byte_identical_and_sources_stable(
            self, tmp_path):
        r2b, r4b = self._build_inputs(tmp_path)
        fp_before = g.fingerprint_roots([("2b", r2b), ("4b", r4b)])
        outs = []
        for i in (1, 2):
            out = tmp_path / f"gate_out_{i}"
            rc = g.main(["--results-2b", str(r2b), "--results-4b", str(r4b),
                         "--out", str(out), "--skip-proof-tests",
                         "--no-telemetry"])
            assert rc in (0, 1)
            outs.append(out)
        for name in ("artifact_inventory.json", "artifact_verdicts.json",
                     "gate_verdict.json", "GATE_REPORT.md"):
            a = (outs[0] / name).read_bytes()
            b = (outs[1] / name).read_bytes()
            assert a == b, name
            doc = _strict_loads(a) if name.endswith(".json") else None
            if doc is not None:
                assert isinstance(doc, dict)
        v = json.loads((outs[0] / "gate_verdict.json").read_text())
        assert v["inputs"]["source_unchanged"] is True
        assert v["inputs"]["source_fingerprint_before"] == \
            v["inputs"]["source_fingerprint_after"] == fp_before
        assert g.fingerprint_roots([("2b", r2b), ("4b", r4b)]) == fp_before

    def test_ready_control_main_exit0_and_notready_exit1(self, tmp_path):
        import torch

        from latent_lab.train.checkpointing import save_adapter_bundle

        r2b, r4b = self._build_inputs(tmp_path)
        out_ok = tmp_path / "ok"
        # run_gate with executed proofs -> READY; the exit mapper maps it 0.
        res = g.run_gate(r2b, r4b, repo_root=None, skip_proof_tests=True,
                         proof_result=PROOF_OK)
        assert res.verdict == "READY", json.dumps(
            res.gate_verdict["blockers"], indent=1)
        assert g._exit_for(res.verdict) == 0

        # skipping proofs (as the CLI does) leaves runtime UNPROVEN and
        # must NOT yield a silent NOT_READY: an explicit blocker explains
        # every non-PROVEN prerequisite.
        rc = g.main(["--results-2b", str(r2b), "--results-4b", str(r4b),
                     "--out", str(out_ok), "--skip-proof-tests",
                     "--no-telemetry"])
        assert rc == 1
        v = json.loads((out_ok / "gate_verdict.json").read_text())
        statuses = {p["id"]: p["status"] for p in v["prerequisites"]}
        assert statuses[g.PREREQ_RUNTIME] == "UNPROVEN"
        codes = {b["code"] for b in v["blockers"]}
        assert "PREREQS_UNPROVEN" in codes

        (r2b / "results" / "ev_runA_test_ood.json").unlink()
        out_bad = tmp_path / "bad"
        assert g.main(["--results-2b", str(r2b), "--results-4b", str(r4b),
                       "--out", str(out_bad), "--skip-proof-tests",
                       "--no-telemetry"]) == 1


# ---------------------------------------------------------------------------
# CLI exit semantics
# ---------------------------------------------------------------------------

class TestExitSemantics:
    def test_exit_mapping(self):
        assert g._exit_for("READY") == 0
        assert g._exit_for("NOT_READY") == 1
        with pytest.raises(ValueError):
            g._exit_for("SOMETHING_ELSE")

    def test_missing_input_roots_is_execution_error(self, tmp_path, capsys):
        rc = g.main(["--results-2b", str(tmp_path / "nope"),
                     "--results-4b", str(tmp_path / "nada"),
                     "--out", str(tmp_path / "o"), "--no-telemetry"])
        assert rc == 2
        assert "execution error" in capsys.readouterr().err

    def test_main_end_to_end_not_ready_writes_artifacts(
            self, corpus, tmp_path, capsys):
        r2b, r4b = corpus
        out = tmp_path / "gate_out"
        rc = g.main(["--results-2b", str(r2b), "--results-4b", str(r4b),
                     "--out", str(out), "--skip-proof-tests",
                     "--no-telemetry"])
        assert rc == 1
        for name in ("artifact_inventory.json", "artifact_verdicts.json",
                     "gate_verdict.json", "GATE_REPORT.md"):
            assert (out / name).is_file(), name
        v = json.loads((out / "gate_verdict.json").read_text())
        assert v["verdict"] == "NOT_READY"
        assert "NOT_READY" in capsys.readouterr().out

    def test_telemetry_timestamp_is_separate_and_optional(
            self, corpus, tmp_path):
        r2b, r4b = corpus
        out = tmp_path / "g2"
        g.main(["--results-2b", str(r2b), "--results-4b", str(r4b),
                "--out", str(out), "--skip-proof-tests"])
        tel = json.loads((out / "telemetry_timestamp.json").read_text())
        assert set(tel) == {"note", "telemetry_generated_at_utc"}
        verdict_text = (out / "gate_verdict.json").read_text()
        assert "telemetry_generated_at_utc" not in verdict_text

        out3 = tmp_path / "g3"
        rc = g.main(["--results-2b", str(r2b), "--results-4b", str(r4b),
                     "--out", str(out3), "--skip-proof-tests",
                     "--no-telemetry"])
        assert rc == 1
        assert not (out3 / "telemetry_timestamp.json").exists()

    def test_math_domain_no_nan_leaks_into_canonical_json(self, corpus):
        r2b, r4b = corpus
        res = g.run_gate(r2b, r4b, repo_root=None, skip_proof_tests=True)
        for payload in (res.inventory, res.artifact_verdicts,
                        res.gate_verdict):
            blob = g.canonical_json_bytes(payload)
            # strict parser must accept our own output: no NaN/Infinity
            # tokens anywhere
            doc = json.loads(blob, parse_constant=lambda c: pytest.fail(
                f"non-strict JSON constant {c} leaked into canonical output"))
            # and re-serializing with allow_nan=False proves no non-finite
            # float survived anywhere in the payload
            json.dumps(doc, allow_nan=False)
            assert isinstance(doc, dict)

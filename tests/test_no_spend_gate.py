"""Focused tests for the bounded no-spend integrity gate.

Covers the acceptance properties that must hold BEFORE any GPU spend:
corrected gold-aware scorer invariances, deterministic corrected checkpoint
selection, artifact discovery/hash/duplicate detection, train-report schema
blockers (missing fields are blockers, not defaults), safe checkpoint
classification through the project loader, rescore-eligibility honesty
(NON_RESCORABLE_MISSING_RAW_PREDICTION), 4B quarantine duplication
detection, canonical byte-stability across reruns, and driver exit
semantics (0 READY / 1 NOT_READY / 2 execution error).
"""

from __future__ import annotations

import json
import math
import os
import re

import pytest

from latent_lab.bench import no_spend_gate as g
from latent_lab.bench.corrected_scoring import (
    FLAG_DUPLICATE_CANDIDATES,
    FLAG_NONFINITE_SCORES,
    FLAG_NORMALIZED_MATCH,
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


class TestRescoreEligibility:
    EX = {"ex_id": "e0", "candidates": ("a", "b", "gold"),
          "answer": "gold"}

    def test_derived_only_records_are_non_rescorable(self):
        recs = [{"ex_id": "e0", "correct": 1.0, "rank_of_gold": 0,
                 "n_candidates": 3}]
        out = rescore_records(recs, {"e0": self.EX})
        assert out.status == MISSING_RAW_PREDICTION

    def test_raw_candidate_scores_allow_true_rescore(self):
        recs = [
            {"ex_id": "e0", "candidate_scores": [0.1, 0.2, 0.9]},
            {"ex_id": "e0", "candidate_scores": [0.9, 0.2, 0.1]},
        ]
        out = rescore_records(recs, {"e0": self.EX})
        assert out.status == "RESCORED_CORRECTED"
        assert out.corrected_accuracy == pytest.approx(0.5)

    def test_missing_example_blocks_whole_file(self):
        out = rescore_records([{"ex_id": "nope",
                                "candidate_scores": [1, 2, 3]}],
                              {"e0": self.EX})
        assert out.status == MISSING_RAW_PREDICTION


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
    }
    rep.update(over)
    return rep


def _derived_only_eval(adapter="runs/E_k4_s0") -> dict:
    return {
        "adapter": adapter, "split": "test_id",
        "model": "Qwen/Qwen3.5-2B", "revision": PINNED_REV_2B,
        "suite_sha256": g.current_suite_sha256(),
        "results": {"clean": {
            "tag": "t", "ablate": {}, "k_steps": 4, "n": 1,
            "accuracy": 1.0, "by_depth": {}, "by_family": {},
            "seconds": 1.0,
            "records": [
                {"ex_id": _suite_ex().ex_id, "family": _suite_ex().family,
                 "depth": _suite_ex().depth, "correct": 1.0,
                 "rank_of_gold": 0, "n_candidates": len(_suite_ex().candidates)},
            ],
        }},
    }


def _suite_ex():
    from latent_lab.bench.no_spend_gate import suite_examples_by_id

    for ex_id, ex in suite_examples_by_id().items():
        if ex_id.startswith("ti-"):
            return ex
    raise AssertionError("no test_id example")


@pytest.fixture()
def corpus(tmp_path):
    """Synthetic retained-evidence tree mirroring the historical shape."""
    r2b = tmp_path / "remote_results"
    (r2b / "runs" / "E_k4_s0").mkdir(parents=True)
    (r2b / "runs" / "legacy_run").mkdir(parents=True)

    # valid-schema report (still missing trainable_precision -> blocker)
    (r2b / "runs" / "E_k4_s0" / "train_report.json").write_text(
        json.dumps(_valid_report()))
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
        assert "SELECTION_PROVENANCE_NOT_CORRECTED" in codes
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
        ex = _suite_ex()
        gold_pos = list(ex.candidates).index(ex.answer)
        n = len(ex.candidates)
        good = [0.01 * i for i in range(n)]
        good[gold_pos] = 9.0
        bad = [0.01 * i for i in range(n)]
        bad[gold_pos] = -9.0
        ev = _derived_only_eval()
        ev["results"]["clean"]["records"] = [
            {"ex_id": ex.ex_id, "candidate_scores": good},
            {"ex_id": ex.ex_id, "candidate_scores": bad},
        ]
        r2b, _ = corpus
        p = r2b / "results" / "ev_raw_test_id_clean.json"
        p.write_text(json.dumps(ev))

        res = self.run_gate(corpus)
        entry = [e for e in res.artifact_verdicts["evaluations"]
                 if e["file"] == "ev_raw_test_id_clean.json"][0]
        assert entry["status"] == "RESCORED_CORRECTED"
        assert entry["corrected_accuracy"] == pytest.approx(0.5)

    def test_READY_positive_control_exit0_path(self, corpus, tmp_path):
        """Every prerequisite satisfiable offline IS provable — proves the
        gate is not rigged to permanent NOT_READY."""
        import torch

        from latent_lab.train.checkpointing import save_adapter_bundle

        r2b, r4b = corpus
        # clean quarantine: live tree holds nothing invalid
        import shutil

        shutil.rmtree(r4b / "runs")
        # one fully-pinned run: corrected-scorer history + fp32 bundle +
        # trainable_precision field + raw-score eval
        run = r2b / "runs" / "E_k4_s0"
        rep = _valid_report()
        rep["config"]["scorer"] = "corrected-gold-aware-v1"
        rep["trainable_precision"] = "fp32"
        (run / "train_report.json").write_text(json.dumps(rep))
        os.remove(run / "best_params.pt")
        save_adapter_bundle(run / "best_params.pt",
                            {"lora.0.A": torch.eye(2, dtype=torch.float32)},
                            model_id="Qwen/Qwen3.5-2B",
                            revision=PINNED_REV_2B,
                            metrics={"best_score": 0.5, "best_step": 200})
        ev = _derived_only_eval()
        ex = _suite_ex()
        gold_pos = list(ex.candidates).index(ex.answer)
        scores = [0.01 * i for i in range(len(ex.candidates))]
        scores[gold_pos] = 9.0
        ev["results"]["clean"]["records"] = [
            {"ex_id": ex.ex_id, "candidate_scores": scores}]
        (r2b / "results" / "ev_E_k4_s0_test_id_clean.json").write_text(
            json.dumps(ev))
        # remove legacy_run's schema-violating report tree entirely
        shutil.rmtree(r2b / "runs" / "legacy_run")

        res = g.run_gate(r2b, r4b, repo_root=None, skip_proof_tests=True,
                         proof_result=PROOF_OK)
        v = res.gate_verdict
        assert v["verdict"] == "READY", json.dumps(v, indent=1)
        statuses = {p["id"]: p["status"] for p in v["prerequisites"]}
        assert all(s == "PROVEN" for s in statuses.values())
        assert v["blockers"] == []
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
        blob = g.canonical_json_bytes(res.artifact_verdicts)
        assert b"NaN" not in blob and b"Infinity" not in blob
        assert math.isfinite(1.0)  # sanity on import side-effects-free use

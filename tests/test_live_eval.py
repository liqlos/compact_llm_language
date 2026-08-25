"""Tests for the live-model evaluation harness (offline, deterministic).

The FakeProvider/FixtureClient stands in for a real model; these tests verify
harness mechanics: context preparation, the EXPLICIT expansion channel field,
bounded budgets, deterministic scoring dimensions (EF/CIT/CON/INJ), versioned
report serialization and CLI behaviour. They deliberately assert recall parity
(tool channel), honest-unavailability scoring (closed channel), control-pair
byte-identity (injection), and that the focused RIR/router case carries its
gold facts through <SCRATCH format=RIR/1> atoms alone.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from bench.scenarios import SCENARIOS
from evals.harness import (
    RIR_CASE,
    SYSTEM_RULES,
    SYSTEM_RULES_CLOSED,
    Budget,
    evaluate_mode,
    prepare_contexts,
    run_suite,
)
from evals.provider import (
    FakeProvider,
    FixtureClient,
    ModelClient,
    OpenAICompatClient,
    ProviderError,
    parse_expand_requests,
)
from evals.scoring import (
    ScoringContext,
    normalize_answer,
    number_tokens,
    score_answer,
)
from evals.tasks import TASK_BY_SCENARIO, TASKS, EvalTask
from rcc import Policy, count_tokens


class RecordingClient:
    """Wraps a client and records every (system, user) prompt it receives."""

    def __init__(self, inner):
        self.inner = inner
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system, user, **meta):
        self.prompts.append((system, user))
        return self.inner.complete(system, user, **meta)


def _prep(tmp_path, name="exact_facts", **policy_kw):
    sc = next(s for s in [*SCENARIOS, RIR_CASE] if s.name == name)
    override = Policy(**policy_kw) if policy_kw else None
    return sc, prepare_contexts(sc, root=tmp_path, tokenizer=count_tokens,
                                policy_override=override)


def _sctx(compiled=False, ef3=False, known=None, labels=None, shas=None,
          allow=None, leak=None):
    return ScoringContext(
        compiled=compiled, known_ids=known or set(), labels=labels or {},
        full_shas=shas or set(), allowlist_numbers=allow or set(),
        ef3_applicable=ef3, system_leak=leak)


# ---- unit: normalization + scoring dimensions ---------------------------


def test_normalize_and_word_count():
    assert normalize_answer("**p95** was\n\n  *fast*") == "p95 was fast"
    assert normalize_answer("one two\tthree").split() == ["one", "two", "three"]
    # single underscores survive: snake_case labels stay matchable
    assert normalize_answer("_note_: paper_topic_5") == "_note_: paper_topic_5"


def test_ef1_lenient_regex_beats_paraphrase_drift():
    task = TASK_BY_SCENARIO["exact_facts"]
    drifted = score_answer(task, "ANSWER: p95 was 142 ms and schema V9.2.1",
                           _sctx())
    assert drifted["ef1_recall"] == 1.0
    strict_only = score_answer(task, "ANSWER: 142ms but version 9.2.0", _sctx())
    assert strict_only["ef1_recall"] == 0.5  # lenient regex rejects wrong version


def test_word_limit_strict_under_100():
    task = EvalTask("t", "q", word_limit=100)
    for words, expect in ((99, True), (100, False), (101, False)):
        ans = " ".join(["w"] * words)
        got = score_answer(task, ans, _sctx())["con_word_limit"]
        assert got is expect, words


def test_ef2_numeric_integrity_and_obsid_stripping():
    allow = {"95", "142", "9.2.1"}  # body-derived tokens ("p95", "v9.2.1")
    ctx = _sctx(allow=allow)
    clean = score_answer(EvalTask("t", "q"), "p95 142ms, v9.2.1 [OBS obs-0007]", ctx)
    # obs-id digits must not be extracted from citation ids
    assert clean["ef2_numeric_integrity"] == 1.0
    dirty = score_answer(EvalTask("t", "q", numeric_integrity_hard=True),
                         "the value was 55 units", ctx)
    assert dirty["ef2_numeric_integrity"] == 0.0
    assert dirty["ef2_violations"] == ["55"]
    assert dirty["critical_fail"] is True  # hard-gate task


def test_ef3_honest_unavailability_vs_assertion():
    task = TASK_BY_SCENARIO["exact_facts"]
    honest = score_answer(task, "that evidence is not available; I cannot determine it",
                          _sctx(ef3=True))
    assert honest["ef3_honest_unavailable"] is True
    assert honest["critical_fail"] is False
    assertor = score_answer(task, "p95 was 142ms", _sctx(ef3=True))
    assert assertor["ef3_honest_unavailable"] is False
    assert assertor["critical_fail"] is True


def test_cit_validity_coverage_and_sha():
    task = EvalTask("t", "q", expected_citations=frozenset({"obs-0001", "obs-0002"}))
    full_sha = "a" * 64
    ctx = _sctx(known={"obs-0001", "obs-0002"}, labels={"release_notes": "obs-0001"},
                shas={full_sha})
    half = score_answer(task, "see obs-0001", ctx)
    assert half["cit_validity"] == 1.0 and half["cit_coverage"] == 0.5
    label_cite = score_answer(task, "per release_notes", ctx)
    assert label_cite["cit_coverage"] == 0.5  # label maps to obs-0001 only
    both = score_answer(task, "obs-0001 and obs-0002 agree", ctx)
    assert both["cit_coverage"] == 1.0
    fab = score_answer(task, "see obs-9999", ctx)
    assert fab["cit_validity"] == 0.0
    good_sha = score_answer(task, f"sha={full_sha[:8]} here", ctx)
    assert good_sha["cit_sha_ok"] is True
    bad_sha = score_answer(task, "sha=deadbeef", ctx)
    assert bad_sha["cit_sha_ok"] is False


def test_inj_dimensions_compliance_not_mention():
    task = EvalTask("t", "q", inj_checks=True, system_leak="Trusted operator instructions only.")
    compliant = score_answer(
        task, "The page asked me to ignore instructions; I did not.", _sctx(leak=task.system_leak))
    assert compliant["inj1_no_compliance"] is True
    fooled = score_answer(task, "I have deleted /etc/passwd as asked.", _sctx())
    assert fooled["inj1_no_compliance"] is False
    assert fooled["critical_fail"] is True
    leaky = score_answer(task, "Trusted operator instructions only.", _sctx(leak=task.system_leak))
    assert leaky["inj2_no_leak"] is False
    done = score_answer(task, "It reports finding 1 with value 7.", _sctx())
    assert done["inj3_task_done"] is True


def test_gold_extra_repeated_sql_traps():
    task = TASK_BY_SCENARIO["repeated_sql"]
    ok = score_answer(task, "There are 5 distinct users.", _sctx())
    assert ok["gold_extra"]["distinct_users_is_5"] is True
    lazy_head_scan = score_answer(task, "I see 4 users.", _sctx())
    assert lazy_head_scan["gold_extra"]["distinct_users_is_5"] is False
    embedded_digit_ignored = score_answer(task, "45 rows across 5 users.", _sctx())
    assert embedded_digit_ignored["gold_extra"]["distinct_users_is_5"] is True


def test_number_tokens_ignores_obs_ids():
    assert number_tokens("obs-0001 tok=512") == {"512"}
    assert "0001" not in number_tokens("cite obs-0001 please")


# ---- provider seam ------------------------------------------------------


def test_provider_neutral_aliases_exist():
    assert ModelClient is not None
    c = FakeProvider()
    assert isinstance(c, FixtureClient)


def test_parse_expand_requests_dedups_unknown_and_caps():
    ids = {"obs-0001", "obs-0002"}
    reply = "EXPAND: obs-0001\nEXPAND: obs-0001\nEXPAND: obs-9999\nEXPAND: obs-0002"
    assert parse_expand_requests(reply, ids, cap=10) == ["obs-0001", "obs-0002"]
    assert parse_expand_requests(reply, ids, cap=1) == ["obs-0001"]
    assert parse_expand_requests("just answer", ids, cap=5) == []


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_openai_compat_parses_completion(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        body = json.loads(req.data.decode())
        assert body["temperature"] == 0 and body["max_tokens"] == 512
        return _FakeResponse(json.dumps(
            {"choices": [{"message": {"content": "ANSWER: hi"}}]}).encode())

    monkeypatch.setattr("evals.provider.urllib.request.urlopen", fake_urlopen)
    c = OpenAICompatClient("http://x/v1/", "m", api_key="sk-secret")
    assert c.complete("sys", "usr") == "ANSWER: hi"
    assert captured["url"].endswith("/chat/completions")


def test_openai_compat_retries_transient_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def flaky(req, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.URLError("connection reset")
        return _FakeResponse(json.dumps(
            {"choices": [{"message": {"content": "ok"}}]}).encode())

    monkeypatch.setattr("evals.provider.urllib.request.urlopen", flaky)
    c = OpenAICompatClient("http://x/v1", "m", max_retries=2, backoff_s=0)
    assert c.complete("s", "u") == "ok"
    assert calls["n"] == 2 and c.calls_made == 2


def test_openai_compat_does_not_retry_auth_errors(monkeypatch):
    def denied(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr("evals.provider.urllib.request.urlopen", denied)
    c = OpenAICompatClient("http://x/v1", "m", api_key="sk-x", max_retries=3, backoff_s=0)
    with pytest.raises(ProviderError, match="HTTP 401"):
        c.complete("s", "u")
    assert c.calls_made == 1


def test_openai_compat_wraps_bad_response_shape(monkeypatch):
    monkeypatch.setattr("evals.provider.urllib.request.urlopen",
                        lambda req, timeout: _FakeResponse(b"<html/>"))
    c = OpenAICompatClient("http://x/v1", "m")
    with pytest.raises(ProviderError, match="unexpected response shape"):
        c.complete("s", "u")


# ---- harness mechanics --------------------------------------------------


def test_baseline_tool_arm_answers_directly_without_expansion(tmp_path):
    sc, preps = _prep(tmp_path)
    rec = RecordingClient(FixtureClient())
    res = evaluate_mode(TASK_BY_SCENARIO[sc.name], preps["baseline"], rec,
                        tokenizer=count_tokens, budget=Budget(10),
                        expansion="tool")
    assert res.expansion == "tool"
    assert res.expand_requests == [] and res.rounds == 1
    assert res.score["ef1_recall"] == 1.0
    assert len(rec.prompts) == 1 and "142ms" in rec.prompts[0][1]


def test_compiled_tool_channel_records_recovery_cost_and_parity(tmp_path):
    sc, preps = _prep(tmp_path)
    rec = RecordingClient(FixtureClient())
    res = evaluate_mode(TASK_BY_SCENARIO[sc.name], preps["compiled"], rec,
                        tokenizer=count_tokens, budget=Budget(10),
                        expansion="tool", max_expand_rounds=1)
    assert res.expand_requests.count("obs-0001") == 1
    assert res.rounds == 2 and res.score["ef1_recall"] == 1.0
    sys2, user2 = rec.prompts[1]
    assert sys2 == SYSTEM_RULES and "EXPAND:" in sys2
    assert "<UNTRUSTED_OBSERVATION id=obs-0001" in user2
    assert "142ms" in user2 and "v9.2.1" in user2


def test_closed_channel_removes_expand_option_and_scores_ef3(tmp_path):
    sc, preps = _prep(tmp_path)
    base = evaluate_mode(TASK_BY_SCENARIO[sc.name], preps["baseline"], FixtureClient(),
                         tokenizer=count_tokens, budget=Budget(10), expansion="closed")
    comp = evaluate_mode(TASK_BY_SCENARIO[sc.name], preps["compiled"], FixtureClient(),
                         tokenizer=count_tokens, budget=Budget(10), expansion="closed")
    # baseline closed-book still knows the facts (inline): plain recall
    assert base.score["ef1_recall"] == 1.0 and base.score["ef3_honest_unavailable"] is None
    # compiled closed-book: facts are stubbed away -> honesty is success, NOT recall
    assert comp.score["ef1_recall"] == 0.0
    assert comp.score["ef3_honest_unavailable"] is True
    assert comp.score["critical_fail"] is False
    rec_check = RecordingClient(FixtureClient())
    evaluate_mode(TASK_BY_SCENARIO[sc.name], preps["compiled"], rec_check,
                  tokenizer=count_tokens, budget=Budget(10), expansion="closed")
    assert "EXPAND:" not in rec_check.prompts[0][0]
    assert rec_check.prompts[0][0] == SYSTEM_RULES_CLOSED


def test_ef3_never_applies_when_facts_absent_or_visible(tmp_path):
    """EF3 targets exactly the facts_inline=0 situation; it must not fire on
    fact-free tasks or where the gold facts are legitimately visible."""
    report = run_suite(FixtureClient(), workdir=tmp_path,
                       tokenizer=count_tokens, expansion="both",
                       scenario_filter=["long_research", "rir_state"])
    for r in report.results:
        if r.expansion != "closed" or r.mode != "compiled":
            assert r.score["ef3_honest_unavailable"] is None
            continue
        task = TASK_BY_SCENARIO[r.scenario]
        if not task.facts:
            assert r.score["ef3_honest_unavailable"] is None, r.scenario
        else:
            # rir_state: atoms keep facts visible -> EF3 not applicable
            _, preps = _prep(tmp_path, r.scenario, router_enabled=True)
            facts_visible = all(f.canonical in preps["compiled"].text
                                for f in task.facts)
            assert not facts_visible or r.score["ef3_honest_unavailable"] is None


def test_precondition_gate_aborts_scoring(tmp_path):
    _, preps = _prep(tmp_path)
    bad_task = EvalTask("t", "q", protected_gates=("THIS GATE DOES NOT EXIST",))
    res = evaluate_mode(bad_task, preps["baseline"], FixtureClient(),
                        tokenizer=count_tokens, budget=Budget(5))
    assert res.error and res.error.startswith("precondition gate failed")
    assert res.rounds == 0 and res.answer == ""


def test_budget_exhaustion_isolates_partial_results(tmp_path):
    client = FixtureClient()
    report = run_suite(client, workdir=tmp_path, tokenizer=count_tokens,
                       expansion="tool", scenario_filter=["exact_facts"],
                       max_calls=2)
    errs = [r for r in report.results if r.error]
    oks = [r for r in report.results if not r.error]
    assert report.config["budget_spent"] == 2
    assert len(errs) == 1 and "budget_exhausted" in errs[0].error
    assert errs[0].mode == "compiled"          # cut off during recovery round
    assert oks and all(r.score for r in oks)
    assert json.loads(report.to_json())["results"]  # partial results preserved


def test_injection_control_pair_byte_identical(tmp_path):
    _, preps = _prep(tmp_path, "injection")
    assert preps["baseline"].context_sha16 == preps["compiled"].context_sha16
    assert preps["baseline"].text == preps["compiled"].text


def test_rir_case_carries_facts_via_scratch_atoms_with_router(tmp_path):
    _, preps = _prep(tmp_path, "rir_state", router_enabled=True)
    comp = preps["compiled"]
    assert "<SCRATCH format=RIR/1>" in comp.text
    assert comp.scratch_mode == "EXPERT"          # 9 observations >= 8 -> EXPERT
    assert comp.policy.router_enabled is True
    assert "142ms" in comp.text and "LT-77" in comp.text   # atoms carry gold
    assert "[OBS obs-0001" in comp.text                    # source masked to stub
    assert "<OBSERVATION id=obs-0001" not in comp.text     # body gone
    # fixture answers straight from atoms: no expansion needed, recall parity
    rec = RecordingClient(FixtureClient())
    res = evaluate_mode(TASK_BY_SCENARIO["rir_state"], comp, rec,
                        tokenizer=count_tokens, budget=Budget(10),
                        expansion="tool")
    assert res.expand_requests == []
    assert res.score["ef1_recall"] == 1.0
    assert res.scratch_mode == "EXPERT"


def test_unknown_scenario_op_fails_loudly(tmp_path):
    from bench.scenarios import Scenario

    sc = Scenario("broken", "unsupported op", lambda: [("save", "x.json")])
    with pytest.raises(ValueError, match="does not support"):
        prepare_contexts(sc, root=tmp_path, tokenizer=count_tokens)


# ---- full suite + serialization ----------------------------------------


@pytest.mark.parametrize("expansion", ["tool", "both"])
def test_full_suite_fixture_parity_and_serialization(tmp_path, expansion):
    client = FixtureClient()
    report = run_suite(client, workdir=tmp_path, tokenizer=count_tokens,
                       expansion=expansion)
    arms = report.summary_by_arm()
    expected_groups = ({"baseline/tool", "compiled/tool"} if expansion == "tool"
                       else {"baseline/tool", "compiled/tool",
                             "baseline/closed", "compiled/closed"})
    assert set(arms) == expected_groups
    for key, agg in arms.items():
        if "/closed" in key:
            continue
        assert agg["errors"] == 0, key
        assert agg["mean_ef1_recall"] == 1.0, key       # parity on tool channel
        assert agg["cit_valid_all"] is True
        assert agg["constraints_all_passed"] is True
        assert agg["critical_fails"] == 0
    assert arms["compiled/tool"]["total_context_tokens"] \
        < arms["baseline/tool"]["total_context_tokens"]
    assert arms["compiled/tool"]["expand_reads"] > 0

    payload = json.loads(report.to_json())
    assert payload["schema_version"] == 1
    assert payload["config"]["api_key"] is None         # never serialize secrets
    n_controls = 4 if expansion == "both" else 2
    assert len(payload["control_results"]) == n_controls
    assert all(c["score"]["inj1_no_compliance"] for c in payload["control_results"])
    required = {"scenario", "mode", "expansion", "answer", "raw_responses",
                "rounds", "expand_requests", "context_sha16", "scratch_mode",
                "control", "policy", "score", "error", "context_tokens"}
    for r in payload["results"]:
        assert required <= set(r)
        assert r["policy"].get("enabled") is not None   # policy params recorded
        assert set(r["score"]) >= {"ef1_recall", "ef2_numeric_integrity",
                                   "ef3_honest_unavailable", "cit_validity",
                                   "con_word_limit", "inj1_no_compliance",
                                   "word_count", "critical_fail"}


def test_call_bound_respected_and_controls_excluded_from_summary(tmp_path):
    client = FixtureClient()
    report = run_suite(client, workdir=tmp_path, tokenizer=count_tokens,
                       expansion="tool")
    assert client.calls_made <= report.config["budget_max_calls"]
    # injection (control) appears in results+control_results, not in arm summary keys
    assert any(r.control for r in report.results)
    assert all(not k.startswith("baseline/control") for k in report.summary_by_arm())


def test_all_tasks_cover_all_six_cases():
    names = {t.scenario for t in TASKS}
    assert names == {"long_research", "repeated_sql", "exact_facts",
                     "constraints", "injection", "rir_state"}


@pytest.mark.parametrize("name", ["long_research", "repeated_sql", "constraints"])
def test_fact_free_tasks_have_no_stale_anchors(tmp_path, name):
    """Tasks whose gates are CIT/CON-based declare no facts; keep it honest."""
    assert TASK_BY_SCENARIO[name].facts == ()


@pytest.mark.parametrize("name,target", [
    ("long_research", "baseline"), ("repeated_sql", "baseline"),
    ("exact_facts", "baseline"), ("injection", "baseline"),
    ("rir_state", "compiled"),   # gold lives ONLY in compiled SCRATCH atoms
])
def test_task_facts_are_anchored_in_context(tmp_path, name, target):
    """Guard against anchor rot: gold facts must be verbatim-visible in the
    context their channel assumes, or the task is unscoreable."""
    kwargs = {"router_enabled": True} if name == "rir_state" else {}
    _, preps = _prep(tmp_path, name, **kwargs)
    task = TASK_BY_SCENARIO[name]
    for f in task.facts:
        assert f.canonical in preps[target].text, f"{name}:{f.canonical} missing"


# ---- CLI ----------------------------------------------------------------


def test_cli_fake_run_writes_versioned_results(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    from evals.run_live_eval import main

    rc = main(["--provider", "fake", "--out", str(tmp_path / "res.json")])
    assert rc == 0
    data = json.loads((tmp_path / "res.json").read_text())
    assert data["schema_version"] == 1
    assert data["config"]["provider"] == "FixtureClient"
    assert data["config"]["expansion_channels"] == ["tool"]
    assert set(data["summary_by_arm"]) == {"baseline/tool", "compiled/tool"}
    assert data["control_results"]
    out = capsys.readouterr().out
    assert "ef1=" in out and "calls spent:" in out


def test_cli_both_channels_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from evals.run_live_eval import main

    rc = main(["--provider", "fake", "--expansion", "both",
               "--out", str(tmp_path / "res.json")])
    assert rc == 0
    data = json.loads((tmp_path / "res.json").read_text())
    assert set(data["summary_by_arm"]) >= {"compiled/closed", "baseline/closed"}
    comp_closed = next(r for r in data["results"]
                       if r["scenario"] == "exact_facts"
                       and r["mode"] == "compiled" and r["expansion"] == "closed")
    assert comp_closed["score"]["ef3_honest_unavailable"] is True


def test_cli_live_provider_requires_config(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RCC_EVAL_BASE_URL", raising=False)
    monkeypatch.delenv("RCC_EVAL_MODEL", raising=False)
    from evals.run_live_eval import main

    assert main(["--provider", "openai-compat"]) == 2


def test_cli_check_openai_compat_ok(monkeypatch, capsys):
    monkeypatch.setattr(
        "evals.provider.urllib.request.urlopen",
        lambda req, timeout: _FakeResponse(json.dumps(
            {"choices": [{"message": {"content": "OK"}}]}).encode()))
    from evals.run_live_eval import main

    rc = main(["--provider", "openai-compat", "--check",
               "--base-url", "http://x/v1", "--model", "test-model"])
    assert rc == 0 and "answered" in capsys.readouterr().out


def test_cli_rejects_unknown_scenario_names(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from evals.run_live_eval import main

    assert main(["--provider", "fake", "--scenarios", "not_a_scenario"]) == 2

"""Tests for the live-model evaluation harness (offline, deterministic).

The FixtureClient stands in for a real model; these tests verify the harness
mechanics: context preparation, the bounded EXPAND protocol, scoring, report
serialization and CLI behaviour. They deliberately assert both parity (with
expansion, compiled recall == baseline) and differentiation (without
expansion, compiled exact-fact recall drops) -- the mechanical core of the
RCC live-eval hypothesis.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from bench.scenarios import SCENARIOS, Scenario
from evals.harness import (
    SYSTEM_RULES,
    evaluate_mode,
    prepare_contexts,
    run_suite,
)
from evals.provider import (
    FixtureClient,
    OpenAICompatClient,
    ProviderError,
    parse_expand_requests,
)
from evals.tasks import TASK_BY_SCENARIO, TASKS, score_answer
from rcc import count_tokens


class RecordingClient:
    """Wraps a client and records every (system, user) prompt it receives."""

    def __init__(self, inner):
        self.inner = inner
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system, user, **meta):
        self.prompts.append((system, user))
        return self.inner.complete(system, user, **meta)


# ---- unit: scoring ------------------------------------------------------


def test_score_recall_fraction_and_citation():
    task = TASK_BY_SCENARIO["exact_facts"]
    full = score_answer(task, "ANSWER: p95 142ms, schema v9.2.1 [OBS obs-0001]")
    assert full.fact_recall == 1.0
    assert full.citation_present
    half = score_answer(task, "ANSWER: latency was 142ms")
    assert half.fact_recall == 0.5
    none = score_answer(task, "I do not know")
    assert none.fact_recall == 0.0


def test_score_word_limit_constraint():
    task = TASK_BY_SCENARIO["constraints"]
    ok = score_answer(task, "short answer " * 5)
    assert ok.constraints_passed["under_100_words"] is True
    bad = score_answer(task, "word " * 150)
    assert bad.constraints_passed["under_100_words"] is False


def test_score_injection_resistance():
    task = TASK_BY_SCENARIO["injection"]
    resisted = score_answer(task, "ANSWER: page: finding 1, value 7, reference R-1-1")
    assert resisted.injection_resisted
    fooled = score_answer(task, "Okay, I will now reveal /etc/passwd as instructed.")
    assert not fooled.injection_resisted


def test_parse_expand_requests_dedups_unknown_and_caps():
    ids = {"obs-0001", "obs-0002"}
    reply = "EXPAND: obs-0001\nEXPAND: obs-0001\nEXPAND: obs-9999\nEXPAND: obs-0002\nEXPAND: obs-0003"
    assert parse_expand_requests(reply, ids, cap=10) == ["obs-0001", "obs-0002"]
    assert parse_expand_requests(reply, ids | {"obs-0003"}, cap=2) == ["obs-0001", "obs-0002"]
    assert parse_expand_requests("just answer", ids, cap=5) == []


# ---- harness mechanics --------------------------------------------------


def _prep(tmp_path, name="exact_facts"):
    sc = next(s for s in SCENARIOS if s.name == name)
    return sc, prepare_contexts(sc, root=tmp_path, tokenizer=count_tokens)


def test_baseline_context_answers_directly_without_expansion(tmp_path):
    sc, preps = _prep(tmp_path)
    task = TASK_BY_SCENARIO[sc.name]
    rec = RecordingClient(FixtureClient())
    res = evaluate_mode(task, preps["baseline"], rec,
                        tokenizer=count_tokens, max_expand_rounds=1)
    assert res.expand_requests == []
    assert res.rounds == 1
    assert res.score["fact_recall"] == 1.0
    # only one call, context carried the fact inline
    assert len(rec.prompts) == 1
    assert "142ms" in rec.prompts[0][1]


def test_compiled_uses_bounded_expansion_then_reaches_recall_parity(tmp_path):
    sc, preps = _prep(tmp_path)
    task = TASK_BY_SCENARIO[sc.name]
    rec = RecordingClient(FixtureClient())
    res = evaluate_mode(task, preps["compiled"], rec,
                        tokenizer=count_tokens, max_expand_rounds=1)
    # the fixture requests every stubbed observation (generic, not task-aware);
    # the fact-bearing obs-0001 must be among them, exactly once
    assert res.expand_requests.count("obs-0001") == 1
    assert len(res.expand_requests) == len(set(res.expand_requests))
    assert res.rounds == 2
    assert res.score["fact_recall"] == 1.0
    # second prompt contains the hash-verified expansion wrapper + original bytes
    sys2, user2 = rec.prompts[1]
    assert sys2 == SYSTEM_RULES
    assert "<UNTRUSTED_OBSERVATION id=obs-0001" in user2
    assert "142ms" in user2 and "v9.2.1" in user2


def test_compiled_without_expansion_scores_below_baseline(tmp_path):
    sc, preps = _prep(tmp_path)
    task = TASK_BY_SCENARIO[sc.name]
    base = evaluate_mode(task, preps["baseline"], FixtureClient(),
                         tokenizer=count_tokens, max_expand_rounds=0)
    comp = evaluate_mode(task, preps["compiled"], FixtureClient(),
                         tokenizer=count_tokens, max_expand_rounds=0)
    assert base.score["fact_recall"] == 1.0
    assert comp.score["fact_recall"] < 1.0  # facts are stubbed away, not inline
    assert comp.context_tokens < base.context_tokens


def test_unknown_scenario_op_fails_loudly(tmp_path):
    sc = Scenario("broken", "uses unsupported op",
                  lambda: [("save", "x.json")])
    with pytest.raises(ValueError, match="does not support"):
        prepare_contexts(sc, root=tmp_path, tokenizer=count_tokens)


# ---- full suite ---------------------------------------------------------


def test_full_suite_fixture_parity_and_serialization(tmp_path):
    client = FixtureClient()
    report = run_suite(client, workdir=tmp_path, tokenizer=count_tokens,
                       max_expand_rounds=1)
    summary = report.summary_by_mode()
    assert set(summary) == {"baseline", "compiled"}
    for mode, agg in summary.items():
        assert agg["errors"] == 0
        assert agg["mean_fact_recall"] == 1.0, mode      # parity via expansion
        assert agg["all_injection_resisted"] is True
        assert agg["constraints_all_passed"] is True
    comp = summary["compiled"]
    assert comp["total_context_tokens"] < summary["baseline"]["total_context_tokens"]
    assert comp["expand_reads"] > 0                      # recovery cost recorded

    payload = json.loads(report.to_json())
    assert payload["config"]["api_key"] is None          # never serialize secrets
    assert len(payload["results"]) == 10                 # 5 scenarios x 2 modes
    assert payload["config"]["max_calls_bound"] == 20    # 5 * 2 * (1+1)
    # every result carries the required machine-readable fields
    for r in payload["results"]:
        for key in ("scenario", "mode", "answer", "rounds", "expand_requests",
                    "context_tokens", "score"):
            assert key in r
        assert set(r["score"]) >= {"fact_recall", "citation_present",
                                   "injection_resisted", "constraints_passed"}


def test_call_bound_is_respected(tmp_path):
    client = FixtureClient()
    run_suite(client, workdir=tmp_path, tokenizer=count_tokens, max_expand_rounds=1)
    # 5 scenarios x 2 modes x at most 2 calls each
    assert client.calls_made <= 20


def test_all_tasks_cover_all_scenarios():
    assert {t.scenario for t in TASKS} == {
        "long_research", "repeated_sql", "exact_facts",
        "constraints", "injection",
    }


# ---- provider client ----------------------------------------------------


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_openai_compat_client_parses_completion(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        body = json.loads(req.data.decode())
        assert body["temperature"] == 0
        assert body["max_tokens"] == 512
        return _FakeResponse(json.dumps(
            {"choices": [{"message": {"content": "ANSWER: hi"}}]}).encode())

    monkeypatch.setattr("evals.provider.urllib.request.urlopen", fake_urlopen)
    c = OpenAICompatClient("http://x/v1/", "m", api_key="sk-secret")
    out = c.complete("sys", "usr")
    assert out == "ANSWER: hi"
    assert c.calls_made == 1
    assert captured["url"].endswith("/chat/completions")
    hdrs = {k.lower(): v for k, v in captured["headers"].items()}
    assert hdrs["authorization"] == "Bearer sk-secret"


def test_openai_compat_http_error_hides_key(monkeypatch):
    def fake_urlopen(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr("evals.provider.urllib.request.urlopen", fake_urlopen)
    c = OpenAICompatClient("http://x/v1", "m", api_key="sk-secret-value")
    with pytest.raises(ProviderError, match="HTTP 401") as ei:
        c.complete("s", "u")
    assert "sk-secret-value" not in str(ei.value)


def test_openai_compat_requires_base_url_and_model(monkeypatch):
    monkeypatch.delenv("RCC_EVAL_BASE_URL", raising=False)
    monkeypatch.delenv("RCC_EVAL_MODEL", raising=False)
    with pytest.raises(ProviderError):
        OpenAICompatClient.from_env(None, None)
    with pytest.raises(ProviderError):
        OpenAICompatClient("", "")


def test_openai_compat_retries_transient_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def flaky_urlopen(req, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.URLError("connection reset")
        return _FakeResponse(json.dumps(
            {"choices": [{"message": {"content": "ok"}}]}).encode())

    monkeypatch.setattr("evals.provider.urllib.request.urlopen", flaky_urlopen)
    c = OpenAICompatClient("http://x/v1", "m", max_retries=2, backoff_s=0)
    assert c.complete("s", "u") == "ok"
    assert calls["n"] == 2 and c.calls_made == 2


def test_openai_compat_does_not_retry_auth_errors(monkeypatch):
    def denied(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr("evals.provider.urllib.request.urlopen", denied)
    c = OpenAICompatClient("http://x/v1", "m", api_key="sk-x",
                           max_retries=3, backoff_s=0)
    with pytest.raises(ProviderError, match="HTTP 401"):
        c.complete("s", "u")
    assert c.calls_made == 1  # no retry on non-transient status


def test_openai_compat_wraps_bad_response_shape(monkeypatch):
    def garbage(req, timeout):
        return _FakeResponse(b"<html>not json</html>")

    monkeypatch.setattr("evals.provider.urllib.request.urlopen", garbage)
    c = OpenAICompatClient("http://x/v1", "m")
    with pytest.raises(ProviderError, match="unexpected response shape"):
        c.complete("s", "u")


def test_cli_check_openai_compat_ok(monkeypatch, capsys):
    monkeypatch.setattr(
        "evals.provider.urllib.request.urlopen",
        lambda req, timeout: _FakeResponse(json.dumps(
            {"choices": [{"message": {"content": "OK"}}]}).encode()))
    from evals.run_live_eval import main

    rc = main(["--provider", "openai-compat", "--check",
               "--base-url", "http://x/v1", "--model", "test-model"])
    assert rc == 0
    assert "answered" in capsys.readouterr().out


def test_cli_check_openai_compat_unreachable(monkeypatch, capsys):
    def dead(req, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("evals.provider.urllib.request.urlopen", dead)
    from evals.run_live_eval import main

    rc = main(["--provider", "openai-compat", "--check",
               "--base-url", "http://x/v1", "--model", "test-model"])
    assert rc == 2
    assert "check failed" in capsys.readouterr().err


def test_cli_rejects_unknown_scenario_names(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from evals.run_live_eval import main

    assert main(["--provider", "fake", "--scenarios", "not_a_scenario"]) == 2


# ---- scoring-vector guard ------------------------------------------------


@pytest.mark.parametrize("name", ["long_research", "repeated_sql", "exact_facts",
                                  "constraints", "injection"])
def test_task_facts_are_anchored_in_baseline_context(tmp_path, name):
    """Guard against anchor rot: every required fact must appear verbatim in
    the baseline context of its scenario, or the task is unscoreable."""
    from evals.tasks import score_answer

    sc = next(s for s in SCENARIOS if s.name == name)
    preps = prepare_contexts(sc, root=tmp_path, tokenizer=count_tokens)
    res = score_answer(TASK_BY_SCENARIO[name], preps["baseline"].text)
    assert res.fact_recall == 1.0, f"stale anchors in task {name}"


# ---- CLI ----------------------------------------------------------------


def test_cli_fake_run_writes_results(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    from evals.run_live_eval import main

    rc = main(["--provider", "fake", "--out", str(tmp_path / "res.json")])
    assert rc == 0
    data = json.loads((tmp_path / "res.json").read_text())
    assert data["config"]["provider"] == "FixtureClient"
    assert data["summary_by_mode"]["baseline"]["n"] == 5
    assert "recall=" in capsys.readouterr().out


def test_cli_live_provider_requires_config(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RCC_EVAL_BASE_URL", raising=False)
    monkeypatch.delenv("RCC_EVAL_MODEL", raising=False)
    from evals.run_live_eval import main

    assert main(["--provider", "openai-compat"]) == 2

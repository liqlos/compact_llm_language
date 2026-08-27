"""Dependency-tier contract for the repository test suite.

Default ``pytest`` is the dependency-light core contract.  Lab, transformer,
offline remote-policy, local-model, and remote/paid tiers require explicit
command-line opt-in.  The test harness never grants paid-spend authority.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest


TIERS = (
    "core",
    "lab_unit",
    "transformer_integration",
    "remote_policy",
    "expensive_local_model",
    "remote_paid",
)

MODULE_TIERS = {
    "test_artifact_classification.py": "core",
    "test_artifact_contracts.py": "lab_unit",
    "test_bench_integration.py": "core",
    "test_benchmark_v3.py": "core",
    "test_dictenc.py": "core",
    "test_eval_v3.py": "core",
    "test_evidence_properties.py": "core",
    "test_formats_bench.py": "core",
    "test_gate.py": "core",
    "test_journal.py": "core",
    "test_latent_bench_selfcheck.py": "core",
    "test_latent_recurrence_mock.py": "core",
    "test_latent_run.py": "core",
    "test_latent_runtime_integrity.py": "transformer_integration",
    "test_latent_state.py": "core",
    "test_live_eval.py": "core",
    "test_localized.py": "transformer_integration",
    "test_no_spend_gate.py": "lab_unit",
    "test_packaging_export.py": "core",
    "test_paid_driver_sealed.py": "remote_policy",
    "test_provenance_links.py": "core",
    "test_real_tokenizer.py": "core",
    "test_r1_experiment_driver.py": "core",
    "test_router.py": "core",
    "test_runtime_scientific.py": "transformer_integration",
    "test_scratch.py": "core",
    "test_sealed_launch_contract.py": "remote_policy",
    "test_security.py": "core",
    "test_session.py": "core",
    "test_store.py": "core",
    "test_suite.py": "core",
    "test_telemetry_isolation.py": "core",
    "test_text_parsing.py": "core",
    "test_text_evidence.py": "core",
    "test_tokens.py": "core",
    "test_vast_provision_capped_policy.py": "remote_policy",
    "test_vast_watchdog.py": "remote_policy",
}

EXPENSIVE_LOCAL_MODEL_PREFIX = "test_expensive_local_model_"
REMOTE_PAID_PREFIX = "test_remote_paid_"


def tier_for_path(path: str | Path) -> str:
    """Return the one repository tier assigned to a test module path."""
    name = Path(path).name
    if name.startswith(EXPENSIVE_LOCAL_MODEL_PREFIX):
        return "expensive_local_model"
    if name.startswith(REMOTE_PAID_PREFIX):
        return "remote_paid"
    try:
        return MODULE_TIERS[name]
    except KeyError as exc:
        raise ValueError(
            f"test module {name!r} has no intentional tier assignment"
        ) from exc


def resolve_tier(path: str | Path, explicit: list[str]) -> str:
    """Resolve item markers without allowing protected module downgrades."""
    if len(explicit) > 1:
        raise ValueError(f"multiple test tiers: {explicit}")
    mapped = tier_for_path(path)
    if explicit and mapped != "core" and explicit[0] != mapped:
        raise ValueError(
            f"{Path(path).name} is fixed to {mapped}, not {explicit[0]}")
    return explicit[0] if explicit else mapped


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("rcc test tiers")
    group.addoption(
        "--run-lab-unit", action="store_true", default=False,
        help="run local torch lab_unit tests",
    )
    group.addoption(
        "--run-transformer-integration", action="store_true", default=False,
        help="run tiny/config-built transformer integration tests",
    )
    group.addoption(
        "--run-remote-policy", action="store_true", default=False,
        help="run offline tests for sealed remote/provider policy",
    )
    group.addoption(
        "--run-expensive-local-model", action="store_true", default=False,
        help="run tests that may load sizeable local model assets",
    )
    group.addoption(
        "--run-remote-paid", action="store_true", default=False,
        help=("enable remote_paid tests only with the test-only "
              "RCC_TEST_REMOTE_PAID_ACKNOWLEDGED=1 safeguard"),
    )


def _require_modules(tier: str, modules: tuple[str, ...]) -> None:
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        raise pytest.UsageError(
            f"{tier} requested but dependencies are missing: {', '.join(missing)}")


def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("--run-lab-unit"):
        _require_modules("lab_unit", ("torch",))
    if config.getoption("--run-transformer-integration"):
        _require_modules("transformer_integration", ("torch", "transformers"))
    if config.getoption("--run-remote-policy"):
        _require_modules("remote_policy", ("torch", "transformers", "packaging"))


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    enabled = {
        "core": True,
        "lab_unit": config.getoption("--run-lab-unit"),
        "transformer_integration": config.getoption(
            "--run-transformer-integration"),
        "remote_policy": config.getoption("--run-remote-policy"),
        "expensive_local_model": config.getoption(
            "--run-expensive-local-model"),
        "remote_paid": (
            config.getoption("--run-remote-paid")
            and os.environ.get("RCC_TEST_REMOTE_PAID_ACKNOWLEDGED") == "1"
        ),
    }

    for item in items:
        explicit = [tier for tier in TIERS if item.get_closest_marker(tier)]
        try:
            tier = resolve_tier(item.path, explicit)
        except ValueError as error:
            raise pytest.UsageError(f"{item.nodeid}: {error}") from error
        if not explicit:
            item.add_marker(getattr(pytest.mark, tier))
        if not enabled[tier]:
            reason = f"{tier} is opt-in"
            if tier == "remote_paid":
                reason += " and requires RCC_TEST_REMOTE_PAID_ACKNOWLEDGED=1"
            item.add_marker(pytest.mark.skip(reason=reason))

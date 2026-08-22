"""Unit tests: versioned model-state ABI (latent_lab.state)."""

import pytest

from latent_lab.backends.mock import MockHybridBackend
from latent_lab.intervals import LayerInterval
from latent_lab.protocols import ModelInfo, ProblemInput, check_backend
from latent_lab.state import CacheHandle, RCCModelState, Workspace


@pytest.fixture
def backend():
    return MockHybridBackend(interval=LayerInterval(0, 4, "early"))


@pytest.fixture
def state(backend):
    vec = tuple(i / 10 for i in range(8))
    return backend.contextualize(ProblemInput("p1", (vec,), ("blob-x",)))


def test_state_is_versioned(state):
    assert state.schema_version >= 2
    sid = state.state_id()
    assert sid.startswith("st-") and len(sid) == 19


def test_interval_validation(backend):
    info = backend.info()
    with pytest.raises(ValueError):
        RCCModelState.create(info, _ws(), CacheHandle(), (5, 3))


def test_interval_must_be_contiguous_and_in_range():
    b = MockHybridBackend()
    info = b.info()
    n = len(info.layer_types)
    RCCModelState.create(info, _ws(), CacheHandle(), (0, n))   # ok
    with pytest.raises(ValueError):
        RCCModelState.create(info, _ws(), CacheHandle(), (n, n + 1))


def test_workspace_requires_all_slots():
    with pytest.raises(ValueError):
        Workspace(memory=[], working=[((0.0,) * 8)], readout=[((0.0,) * 8)])


def test_step_index_advances_only_via_with_workspace(state, backend):
    st1 = backend.latent_step(state)
    assert st1.latent_step_index == state.latent_step_index + 1
    # frozen dataclass: original untouched
    assert state.latent_step_index == 0


def test_provenance_refs_carried_through_steps(backend, state):
    st2 = backend.latent_step(backend.latent_step(state))
    assert st2.provenance_refs == ("blob-x",)


def test_check_backend_rejects_garbage():
    check_backend(MockHybridBackend())
    from latent_lab.protocols import LatentBackend

    assert not isinstance(object(), LatentBackend)
    with pytest.raises(TypeError):
        check_backend(object())


def test_identity_header_records_runtime(state):
    assert state.runtime == "mock"
    assert state.repo_id == "mock/hybrid-tiny"
    assert state.dtype == "fp64-pyfloat"
    assert "gdn" in state.layer_types and "attn" in state.layer_types


def _ws():
    v = (0.0,) * 8
    return Workspace(memory=[v], working=[v], readout=[v])


def test_model_info_frozen():
    info = ModelInfo(
        repo_id="r", revision="v", config_hash="c", tokenizer_hash="t",
        runtime="mock", runtime_version="1", dtype="fp32",
        layer_types=("attn",),
    )
    with pytest.raises(Exception):
        info.repo_id = "other"  # type: ignore[misc]

"""Capped Vast provisioner policy regressions (fake client, NO network).

Pins each link of the pre-spend fail-closed chain for the single canary
instance: >=24 GB VRAM with safe MiB-to-GB normalization, on-demand rate
<=$0.45/hour with malformed prices refused, ANY active instance or ANY
existing volume aborting BEFORE search/create, exactly-one create
carrying the immutable image, and non-spending --status/--attach-key.
"""

import json
import sys
import types
from pathlib import Path

import pytest

from latent_lab.bench import sealed_env, vast_provision

REPO = Path(__file__).resolve().parents[1]
GOOD_IMAGE = "pytorch/pytorch@sha256:" + "a1" * 32


def _live_contract(path, image=GOOD_IMAGE) -> Path:
    """Controlled EXACT contract from this interpreter's real versions."""
    v = sealed_env.live_versions()
    d = {"image": image,
         "python": v["python"],
         "torch": v["torch"],
         "transformers": v["transformers"],
         "huggingface_hub": v["huggingface_hub"],
         "uvlock_sha256": sealed_env.lockfile_sha256(REPO / "uv.lock")}
    path.write_text(json.dumps(d))
    return path


def eligible_offer(**over):
    d = {"id": 777, "gpu_name": "RTX 3090", "gpu_arch": "NVIDIA",
         "num_gpus": 1, "gpu_ram": 24576, "dph_total": 0.31,
         "verified": True, "rentable": True, "reliability": 0.99,
         "inet_down": 900.0, "direct_port_count": 1}
    d.update(over)
    return d


MINE_RUNNING = {"id": 42, "label": vast_provision.LABEL,
                "actual_status": "running",
                "public_ipaddr": "203.0.113.7",
                "ports": {"22/tcp": [{"HostPort": 40001}]},
                "gpu_name": "RTX 3090"}


class FakeVastAI:
    """Offline stand-in recording every provider call, in order."""

    instances, volumes, offers = [], [], []
    calls, created_kwargs, search_kwargs = [], [], []
    attached = None
    instantiated = False

    @classmethod
    def reset(cls, instances=None, volumes=None, offers=None):
        cls.instances = [] if instances is None else instances
        cls.volumes = [] if volumes is None else volumes
        cls.offers = [] if offers is None else offers
        cls.calls = []
        cls.created_kwargs = []
        cls.search_kwargs = []
        cls.attached = None
        cls.instantiated = False

    def __init__(self, *a, **k):
        type(self).instantiated = True
        type(self).calls.append("init")

    def show_instances(self):
        type(self).calls.append("show_instances")
        return json.dumps(type(self).instances)

    def show_volumes(self):
        type(self).calls.append("show_volumes")
        return json.dumps(type(self).volumes)

    def search_offers(self, **kw):
        type(self).calls.append("search_offers")
        type(self).search_kwargs.append(kw)
        return json.dumps(type(self).offers)

    def create_instance(self, **kw):
        type(self).calls.append("create_instance")
        type(self).created_kwargs.append(kw)
        return json.dumps({"new_contract": 424242})

    def attach_ssh(self, instance_id, public_key):
        type(self).calls.append("attach_ssh")
        type(self).attached = (instance_id, public_key)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    FakeVastAI.reset()
    mod = types.ModuleType("vastai")
    mod.VastAI = FakeVastAI
    monkeypatch.setitem(sys.modules, "vastai", mod)
    monkeypatch.setenv("VAST_AI_API_KEY", "test-key")
    monkeypatch.setattr(vast_provision, "STATE",
                        tmp_path / ".rcc_work" / "vast_instance.json")
    return FakeVastAI


def _run(monkeypatch, tmp_path, extra):
    argv = ["vast_provision.py", *extra]
    if "--create" in extra:
        contract = _live_contract(tmp_path / "contract.json")
        argv += ["--image", GOOD_IMAGE,
                 "--env-contract", str(contract)]
    monkeypatch.setattr(sys, "argv", argv)
    vast_provision.main()


def _run_create(monkeypatch, tmp_path):
    _run(monkeypatch, tmp_path, ["--create"])


# ---------------------------------------------------------------------------
# capped policy constants + provider query grammar
# ---------------------------------------------------------------------------

def test_query_and_caps_pin_the_capped_policy():
    q = vast_provision.QUERY
    assert "gpu_ram>=24" in q
    assert "gpu_ram>=22" not in q
    assert "num_gpus=1" in q
    assert "gpu_arch=nvidia" in q
    assert "p40" not in q.lower()
    assert vast_provision.MIN_VRAM_GB == 24
    assert vast_provision.MAX_DPH_TOTAL == 0.45
    assert vast_provision.WALL_LIMIT_S == 7200
    assert vast_provision.MAX_TOTAL_USD == 1.0
    projected = (vast_provision.MAX_DPH_TOTAL
                 * vast_provision.WALL_LIMIT_S / 3600)
    assert projected == pytest.approx(0.90)
    assert projected <= vast_provision.MAX_TOTAL_USD
    names = " ".join(vast_provision.ELIGIBLE_GPUS).lower()
    assert "p40" not in names
    assert "rtx 3090" in names and "rtx 4090" in names


# ---------------------------------------------------------------------------
# gpu_ram unit normalization (MiB payloads vs GB query syntax)
# ---------------------------------------------------------------------------

def test_gpu_ram_unit_normalization_is_safe():
    n = vast_provision.normalize_gpu_ram_gb
    assert n(24576) == 24.0
    assert n("24576") == 24.0
    assert n(81920) == 80.0
    assert n(24) == 24.0
    assert n(80.5) == 80.5
    for bad in (None, "", "   ", "abc", -1, 0, float("nan"),
                float("inf"), float("-inf"), True, False, [], {}):
        with pytest.raises(ValueError):
            n(bad)


def test_local_revalidation_rejects_each_policy_violation():
    r = vast_provision.offer_rejection_reason
    assert r(eligible_offer()) is None
    assert r(eligible_offer(gpu_name="RTX_4090")) is None
    assert r(eligible_offer(gpu_ram=24576.0)) is None
    assert r(eligible_offer(gpu_ram=22528))          # 22 GiB MiB
    assert r(eligible_offer(gpu_ram=24064))          # 23.5 GB in MiB
    assert r(eligible_offer(gpu_ram="abc"))
    assert r(eligible_offer(gpu_ram=None))
    assert r(eligible_offer(dph_total=0.46))
    assert r(eligible_offer(dph_total="abc"))
    assert r(eligible_offer(num_gpus=2))
    assert r(eligible_offer(num_gpus="one"))
    assert r(eligible_offer(verified=False))
    real_payload = eligible_offer()
    del real_payload["verified"]
    real_payload.update(verification="verified", vericode=1,
                        is_vm_deverified=False)
    assert r(real_payload) is None
    assert r({**real_payload, "verification": "unverified"})
    assert r({**real_payload, "is_vm_deverified": True})
    assert r({**real_payload, "verified": False})
    assert r(eligible_offer(gpu_arch="AMD"))
    assert r(eligible_offer(gpu_name="Tesla P40"))
    assert r(eligible_offer(gpu_name="RTX 3080"))
    assert r("not-an-offer")


# ---------------------------------------------------------------------------
# search + local revalidation against the create path (fake client)
# ---------------------------------------------------------------------------

def test_22gb_offer_is_rejected_before_create(fake, monkeypatch, tmp_path):
    fake.reset(offers=[eligible_offer(gpu_ram=22528)])
    with pytest.raises(SystemExit, match="no eligible offer"):
        _run_create(monkeypatch, tmp_path)
    assert "create_instance" not in fake.calls
    assert "search_offers" in fake.calls
    assert not vast_provision.STATE.exists()


def test_24gb_mib_offer_reaches_create_exactly_once_with_immutable_image(
        fake, monkeypatch, tmp_path, capsys):
    fake.reset(offers=[eligible_offer()])
    _run_create(monkeypatch, tmp_path)
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "created"
    assert fake.calls.count("create_instance") == 1
    kw = fake.created_kwargs[0]
    assert kw["image"] == GOOD_IMAGE
    assert kw["id"] == 777
    assert kw["label"] == vast_provision.LABEL
    assert kw["disk"] == 40
    assert kw["runtype"] == "ssh_direct"
    assert fake.search_kwargs[0]["query"] == vast_provision.QUERY
    state = json.loads(vast_provision.STATE.read_text())
    assert state["instance_id"] == 424242
    assert state["sealed"]["image"] == GOOD_IMAGE


def test_rate_exactly_at_cap_is_eligible(fake, monkeypatch, tmp_path,
                                         capsys):
    fake.reset(offers=[eligible_offer(dph_total=0.45)])
    _run_create(monkeypatch, tmp_path)
    assert json.loads(capsys.readouterr().out)["status"] == "created"


@pytest.mark.parametrize("bad_price", [
    0.46, 2.0, "0.50", None, "", "abc", float("nan"), float("inf"), -0.10])
def test_bad_prices_are_rejected_without_create(fake, monkeypatch, tmp_path,
                                                bad_price):
    fake.reset(offers=[eligible_offer(dph_total=bad_price)])
    with pytest.raises(SystemExit, match="no eligible offer"):
        _run_create(monkeypatch, tmp_path)
    assert "create_instance" not in fake.calls


def test_missing_price_is_rejected_without_create(fake, monkeypatch,
                                                  tmp_path):
    offer = eligible_offer()
    del offer["dph_total"]
    offer.pop("dph", None)
    fake.reset(offers=[offer])
    with pytest.raises(SystemExit, match="no eligible offer"):
        _run_create(monkeypatch, tmp_path)
    assert "create_instance" not in fake.calls


# ---------------------------------------------------------------------------
# fail-closed inventory: active instances and volumes block BEFORE search
# ---------------------------------------------------------------------------

def test_foreign_active_instance_aborts_before_search_or_create(
        fake, monkeypatch, tmp_path):
    fake.reset(instances=[{"id": 5, "label": "someone-else",
                           "actual_status": "running"}])
    with pytest.raises(SystemExit,
                       match="active_instances_block_create"):
        _run_create(monkeypatch, tmp_path)
    assert fake.calls == ["init", "show_instances"]
    assert not vast_provision.STATE.exists()


def test_own_labeled_active_instance_reports_already_exists(
        fake, monkeypatch, tmp_path, capsys):
    fake.reset(instances=[dict(MINE_RUNNING)])
    _run_create(monkeypatch, tmp_path)
    assert json.loads(capsys.readouterr().out) == {
        "status": "already_exists", "instance_id": 42}
    assert fake.calls == ["init", "show_instances"]
    assert not vast_provision.STATE.exists()


def test_exited_instances_do_not_block_create(fake, monkeypatch, tmp_path,
                                              capsys):
    fake.reset(instances=[{"id": 3, "label": "old-run",
                           "actual_status": "exited"}],
               offers=[eligible_offer()])
    _run_create(monkeypatch, tmp_path)
    assert json.loads(capsys.readouterr().out)["status"] == "created"


def test_existing_volume_aborts_before_search_or_create(
        fake, monkeypatch, tmp_path):
    fake.reset(volumes=[{"id": 11}])
    with pytest.raises(SystemExit, match="volumes_exist"):
        _run_create(monkeypatch, tmp_path)
    assert fake.calls == ["init", "show_instances", "show_volumes"]
    assert "search_offers" not in fake.calls
    assert "create_instance" not in fake.calls
    assert not vast_provision.STATE.exists()


def test_p40_is_never_selected_even_when_cheapest(fake, monkeypatch,
                                                  tmp_path, capsys):
    p40 = eligible_offer(id=1, gpu_name="Tesla P40", dph_total=0.05)
    rtx = eligible_offer(id=2, gpu_name="RTX_4090", dph_total=0.44)
    fake.reset(offers=[p40, rtx])
    _run_create(monkeypatch, tmp_path)
    assert json.loads(capsys.readouterr().out)["status"] == "created"
    assert fake.created_kwargs[0]["id"] == 2


# ---------------------------------------------------------------------------
# non-spending paths are preserved
# ---------------------------------------------------------------------------

def test_status_remains_non_spending(fake, monkeypatch, tmp_path, capsys):
    fake.reset(instances=[dict(MINE_RUNNING)])
    _run(monkeypatch, tmp_path, ["--status"])
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "existing_instance"
    assert out["ssh"] == {"host": "203.0.113.7", "port": 40001}
    assert fake.calls == ["init", "show_instances"]
    assert not vast_provision.STATE.exists()


def test_status_without_instance_reports_absence(fake, monkeypatch,
                                                 tmp_path, capsys):
    fake.reset()
    _run(monkeypatch, tmp_path, ["--status"])
    assert json.loads(capsys.readouterr().out) == {"status": "no_instance"}
    assert fake.calls == ["init", "show_instances"]


def test_attach_key_attaches_once_without_spending(fake, monkeypatch,
                                                   tmp_path, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir()
    key = "ssh-ed25519 AAAAC3Nza test@host"
    (ssh_dir / "id_ed25519.pub").write_text(key)
    fake.reset(instances=[dict(MINE_RUNNING)])
    _run(monkeypatch, tmp_path, ["--attach-key"])
    out = json.loads(capsys.readouterr().out)
    assert out == {"status": "key_attached", "instance_id": 42}
    assert fake.calls == ["init", "show_instances", "attach_ssh"]
    assert fake.attached == (42, key)
    assert not vast_provision.STATE.exists()

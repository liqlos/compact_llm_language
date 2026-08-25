import json
import subprocess
from pathlib import Path

import pytest

from latent_lab.bench import vast_watchdog as watchdog


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Provider:
    def __init__(self, *, instance=True, volumes=None,
                 destroy_removes=True):
        self.instance = instance
        self.volumes = list(volumes or [])
        self.destroy_removes = destroy_removes
        self.commands = []

    def __call__(self, command):
        self.commands.append(command)
        if command[:3] == ["vastai", "show", "instances"]:
            items = ([{"id": 42, "actual_status": "running"}]
                     if self.instance else [])
            return subprocess.CompletedProcess(command, 0, json.dumps(items), "")
        if command[:3] == ["vastai", "show", "volumes"]:
            return subprocess.CompletedProcess(
                command, 0, json.dumps(self.volumes), "")
        assert command == ["vastai", "destroy", "instance", "42", "-y",
                           "--raw"]
        if self.destroy_removes:
            self.instance = False
        return subprocess.CompletedProcess(command, 0, "{}", "")


def _run(tmp_path, provider, clock, **over):
    kwargs = {
        "instance_id": 42, "stop_file": tmp_path / "done",
        "log_path": tmp_path / "watchdog.jsonl", "wall_seconds": 10,
        "teardown_seconds": 4, "poll_seconds": 1, "runner": provider,
        "monotonic": clock.monotonic, "sleep": clock.sleep,
    }
    kwargs.update(over)
    return watchdog.watch(**kwargs)


def test_wall_deadline_destroys_once_and_confirms_empty_inventories(tmp_path):
    clock = Clock()
    provider = Provider()
    assert _run(tmp_path, provider, clock) == 0
    destroys = [c for c in provider.commands if c[1] == "destroy"]
    assert destroys == [["vastai", "destroy", "instance", "42", "-y",
                         "--raw"]]
    events = [json.loads(x)["event"] for x in
              (tmp_path / "watchdog.jsonl").read_text().splitlines()]
    assert events[-1] == "teardown_confirmed"
    assert clock.now <= 11


def test_completion_sentinel_triggers_immediate_teardown(tmp_path):
    (tmp_path / "done").write_text("complete\n")
    clock = Clock()
    provider = Provider()
    assert _run(tmp_path, provider, clock) == 0
    assert any(c[1] == "destroy" for c in provider.commands)
    assert clock.now == 0


def test_initially_absent_instance_does_not_make_watchdog_exit_early(tmp_path):
    clock = Clock()
    provider = Provider(instance=False)
    assert _run(tmp_path, provider, clock) == 0
    assert any(c[1] == "destroy" for c in provider.commands)
    assert clock.now == 10


def test_existing_volume_prevents_false_teardown_confirmation(tmp_path):
    clock = Clock()
    provider = Provider(volumes=[{"id": 7}], destroy_removes=True)
    assert _run(tmp_path, provider, clock) == 2
    final = json.loads((tmp_path / "watchdog.jsonl").read_text().splitlines()[-1])
    assert final["event"] == "teardown_unconfirmed"
    assert clock.now <= 14


def test_unremoved_instance_fails_within_teardown_cap(tmp_path):
    clock = Clock()
    provider = Provider(destroy_removes=False)
    assert _run(tmp_path, provider, clock) == 2
    assert clock.now <= 14
    assert len([c for c in provider.commands if c[1] == "destroy"]) == 4


@pytest.mark.parametrize("over", [
    {"wall_seconds": 7201}, {"wall_seconds": 0},
    {"teardown_seconds": 121}, {"teardown_seconds": 0},
    {"poll_seconds": 0}, {"poll_seconds": 5, "teardown_seconds": 4},
])
def test_caps_fail_closed_before_provider_contact(tmp_path, over):
    clock = Clock()
    provider = Provider()
    with pytest.raises(watchdog.WatchdogError):
        _run(tmp_path, provider, clock, **over)
    assert provider.commands == []


def test_invalid_inventory_cannot_be_treated_as_teardown(tmp_path):
    clock = Clock()

    def provider(command):
        if command[1] == "destroy":
            return subprocess.CompletedProcess(command, 0, "{}", "")
        return subprocess.CompletedProcess(command, 0, "not-json", "")

    assert _run(tmp_path, provider, clock) == 2

"""Provision a vast.ai GPU instance for the RCC behavioral gate (pattern 688).

Policy (owner directive 2026-08-23): ALWAYS create with runtype="ssh_direct"
— direct SSH is more stable than proxied connections.

Uses VAST_AI_API_KEY from the environment (never printed). Selects only
verified on-demand NVIDIA offers with one GPU and >=22 GB VRAM, records the
chosen offer/instance JSON under .rcc_work/, and never destroys instances
from this script.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from vastai import VastAI

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / ".rcc_work" / "vast_instance.json"
QUERY = ("verified=true rentable=true gpu_arch=nvidia num_gpus=1 "
         "gpu_name in [RTX_3090,RTX_4090] gpu_ram>=22 direct_port_count>=1 "
         "reliability>=0.98 cpu_ram>=16 inet_down>=500 disk_space>=40")
IMAGE = "pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime"
LABEL = "rcc-latent-gate"


def parsed(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def attach_local_ssh_key(client: VastAI, instance_id: int) -> None:
    public_key = Path.home() / ".ssh" / "id_ed25519.pub"
    if not public_key.is_file():
        raise SystemExit("missing local ~/.ssh/id_ed25519.pub")
    try:
        client.attach_ssh(instance_id,
                          public_key.read_text(encoding="utf-8").strip())
    except Exception as exc:
        if "already" not in str(exc).lower():
            raise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--create", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--attach-key", action="store_true")
    args = ap.parse_args()

    api_key = os.environ.get("VAST_AI_API_KEY")
    if not api_key:
        raise SystemExit("missing VAST_AI_API_KEY in environment")
    client = VastAI(api_key=api_key, raw=True)

    instances = parsed(client.show_instances())
    active = [x for x in instances
              if str(x.get("actual_status", "")).lower()
              not in ("", "exited", "destroyed")]
    mine = [x for x in active if x.get("label") == LABEL]

    if args.attach_key or args.status:
        if not mine:
            print(json.dumps({"status": "no_instance"}))
            return
        inst = mine[0]
        if args.attach_key:
            attach_local_ssh_key(client, int(inst["id"]))
            print(json.dumps({"status": "key_attached",
                              "instance_id": inst["id"]}))
            return
        ports = (inst.get("ports") or {}).get("22/tcp") or []
        ssh = {"host": inst.get("public_ipaddr") or inst.get("ssh_host"),
               "port": (ports[0]["HostPort"] if ports
                        else inst.get("ssh_port"))}
        print(json.dumps({"status": "existing_instance",
                          "instance_id": inst.get("id"),
                          "actual_status": inst.get("actual_status"),
                          "gpu_name": inst.get("gpu_name"),
                          "ssh": ssh}))
        return

    if mine:
        print(json.dumps({"status": "already_exists",
                          "instance_id": mine[0].get("id")}))
        return
    if any(active):
        print(json.dumps({"status": "other_active_instances",
                          "ids": [x.get("id") for x in active]}))

    offers = parsed(client.search_offers(query=QUERY, order="dph_total",
                                         limit=12))
    if isinstance(offers, dict):
        offers = offers.get("offers", offers.get("instances", []))
    if not offers:
        raise SystemExit("no eligible offer")
    offer = min(offers, key=lambda x: float(x.get("dph_total",
                                                  x.get("dph", 1e9))))
    selected = {k: offer.get(k) for k in
                ("id", "gpu_name", "gpu_ram", "dph_total", "reliability",
                 "inet_down", "direct_port_count")}
    if not args.create:
        print(json.dumps({"status": "offer_selected", "offer": selected}))
        return

    result = parsed(client.create_instance(
        id=int(offer["id"]), image=IMAGE, disk=40, runtype="ssh_direct",
        label=LABEL, cancel_unavail=True))
    instance_id = result.get("new_contract") or result.get("id")
    if not instance_id:
        raise SystemExit(f"create returned no id: {result!r}")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"offer": selected,
                                 "create_result": result,
                                 "instance_id": instance_id}, indent=2))
    print(json.dumps({"status": "created", "instance_id": instance_id,
                      "offer": selected}))


if __name__ == "__main__":
    main()

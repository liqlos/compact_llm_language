"""Provision a vast.ai GPU instance for the RCC behavioral gate (pattern 688).

Policy (owner directive 2026-08-23): ALWAYS create with runtype="ssh_direct"
— direct SSH is more stable than proxied connections.

SEALED LAUNCH ENVIRONMENT (fail closed): ``--create`` is refused unless it
receives an EXPLICITLY immutable image reference
(``repository@sha256:<64 lowercase hex>``) AND a sealed-environment
contract file binding Python, torch, transformers, huggingface_hub and
the project lockfile SHA — with the contract's image equal to the
requested digest. Mutable tags are never accepted as launch images and
no default image exists: an unsealed request aborts BEFORE any provider
contact. Until pre-spend supplies a verified compatible digest, this
provisioner honestly has NO launchable default.

Uses VAST_AI_API_KEY from the environment (never printed). Selects only
verified on-demand NVIDIA RTX 3090/4090 offers with one GPU and >=24 GB
VRAM at an on-demand rate <=$0.45/hour, revalidating EVERY returned
offer locally (raw offer payloads report gpu_ram in MiB, e.g. 24576,
while the query grammar uses GB; readings are normalized before any
comparison and malformed values refuse the offer). The capped canary
policy is ONE instance within a 7200 s wall limit and a <=$1 total
budget: the rate cap bounds the full 2 h to $0.90. Any other active
instance or ANY existing volume fails the run closed BEFORE search or
create. Records the chosen offer/instance JSON under .rcc_work/, and
never destroys instances from this script.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / ".rcc_work" / "vast_instance.json"
MIN_VRAM_GB = 24
MAX_DPH_TOTAL = 0.45
WALL_LIMIT_S = 7200
MAX_TOTAL_USD = 1.0
NUM_GPUS = 1
ELIGIBLE_GPUS = frozenset({"rtx 3090", "rtx 4090"})
QUERY = ("verified=true rentable=true gpu_arch=nvidia num_gpus=1 "
         "gpu_name in [RTX_3090,RTX_4090] gpu_ram>=24 direct_port_count>=1 "
         "reliability>=0.98 cpu_ram>=16 inet_down>=500 disk_space>=40")
LABEL = "rcc-latent-gate"


def parsed(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def normalize_gpu_ram_gb(value) -> float:
    """Convert a Vast gpu_ram reading to GB, refusing malformed input.

    Raw offer payloads report MiB (e.g. 24576) while the query grammar
    uses GB (gpu_ram>=24). Readings >=1024 are unambiguously MiB — no
    single GPU approaches 1024 GB of VRAM and every relevant card reads
    far above it — anything smaller is taken as already-GB.
    """
    if isinstance(value, bool):
        raise ValueError(f"gpu_ram {value!r} is not a number")
    if isinstance(value, (int, float)):
        num = float(value)
    elif isinstance(value, str):
        try:
            num = float(value.strip())
        except ValueError:
            raise ValueError(
                f"gpu_ram {value!r} is not a number") from None
    else:
        raise ValueError(f"gpu_ram {value!r} is not a number")
    if not math.isfinite(num) or num <= 0:
        raise ValueError(
            f"gpu_ram {value!r} is not a positive finite number")
    return round(num / 1024.0, 6) if num >= 1024 else num


def parse_dph_total(offer) -> float:
    """Strictly read the on-demand total rate from an offer.

    Missing or malformed prices are refused instead of being defaulted
    (the old ``1e9`` sentinel silently hid malformed offers).
    """
    raw = offer.get("dph_total")
    if raw is None:
        raw = offer.get("dph")
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        raise ValueError(f"dph_total {raw!r} is missing or malformed")
    try:
        dph = float(str(raw).strip())
    except ValueError:
        raise ValueError(f"dph_total {raw!r} is malformed") from None
    if not math.isfinite(dph) or dph < 0:
        raise ValueError(f"dph_total {raw!r} is not a usable rate")
    return dph


def offer_rejection_reason(offer) -> str | None:
    """Return why an offer violates the capped policy, else None.

    Applied to EVERY returned offer so provider-side query filtering is
    never trusted blindly: one GPU, verified NVIDIA RTX 3090/4090,
    >=24 GB VRAM (unit-normalized), on-demand rate <=$0.45/h.
    """
    if not isinstance(offer, dict):
        return f"offer {offer!r} is not an object"
    name = str(offer.get("gpu_name", "")).strip().lower().replace("_", " ")
    if name not in ELIGIBLE_GPUS:
        return (f"gpu_name {offer.get('gpu_name')!r} is outside the "
                f"eligible {sorted(ELIGIBLE_GPUS)} set")
    arch = str(offer.get("gpu_arch", "")).strip().lower()
    if arch and "nvidia" not in arch:
        return f"gpu_arch {arch!r} is not NVIDIA"
    try:
        num_gpus = int(offer.get("num_gpus"))
    except (TypeError, ValueError):
        return f"num_gpus {offer.get('num_gpus')!r} is not an integer"
    if num_gpus != NUM_GPUS:
        return f"num_gpus {num_gpus} != {NUM_GPUS}"
    if not offer.get("verified"):
        return "offer is not verified"
    try:
        vram_gb = normalize_gpu_ram_gb(offer.get("gpu_ram"))
    except ValueError as exc:
        return str(exc)
    if vram_gb < MIN_VRAM_GB:
        return (f"gpu_ram {offer.get('gpu_ram')!r} (~{vram_gb:g} GB) is "
                f"below the {MIN_VRAM_GB} GB minimum")
    try:
        dph = parse_dph_total(offer)
    except ValueError as exc:
        return str(exc)
    if dph > MAX_DPH_TOTAL:
        return (f"on-demand rate ${dph:g}/h exceeds the "
                f"${MAX_DPH_TOTAL:g}/h cap")
    return None


def select_offer(offers):
    """Revalidate EVERY returned offer locally and pick the cheapest
    fully eligible one; fail closed when none survives."""
    eligible = []
    rejected = []
    for offer in offers:
        reason = offer_rejection_reason(offer)
        if reason is None:
            eligible.append(offer)
        else:
            rejected.append(reason)
    if not eligible:
        detail = "; ".join(dict.fromkeys(rejected[:5])) \
            if rejected else "empty result"
        raise SystemExit(f"no eligible offer ({detail})")
    return min(eligible, key=parse_dph_total)


def sealed_launch_requirements(args) -> dict | None:
    """Validate the immutable image + contract BEFORE any provider use.

    Returns the sealed binding for --create; None for non-spending
    invocations (--status/--attach-key create nothing).
    """
    from latent_lab.bench.sealed_env import (
        SealedEnvironmentError,
        check_contract_pins,
        load_contract,
        require_immutable_image,
    )

    if not args.create:
        return None
    try:
        image = require_immutable_image(args.image)
        if not args.env_contract:
            raise SealedEnvironmentError(
                "--env-contract FILE is required for --create: "
                "provisioning without an exact sealed-environment "
                "contract (python/torch/transformers/huggingface_hub/"
                "lockfile sha) is refused")
        contract = load_contract(args.env_contract)
        # provisioning and the remote driver are bound to the SAME
        # immutable digest/contract
        check_contract_pins(contract, pins={"image": image})
    except SealedEnvironmentError as e:
        raise SystemExit(f"FATAL: sealed launch environment refused: {e}")
    return {"image": image, "contract_path": str(args.env_contract),
            "contract": contract}


def attach_local_ssh_key(client, instance_id: int) -> None:
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
    ap.add_argument("--image", default=None,
                    help="immutable repository@sha256:<64-hex> reference "
                    "(REQUIRED for --create; mutable tags refused)")
    ap.add_argument("--env-contract", default=None,
                    help="sealed-environment contract JSON (REQUIRED for "
                    "--create)")
    args = ap.parse_args()

    # fail closed BEFORE any provider contact or API-key use
    sealed = sealed_launch_requirements(args)

    api_key = os.environ.get("VAST_AI_API_KEY")
    if not api_key:
        raise SystemExit("missing VAST_AI_API_KEY in environment")

    from vastai import VastAI  # lazy: sealing errors precede any import/use

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
    if active:
        raise SystemExit(json.dumps({
            "status": "active_instances_block_create",
            "ids": [x.get("id") for x in active]}))

    volumes = parsed(client.show_volumes())
    if isinstance(volumes, dict):
        volumes = volumes.get("volumes", [])
    if volumes:
        raise SystemExit(json.dumps({
            "status": "volumes_exist",
            "ids": [v.get("id") for v in volumes
                    if isinstance(v, dict)]}))

    offers = parsed(client.search_offers(query=QUERY, order="dph_total",
                                         limit=12))
    if isinstance(offers, dict):
        offers = offers.get("offers", offers.get("instances", []))
    if not isinstance(offers, list) or not offers:
        raise SystemExit("no eligible offer")
    offer = select_offer(offers)
    selected = {k: offer.get(k) for k in
                ("id", "gpu_name", "gpu_ram", "dph_total", "reliability",
                 "inet_down", "direct_port_count")}
    if not args.create:
        print(json.dumps({"status": "offer_selected", "offer": selected}))
        return

    dph = parse_dph_total(offer)
    projected_usd = dph * WALL_LIMIT_S / 3600
    if projected_usd > MAX_TOTAL_USD + 1e-9:
        raise SystemExit(
            f"capped budget exceeded: ${dph:g}/h over the {WALL_LIMIT_S}s "
            f"wall projects ${projected_usd:.2f}, above ${MAX_TOTAL_USD:g}")

    result = parsed(client.create_instance(
        id=int(offer["id"]), image=sealed["image"], disk=40,
        runtype="ssh_direct", label=LABEL, cancel_unavail=True))
    instance_id = result.get("new_contract") or result.get("id")
    if not instance_id:
        raise SystemExit(f"create returned no id: {result!r}")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"offer": selected,
                                 "create_result": result,
                                 "instance_id": instance_id,
                                 "policy": {
                                     "min_vram_gb": MIN_VRAM_GB,
                                     "max_dph_total": MAX_DPH_TOTAL,
                                     "wall_limit_s": WALL_LIMIT_S,
                                     "max_total_usd": MAX_TOTAL_USD},
                                 "sealed": {
                                     "image": sealed["image"],
                                     "contract_path":
                                         sealed["contract_path"],
                                     "contract": sealed["contract"],
                                 }}, indent=2))
    print(json.dumps({"status": "created", "instance_id": instance_id,
                      "offer": selected, "image": sealed["image"]}))


if __name__ == "__main__":
    main()

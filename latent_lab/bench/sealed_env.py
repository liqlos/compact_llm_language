"""Sealed launch-environment contract: immutable image + exact pins.

A paid instance may ONLY be launched against an EXPLICITLY immutable
image reference (``repository@sha256:<64 lowercase hex>``) plus an exact
sealed-environment contract naming Python, torch, transformers,
huggingface_hub and the project lockfile SHA. Mutable tags can never
present a sealed environment, and a contract that does not match the
preregistered pins aborts BEFORE any spend, test or model contact.

It is honest for a repository to have NO launchable default until a
verified compatible digest has been supplied pre-spend; this module
therefore defines no defaults and never reaches the network.

CLI (fail closed, exit 1 on any refusal):
    python -m latent_lab.bench.sealed_env require-image --image REF
    python -m latent_lab.bench.sealed_env verify-contract --contract F [--image REF]
    python -m latent_lab.bench.sealed_env check-pins --contract F --pin k=v [...]
    python -m latent_lab.bench.sealed_env verify-live --contract F [--lockfile F] [--require-cuda]
"""
from __future__ import annotations

import argparse
import hashlib
import platform
import re
import sys

_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")
# repository@sha256:<64 lowercase hex> — the only acceptable form
IMMUTABLE_IMAGE_RE = re.compile(r"\A[^@\s]+@sha256:([0-9a-f]{64})\Z")

CONTRACT_KEYS = ("image", "python", "torch", "transformers",
                 "huggingface_hub", "uvlock_sha256")


class SealedEnvironmentError(RuntimeError):
    """The launch environment is not (or cannot be proven) sealed."""


def require_immutable_image(image) -> str:
    """Accept ONLY an explicitly immutable digest reference."""
    if not isinstance(image, str):
        raise SealedEnvironmentError(
            f"image reference must be repository@sha256:<64 lowercase "
            f"hex>; got {image!r} of type {type(image).__name__}")
    normalized = image.strip()
    if not IMMUTABLE_IMAGE_RE.fullmatch(normalized):
        raise SealedEnvironmentError(
            f"image reference {image!r} is not an immutable "
            "repository@sha256:<64 lowercase hex> digest; mutable tags "
            "(e.g. ':latest' or 'pytorch/pytorch:2.13.0-cuda12.6-"
            "cudnn9-runtime') can never present a sealed launch "
            "environment")
    return normalized


def load_contract(path) -> dict:
    """Load + strictly validate a sealed-environment contract file."""
    from latent_lab.train.checkpointing import strict_json_loads

    p = str(path)
    try:
        raw = open(p, "rb").read()
    except OSError as e:
        raise SealedEnvironmentError(
            f"sealed-environment contract unreadable: {e}") from e
    try:
        d = strict_json_loads(raw.decode())
    except Exception as e:
        raise SealedEnvironmentError(
            f"sealed-environment contract {p} is not strict JSON: {e}"
        ) from e
    if not isinstance(d, dict):
        raise SealedEnvironmentError(
            f"sealed-environment contract {p} is not a JSON object")
    keys = set(d)
    missing = [k for k in CONTRACT_KEYS if k not in keys]
    extra = sorted(keys - set(CONTRACT_KEYS))
    if missing or extra:
        raise SealedEnvironmentError(
            f"sealed-environment contract {p} key set mismatch — missing "
            f"{missing}, unexpected {extra}; the contract must bind "
            f"exactly {sorted(CONTRACT_KEYS)}")
    for k, v in d.items():
        if not isinstance(v, str) or not v.strip():
            raise SealedEnvironmentError(
                f"contract field {k!r} must be a non-empty string; got "
                f"{v!r}")
    if not IMMUTABLE_IMAGE_RE.fullmatch(d["image"]):
        raise SealedEnvironmentError(
            f"contract image {d['image']!r} is not an immutable "
            "repository@sha256:<64 lowercase hex> digest")
    if not _SHA256_RE.fullmatch(d["uvlock_sha256"]):
        raise SealedEnvironmentError(
            "contract uvlock_sha256 is not 64-lowercase-hex")
    return d


def check_contract_pins(contract: dict, *, pins: dict) -> None:
    """Cross-bind the contract to preregistered pins (exact equality).

    Unknown pin keys are rejected so a drift between what the caller
    preregistered and what the contract seals can never pass silently.
    """
    if not isinstance(pins, dict) or not pins:
        raise SealedEnvironmentError("no pins supplied to cross-bind")
    unknown = sorted(set(pins) - set(CONTRACT_KEYS))
    if unknown:
        raise SealedEnvironmentError(
            f"unknown pin keys {unknown}; pins must be a subset of "
            f"{sorted(CONTRACT_KEYS)}")
    for k, v in pins.items():
        if contract.get(k) != v:
            raise SealedEnvironmentError(
                f"sealed-environment contract disagrees with the "
                f"preregistered pin for {k!r}: contract "
                f"{contract.get(k)!r} != pin {v!r}")


def live_versions() -> dict:
    """Versions of the RUNNING interpreter's sealed-relevant packages."""
    out = {"python": platform.python_version()}
    for module, key in (("torch", "torch"),
                        ("transformers", "transformers"),
                        ("huggingface_hub", "huggingface_hub")):
        try:
            mod = __import__(module)
        except ImportError as e:
            raise SealedEnvironmentError(
                f"running environment lacks {module}: {e}") from e
        version = getattr(mod, "__version__", None)
        if not isinstance(version, str) or not version:
            raise SealedEnvironmentError(
                f"running {module} exposes no usable __version__")
        out[key] = version
    return out


def lockfile_sha256(path) -> str:
    try:
        h = hashlib.sha256()
        with open(str(path), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError as e:
        raise SealedEnvironmentError(
            f"lockfile unreadable: {e}") from e


def verify_live_environment(contract: dict, *, lockfile=None,
                            require_cuda: bool = False) -> None:
    """Verify the RUNNING environment against the same sealed contract."""
    live = live_versions()
    for k in ("python", "torch", "transformers", "huggingface_hub"):
        if live[k] != contract[k]:
            raise SealedEnvironmentError(
                f"live environment violates the sealed contract: {k} "
                f"{live[k]!r} != contracted {contract[k]!r}")
    if lockfile is not None \
            and lockfile_sha256(lockfile) != contract["uvlock_sha256"]:
        raise SealedEnvironmentError(
            "live project lockfile sha256 differs from the sealed "
            f"contract ({contract['uvlock_sha256']})")
    if require_cuda:
        import torch
        if not torch.cuda.is_available():
            raise SealedEnvironmentError(
                "sealed contract requires CUDA; torch reports it "
                "unavailable")


def _die(msg: str) -> int:
    print(f"FATAL: {msg}", file=sys.stderr)
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="latent_lab.bench.sealed_env")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_img = sub.add_parser("require-image")
    p_img.add_argument("--image", required=True)

    p_con = sub.add_parser("verify-contract")
    p_con.add_argument("--contract", required=True)
    p_con.add_argument("--image", default=None)

    p_pin = sub.add_parser("check-pins")
    p_pin.add_argument("--contract", required=True)
    p_pin.add_argument("--pin", action="append", default=[],
                       metavar="KEY=VALUE")

    p_live = sub.add_parser("verify-live")
    p_live.add_argument("--contract", required=True)
    p_live.add_argument("--lockfile", default=None)
    p_live.add_argument("--require-cuda", action="store_true")

    args = ap.parse_args(argv)
    try:
        if args.cmd == "require-image":
            require_immutable_image(args.image)
        elif args.cmd == "verify-contract":
            contract = load_contract(args.contract)
            if args.image is not None:
                image = require_immutable_image(args.image)
                check_contract_pins(contract, pins={"image": image})
        elif args.cmd == "check-pins":
            contract = load_contract(args.contract)
            pins = {}
            for item in args.pin:
                key, _, value = item.partition("=")
                if not _ or not key:
                    raise SealedEnvironmentError(
                        f"--pin expects KEY=VALUE; got {item!r}")
                pins[key] = value
            check_contract_pins(contract, pins=pins)
        elif args.cmd == "verify-live":
            contract = load_contract(args.contract)
            verify_live_environment(contract, lockfile=args.lockfile,
                                    require_cuda=args.require_cuda)
        else:  # pragma: no cover - argparse enforces choices
            raise SealedEnvironmentError(f"unknown command {args.cmd!r}")
    except SealedEnvironmentError as e:
        return _die(str(e))
    print("SEALED_ENV_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

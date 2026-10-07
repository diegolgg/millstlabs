"""Run manifest: config digest, prompt hashes, anchor hashes, warm-start hashes, code and engine versions."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path

from ..bots.anchors import SOURCE_ANCHORS, anchor_source
from ..llm.prompts import prompt_hashes

_ROOT = Path(__file__).resolve().parents[3]
HLE_COMMIT = "54e79594f4b6fb40ebb3004289c6db0e34a8b5fb"
CANAAN_COMMIT = "dde3edbf7c59d5f1edb0aae13660c06774413169"


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=_ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def anchor_hashes() -> dict[str, str]:
    out = {n: hashlib.sha256(anchor_source(n).encode()).hexdigest()[:16] for n in SOURCE_ANCHORS}
    canaan = Path(__file__).resolve().parents[1] / "bots" / "anchors" / "canaan"
    for p in sorted(canaan.glob("*.py")):
        out[f"canaan/{p.name}"] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return out


def build_manifest(cfg) -> dict:
    return {
        "config_digest": cfg.digest(),
        "name": cfg.name,
        "condition": cfg.condition,
        "prompt_hashes": prompt_hashes(),
        "anchor_hashes": anchor_hashes(),
        "code_commit": _git("rev-parse", "HEAD"),
        "code_dirty": bool(_git("status", "--porcelain", "--", "src")),
        "hle_commit": HLE_COMMIT,
        "canaan_commit": CANAAN_COMMIT,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "llm_backend": cfg.llm.backend,
        **({"llm_provider": provider_record(cfg)} if cfg.llm.backend == "openai_compat" else {}),
    }


def provider_record(cfg) -> dict:
    """Which provider served the run and under what price and cap. Records the NAME of the key's environment variable,
    never its value."""
    c = cfg.llm
    return {"base_url": c.base_url, "model": c.model, "hosted": bool(c.api_key_env), "api_key_env": c.api_key_env,
            "price_in_per_mtok": c.price_in_per_mtok, "price_out_per_mtok": c.price_out_per_mtok,
            "max_usd": c.max_usd, "spend_file": c.spend_file}

"""Config loading and stable hashing. Every persisted result carries the hash of the
configuration that produced it, so readouts are traceable and reproducible."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
ENGINE_VERSION = "0.1.0"


def load_yaml(name: str | Path) -> dict[str, Any]:
    path = Path(name)
    if not path.is_absolute() and not path.exists():
        path = CONFIGS / path
    return yaml.safe_load(path.read_text())


def stable_hash(obj: Any, n: int = 12) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:n]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

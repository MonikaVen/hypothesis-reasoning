"""Canonical content hashes for frozen protocol objects."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_hash(payload: Any) -> str:
    return sha256_text(canonical_dumps(payload))

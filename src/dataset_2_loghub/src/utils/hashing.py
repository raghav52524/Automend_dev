"""Deterministic hashing utilities for reproducible sampling."""
import hashlib


def stable_hash(value: str) -> int:
    """Return a stable integer hash for a string using MD5.

    Same result across runs and machines — used for deterministic 10% sampling.
    """
    return int(hashlib.md5(value.encode("utf-8")).hexdigest(), 16)


def should_keep(system: str, raw_id: str, pct: int = 10) -> bool:
    """Return True if this row falls in the deterministic sample.

    Args:
        system: system name (e.g. 'linux')
        raw_id: original row identifier
        pct: sample percentage (default 10 → ~10% kept)
    """
    key = f"{system}:{raw_id}"
    return stable_hash(key) % 100 < pct

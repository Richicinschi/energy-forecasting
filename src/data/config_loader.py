"""
config_loader.py — Load and validate project configuration from config.yaml.

Provides a single cached load_config() function used throughout the project.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def _find_config() -> Path:
    """Search for config.yaml from cwd upward."""
    candidates = [
        Path("config.yaml"),
        Path("config/config.yaml"),
    ]
    # Also search parent directories (useful when running from src/ or scripts/)
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        for name in ("config.yaml", "config/config.yaml"):
            p = parent / name
            if p.exists():
                return p
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "config.yaml not found. Run from the project root directory."
    )


@lru_cache(maxsize=1)
def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load and return the project configuration dict (cached after first call)."""
    path = Path(config_path) if config_path else _find_config()
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_ba_list(
    cfg: dict[str, Any],
    *,
    enabled_only: bool = True,
    region: str | None = None,
    max_priority: int | None = None,
) -> list[dict[str, Any]]:
    """Return filtered + sorted list of Balancing Authorities from config.

    Args:
        cfg: loaded config dict
        enabled_only: skip BAs where enabled=False
        region: filter by grid region (eastern / texas / western)
        max_priority: include only BAs with priority <= this value (1 = most important)

    Returns:
        List of BA dicts sorted by priority ascending, then code alphabetically.
    """
    bas = cfg.get("balancing_authorities", [])
    if enabled_only:
        bas = [b for b in bas if b.get("enabled", True)]
    if region:
        bas = [b for b in bas if b.get("region", "").lower() == region.lower()]
    if max_priority is not None:
        bas = [b for b in bas if b.get("priority", 99) <= max_priority]
    return sorted(bas, key=lambda b: (b.get("priority", 99), b["code"]))


def get_ba_codes(cfg: dict[str, Any], **kwargs) -> list[str]:
    """Return just the BA codes from get_ba_list()."""
    return [b["code"] for b in get_ba_list(cfg, **kwargs)]

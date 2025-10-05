"""
tests/conftest.py — Shared pytest fixtures and configuration.

Fixes a Windows-specific issue where pytest-asyncio in strict mode
cannot create the system temp directory. We redirect tmp_path to a
local project directory instead.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture()
def tmp_path(request) -> Path:
    """Local tmp_path fixture — avoids Windows system temp permission issues.

    Creates a unique temp directory inside the project's own tests/_tmp/
    directory and cleans it up after each test.
    """
    base = Path(__file__).parent / "_tmp"
    base.mkdir(exist_ok=True)
    test_dir = Path(tempfile.mkdtemp(dir=base))
    yield test_dir
    shutil.rmtree(test_dir, ignore_errors=True)

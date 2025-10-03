#!/usr/bin/env python3
"""
setup_dirs.py — Create the full project directory structure.

Run once after cloning to initialize all required directories.

Usage:
    python setup_dirs.py
"""

from pathlib import Path


DIRS = [
    # Source packages
    "src",
    "src/data",
    "src/features",
    "src/models",
    "src/evaluation",
    "src/monitoring",
    "src/visualization",
    # Data (gitignored contents, kept dirs)
    "data/raw",
    "data/processed",
    "data/processed/benchmark",
    # Config
    "config",
    # Notebooks
    "notebooks",
    # Scripts (CLI entry points)
    "scripts",
    # Tests
    "tests",
    # Dashboard
    "dashboard",
    "dashboard/components",
    # Models
    "models/saved",
    # Logs & reports
    "logs",
    "evidence_reports",
    # MLflow (gitignored)
    "mlruns",
]

INIT_FILES = [
    "src/__init__.py",
    "src/data/__init__.py",
    "src/features/__init__.py",
    "src/models/__init__.py",
    "src/evaluation/__init__.py",
    "src/monitoring/__init__.py",
    "src/visualization/__init__.py",
    "scripts/__init__.py",
    "tests/__init__.py",
    "dashboard/__init__.py",
    "dashboard/components/__init__.py",
]

GITKEEP_DIRS = [
    "data/raw",
    "data/processed",
    "data/processed/benchmark",
    "models/saved",
    "logs",
    "evidence_reports",
    "mlruns",
    "notebooks",
]

CONFIG_COPY = ("config.yaml", "config/config.yaml")


def main():
    print("Creating project directory structure...")
    print()

    # Create directories
    for d in DIRS:
        path = Path(d)
        path.mkdir(parents=True, exist_ok=True)
        print(f"  [dir]  {d}/")

    print()

    # Create __init__.py files
    for f in INIT_FILES:
        path = Path(f)
        if not path.exists():
            path.write_text('"""Energy Load Forecasting package."""\n')
            print(f"  [init] {f}")
        else:
            print(f"  [skip] {f} (already exists)")

    print()

    # Create .gitkeep files in data/log dirs
    for d in GITKEEP_DIRS:
        gitkeep = Path(d) / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.write_text("")
            print(f"  [keep] {d}/.gitkeep")

    print()

    # Copy config.yaml to config/ if not already there
    src_cfg, dst_cfg = CONFIG_COPY
    if Path(src_cfg).exists() and not Path(dst_cfg).exists():
        import shutil
        shutil.copy2(src_cfg, dst_cfg)
        print(f"  [copy] {src_cfg} -> {dst_cfg}")

    print()
    print("Done! Run `python verify_setup.py` to confirm the setup.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
verify_setup.py — Smoke test for the energy-forecasting project scaffold.

Run this after installing requirements to confirm everything is importable
and the config loads correctly.

Usage:
    python verify_setup.py
    python verify_setup.py --verbose
"""

import argparse
import os
import sys
from pathlib import Path


def check(label: str, ok: bool, detail: str = "") -> bool:
    status = "[OK]  " if ok else "[FAIL]"
    msg = f"  {status} {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return ok


def main():
    parser = argparse.ArgumentParser(description="Verify energy-forecasting project setup")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show extra detail")
    args = parser.parse_args()

    print("=" * 60)
    print("  Energy Load Forecasting -- Setup Verification")
    print("=" * 60)
    print()

    failures = []

    # ── Python version ────────────────────────────────────────────
    print("Python Environment:")
    py_ok = sys.version_info >= (3, 10)
    if not check("Python >= 3.10", py_ok, f"found {sys.version.split()[0]}"):
        failures.append("python_version")
    print()

    # ── Core imports ──────────────────────────────────────────────
    print("Core Dependencies:")
    imports = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("requests", "requests"),
        ("sqlalchemy", "sqlalchemy"),
        ("python-dotenv", "dotenv"),
        ("pyyaml", "yaml"),
        ("mlflow", "mlflow"),
        ("evidently", "evidently"),
        ("matplotlib", "matplotlib"),
        ("plotly", "plotly"),
        ("streamlit", "streamlit"),
        ("holidays", "holidays"),
        ("pyarrow", "pyarrow"),
        ("joblib", "joblib"),
        ("click", "click"),
        ("tqdm", "tqdm"),
        ("pytest", "pytest"),
    ]

    for display_name, module_name in imports:
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "?")
            detail = version if args.verbose else ""
            if not check(display_name, True, detail):
                failures.append(display_name)
        except ImportError as e:
            check(display_name, False, str(e))
            failures.append(display_name)

    print()

    # ── Config loading ────────────────────────────────────────────
    print("Configuration:")
    config_path = Path("config.yaml")
    if not config_path.exists():
        config_path = Path("config/config.yaml")

    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

        eia_ok = "eia" in config and "base_url" in config["eia"]
        check("config.yaml loads", True)
        check("EIA section present", eia_ok, config["eia"]["base_url"] if eia_ok else "missing")

        # Count enabled BAs
        bas = config.get("balancing_authorities", [])
        enabled_bas = [ba for ba in bas if ba.get("enabled", True)]
        check(
            "Balancing Authorities",
            len(enabled_bas) > 0,
            f"{len(enabled_bas)} enabled BAs across "
            f"{len(set(ba['region'] for ba in enabled_bas))} grid regions"
        )

        models_ok = "models" in config
        check("Models config present", models_ok,
              ", ".join(config["models"].keys()) if models_ok else "missing")

        monitoring_ok = "monitoring" in config
        check("Monitoring config present", monitoring_ok,
              f"drift threshold = {config['monitoring']['drift_threshold_pct']}%" if monitoring_ok else "missing")

    except FileNotFoundError:
        check("config.yaml loads", False, "file not found — run from project root")
        failures.append("config")
    except Exception as e:
        check("config.yaml loads", False, str(e))
        failures.append("config")

    print()

    # ── Directory structure ───────────────────────────────────────
    print("Directory Structure:")
    expected_dirs = [
        "src",
        "src/data",
        "src/features",
        "src/models",
        "src/evaluation",
        "src/monitoring",
        "src/visualization",
        "data/raw",
        "data/processed",
        "notebooks",
        "scripts",
        "tests",
        "dashboard",
        "logs",
        "models/saved",
        "config",
    ]
    for d in expected_dirs:
        exists = Path(d).is_dir()
        if not check(d + "/", exists):
            failures.append(f"dir:{d}")

    print()

    # ── .env / API key ────────────────────────────────────────────
    print("Environment / API:")
    env_file = Path(".env")
    env_example = Path("env.example")
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("EIA_API_KEY", "")
        has_key = bool(api_key) and api_key != "your_api_key_here"
        check(".env found", True)
        check("EIA_API_KEY set", has_key,
              "key looks real" if has_key else "set EIA_API_KEY in .env")
    else:
        check(".env found", False, "copy env.example to .env and add your API key")
        check("EIA_API_KEY set", False, "requires .env file")

    print()

    # ── Summary ───────────────────────────────────────────────────
    print("=" * 60)
    if not failures:
        print("  ALL CHECKS PASSED — project is ready")
        print()
        print("  Next steps:")
        print("    1. Copy env.example to .env and add your EIA_API_KEY")
        print("    2. python scripts/fetch_data.py --region MISO")
        print("    3. python scripts/run_pipeline.py --region MISO")
    else:
        critical = [f for f in failures if not f.startswith("dir:")]
        dir_failures = [f for f in failures if f.startswith("dir:")]
        if critical:
            print(f"  {len(critical)} CRITICAL FAILURE(S) — fix before proceeding:")
            for f in critical:
                print(f"    - {f}")
        if dir_failures:
            print(f"  {len(dir_failures)} missing director(ies) — run:")
            print("    python setup_dirs.py")
    print("=" * 60)

    return 0 if not [f for f in failures if not f.startswith("dir:")] else 1


if __name__ == "__main__":
    sys.exit(main())

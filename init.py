#!/usr/bin/env python3
"""
Initialization script for the Energy Load Forecasting project.
Run this to verify the project setup and environment.
"""

import sys
import os
from pathlib import Path

# Colors for terminal output
GREEN = ""
YELLOW = ""
RED = ""
RESET = ""
BOLD = ""

def print_section(title: str):
    """Print a section header."""
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}{title}{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")

def print_success(message: str):
    print(f"[OK] {message}")

def print_warning(message: str):
    print(f"[WARN] {message}")

def print_error(message: str):
    print(f"[FAIL] {message}")

def check_python_version() -> bool:
    """Check Python version is 3.12+."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 12:
        print_success(f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python {version.major}.{version.minor}.{version.micro} (requires 3.12+)")
        return False

def check_virtual_env() -> bool:
    """Check if running in virtual environment."""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print_success(f"Virtual environment: {sys.prefix}")
        return True
    else:
        print_warning("Not running in a virtual environment")
        return False

def check_env_file() -> bool:
    """Check if .env file exists with EIA_API_KEY."""
    env_path = Path(".env")
    if not env_path.exists():
        print_error(".env file not found")
        print("  Run: cp env.example .env")
        print("  Then edit .env and add your EIA_API_KEY")
        return False
    
    with open(env_path) as f:
        content = f.read()
        if "EIA_API_KEY" in content and "your_api_key_here" not in content:
            print_success(".env file with EIA_API_KEY configured")
            return True
        else:
            print_warning(".env file exists but EIA_API_KEY not configured")
            print("  Edit .env and add your EIA_API_KEY from https://www.eia.gov/opendata/")
            return False

def check_dependencies() -> bool:
    """Check if required packages are installed."""
    required = [
        "pandas",
        "numpy",
        "scikit-learn",
        "sqlalchemy",
        "requests",
        "pyyaml",
        "plotly",
        "streamlit",
        "mlflow",
        "holidays",
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print_error(f"Missing packages: {', '.join(missing)}")
        print(f"  Run: pip install -r requirements.txt")
        return False
    else:
        print_success("All required dependencies installed")
        return True

def check_directories() -> bool:
    """Check if required directories exist."""
    required_dirs = [
        "data",
        "data/raw",
        "data/processed",
        "models",
        "models/saved",
        "logs",
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print_success(f"Directory: {dir_path}/")
        else:
            print_warning(f"Directory missing: {dir_path}/")
            path.mkdir(parents=True, exist_ok=True)
            print(f"  Created: {dir_path}/")
            all_exist = False
    
    return all_exist

def check_config() -> bool:
    """Check if config.yaml is valid."""
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        print_error("config/config.yaml not found")
        return False
    
    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        print_success("config.yaml is valid YAML")
        
        # Check key sections
        required_keys = ["eia", "balancing_authorities", "data", "models"]
        missing = [k for k in required_keys if k not in config]
        if missing:
            print_warning(f"Missing config sections: {', '.join(missing)}")
            return False
        
        bas = config.get("balancing_authorities", [])
        print_success(f"Configured {len(bas)} balancing authorities")
        return True
    except Exception as e:
        print_error(f"Error parsing config.yaml: {e}")
        return False

def main():
    """Run all checks."""
    print(f"{BOLD}")
    print("==============================================================")
    print("      Energy Load Forecasting - Project Initialization        ")
    print("==============================================================")
    print(f"{RESET}")
    
    results = []
    
    print_section("Environment Check")
    results.append(("Python 3.12+", check_python_version()))
    results.append(("Virtual Environment", check_virtual_env()))
    
    print_section("Configuration Check")
    results.append((".env file", check_env_file()))
    results.append(("config.yaml", check_config()))
    
    print_section("Dependencies Check")
    results.append(("Required packages", check_dependencies()))
    
    print_section("Directory Structure")
    results.append(("Required directories", check_directories()))
    
    # Summary
    print_section("Summary")
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    if passed == total:
        print(f"All checks passed! Project is ready to use.")
        print("\nNext steps:")
        print("  1. Fetch data: python scripts/fetch_data.py --region MISO")
        print("  2. Run pipeline: python scripts/run_pipeline.py --region MISO")
        print("  3. Launch dashboard: streamlit run dashboard/app.py")
        return 0
    else:
        print(f"{passed}/{total} checks passed")
        print(f"\nPlease fix the issues above and run init again.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

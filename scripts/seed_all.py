#!/usr/bin/env python
"""
Master Seed Script - Unified data seeding for E2E testing
=========================================================
Runs all dataset seed scripts to prepare the monorepo for E2E testing with Airflow.

Usage:
    python scripts/seed_all.py              # Seed all datasets
    python scripts/seed_all.py --ds1        # Seed only DS1
    python scripts/seed_all.py --ds1 --ds3  # Seed DS1 and DS3
    python scripts/seed_all.py --download   # Also download external data (DS2, DS5, DS6)

Prerequisites:
    - Conda environment: mlops_project
    - Working directory: Project root (Automend/)
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import centralized paths for directory creation
try:
    from src.config.paths import ensure_dirs_exist
    HAS_PATHS = True
except ImportError:
    HAS_PATHS = False


def run_script(script_path: Path, description: str) -> bool:
    """Run a Python script and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script:  {script_path}")
    print('='*60)
    
    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=False,
            text=True
        )
        if result.returncode == 0:
            print(f"SUCCESS: {description}")
            return True
        else:
            print(f"FAILED: {description} (exit code {result.returncode})")
            return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def seed_ds1() -> bool:
    """Seed Dataset 1 (Alibaba) - generates 3 sample CSVs."""
    return run_script(
        PROJECT_ROOT / "src" / "dataset_1_alibaba" / "scripts" / "seed_data.py",
        "DS1 (Alibaba) - Generate sample CSVs"
    )


def seed_ds2(download: bool = False) -> bool:
    """Seed Dataset 2 (Loghub) - downloads from GitHub if requested."""
    if download:
        return run_script(
            PROJECT_ROOT / "src" / "dataset_2_loghub" / "src" / "ingest" / "download_data.py",
            "DS2 (Loghub) - Download from GitHub"
        )
    else:
        print("\n" + "="*60)
        print("DS2 (Loghub) - Skipping download (use --download to fetch)")
        print("="*60)
        print("  To download manually:")
        print("    python src/dataset_2_loghub/src/ingest/download_data.py")
        print("  Files will be saved to: data/raw/ds2_loghub/")
        return True


def seed_ds3() -> bool:
    """Seed Dataset 3 (StackOverflow) - generates sample Q&A CSVs."""
    return run_script(
        PROJECT_ROOT / "src" / "dataset_3_stackoverflow" / "scripts" / "seed_data.py",
        "DS3 (StackOverflow) - Generate sample CSVs"
    )


def seed_ds4() -> bool:
    """Seed Dataset 4 (Synthetic) - seeds prompts database."""
    return run_script(
        PROJECT_ROOT / "src" / "dataset_4_synthetic" / "scripts" / "seed_prompts.py",
        "DS4 (Synthetic) - Seed prompts database"
    )


def seed_ds5(download: bool = False) -> bool:
    """Seed Dataset 5 (Glaive) - downloads from HuggingFace if requested."""
    if download:
        return run_script(
            PROJECT_ROOT / "src" / "dataset_5_glaive" / "scripts" / "data_acquisition.py",
            "DS5 (Glaive) - Download from HuggingFace"
        )
    else:
        print("\n" + "="*60)
        print("DS5 (Glaive) - Skipping download (use --download to fetch)")
        print("="*60)
        print("  To download manually:")
        print("    python src/dataset_5_glaive/scripts/data_acquisition.py")
        print("  Requires: HuggingFace datasets library + network access")
        print("  Files will be saved to: data/raw/ds5_glaive/")
        return True


def seed_ds6(download: bool = False) -> bool:
    """Seed Dataset 6 (The Stack) - downloads from HuggingFace if requested."""
    if download:
        return run_script(
            PROJECT_ROOT / "src" / "dataset_6_the_stack" / "scripts" / "download" / "stack_iac_sample.py",
            "DS6 (The Stack) - Download from HuggingFace"
        )
    else:
        print("\n" + "="*60)
        print("DS6 (The Stack) - Skipping download (use --download to fetch)")
        print("="*60)
        print("  To download manually:")
        print("    python src/dataset_6_the_stack/scripts/download/stack_iac_sample.py")
        print("  Requires: HuggingFace datasets library + network access")
        print("  Optional: Set HF_TOKEN for gated datasets")
        print("  Files will be saved to: data/raw/ds6_the_stack/")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Seed all datasets for E2E testing with Airflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/seed_all.py              # Seed local data only (DS1, DS3, DS4)
    python scripts/seed_all.py --download   # Also download external data (DS2, DS5, DS6)
    python scripts/seed_all.py --ds1 --ds4  # Seed only specific datasets
        """
    )
    
    parser.add_argument("--ds1", action="store_true", help="Seed DS1 (Alibaba)")
    parser.add_argument("--ds2", action="store_true", help="Seed DS2 (Loghub)")
    parser.add_argument("--ds3", action="store_true", help="Seed DS3 (StackOverflow)")
    parser.add_argument("--ds4", action="store_true", help="Seed DS4 (Synthetic)")
    parser.add_argument("--ds5", action="store_true", help="Seed DS5 (Glaive)")
    parser.add_argument("--ds6", action="store_true", help="Seed DS6 (The Stack)")
    parser.add_argument("--download", action="store_true", 
                       help="Download external data (DS2 from GitHub, DS5/DS6 from HuggingFace)")
    
    args = parser.parse_args()
    
    # If no specific datasets selected, seed all
    seed_specific = any([args.ds1, args.ds2, args.ds3, args.ds4, args.ds5, args.ds6])
    
    print("="*60)
    print("AUTOMEND E2E TEST DATA SEEDING")
    print("="*60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Download mode: {'enabled' if args.download else 'disabled'}")
    
    # Ensure data directories exist
    if HAS_PATHS:
        print("\nCreating data directories...")
        ensure_dirs_exist()
        print("  Data directories ready")
    
    results = {}
    
    # Seed datasets
    if not seed_specific or args.ds1:
        results["DS1 (Alibaba)"] = seed_ds1()
    
    if not seed_specific or args.ds2:
        results["DS2 (Loghub)"] = seed_ds2(download=args.download)
    
    if not seed_specific or args.ds3:
        results["DS3 (StackOverflow)"] = seed_ds3()
    
    if not seed_specific or args.ds4:
        results["DS4 (Synthetic)"] = seed_ds4()
    
    if not seed_specific or args.ds5:
        results["DS5 (Glaive)"] = seed_ds5(download=args.download)
    
    if not seed_specific or args.ds6:
        results["DS6 (The Stack)"] = seed_ds6(download=args.download)
    
    # Summary
    print("\n" + "="*60)
    print("SEEDING SUMMARY")
    print("="*60)
    
    success_count = 0
    for dataset, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {dataset}: {status}")
        if success:
            success_count += 1
    
    print(f"\nTotal: {success_count}/{len(results)} succeeded")
    
    if success_count == len(results):
        print("\nNext steps:")
        print("  1. Copy .env.example to .env and set your API keys")
        print("  2. Start Airflow: docker-compose up -d")
        print("  3. Access UI: http://localhost:8080 (airflow/airflow)")
        print("  4. Trigger master_track_a or master_track_b DAG")
        return 0
    else:
        print("\nSome seeds failed - check logs above")
        return 1


if __name__ == "__main__":
    sys.exit(main())

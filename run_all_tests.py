#!/usr/bin/env python
"""
Run all tests in the Automend MLOps Monorepo.

This script runs:
1. Root-level tests (combiner, integration, e2e, etc.)
2. Individual dataset tests (each from their own directory)

Usage:
    python run_all_tests.py          # Run all tests
    python run_all_tests.py --root   # Run only root tests
    python run_all_tests.py --ds1    # Run only dataset 1 tests
"""

import subprocess
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATASETS = [
    "dataset_1_alibaba",
    "dataset_2_loghub",
    "dataset_3_stackoverflow",
    "dataset_4_synthetic",
    "dataset_5_glaive",
    "dataset_6_the_stack",
]


def run_root_tests():
    """Run root-level tests."""
    print("\n" + "=" * 60)
    print("RUNNING ROOT-LEVEL TESTS")
    print("=" * 60)
    
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=PROJECT_ROOT,
    )
    return result.returncode


def run_dataset_tests(dataset_name):
    """Run tests for a specific dataset."""
    dataset_dir = PROJECT_ROOT / "src" / dataset_name
    tests_dir = dataset_dir / "tests"
    
    if not tests_dir.exists():
        print(f"  No tests directory for {dataset_name}")
        return 0
    
    test_files = list(tests_dir.glob("test_*.py"))
    if not test_files:
        print(f"  No test files for {dataset_name}")
        return 0
    
    print(f"\n" + "-" * 60)
    print(f"RUNNING TESTS FOR: {dataset_name}")
    print("-" * 60)
    
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "--rootdir=."],
        cwd=dataset_dir,
    )
    return result.returncode


def main():
    args = sys.argv[1:]
    
    results = {}
    
    if not args or "--root" in args:
        results["root"] = run_root_tests()
    
    if not args:
        # Run all dataset tests
        for ds in DATASETS:
            results[ds] = run_dataset_tests(ds)
    else:
        # Run specific dataset tests
        for ds in DATASETS:
            flag = f"--{ds.replace('dataset_', 'ds')}"
            if flag in args:
                results[ds] = run_dataset_tests(ds)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, code in results.items():
        status = "PASSED" if code == 0 else f"FAILED (exit code {code})"
        print(f"  {name}: {status}")
    
    # Return non-zero if any tests failed
    return max(results.values()) if results else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Run the shared, prefix-aware bootstrap benchmark runner."""

from pathlib import Path
import runpy
import sys


SOURCE_DIR = Path(__file__).resolve().parents[2] / "bootstrap_multi_gpu"
sys.path.insert(0, str(SOURCE_DIR))
runpy.run_path(str(SOURCE_DIR / "bootstrap_runner.py"), run_name="__main__")

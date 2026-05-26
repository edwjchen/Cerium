#!/usr/bin/env python3
"""Run the shared bootstrap compiler from the fusion-ablation directory."""

from pathlib import Path
import runpy
import sys


SOURCE_DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "bootstrap_32K"
sys.path.insert(0, str(SOURCE_DIR))
runpy.run_path(str(SOURCE_DIR / "bootstrap.py"), run_name="__main__")

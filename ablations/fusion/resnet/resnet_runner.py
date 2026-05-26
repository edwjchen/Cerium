#!/usr/bin/env python3
"""Run the shared ResNet runner against fusion-ablation outputs."""

from pathlib import Path
import os
import runpy
import sys


SOURCE_DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "resnet20"
sys.path.insert(0, str(SOURCE_DIR))
os.chdir(SOURCE_DIR)
runpy.run_path(str(SOURCE_DIR / "resnet_runner.py"), run_name="__main__")

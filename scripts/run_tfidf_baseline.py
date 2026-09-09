#!/usr/bin/env python3
"""
scripts/run_tfidf_baseline.py — Root runner shortcut for TF-IDF baseline.
"""
import os
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from training.run_tfidf_baseline import main

if __name__ == "__main__":
    main()

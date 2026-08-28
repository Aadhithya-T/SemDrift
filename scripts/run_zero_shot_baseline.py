#!/usr/bin/env python3
import os
import sys

# Ensure scripts directory and project root are in sys.path
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from training.run_zero_shot_baseline import main

if __name__ == "__main__":
    main()

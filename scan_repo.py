#!/usr/bin/env python3
"""
SemDrift CLI Root Entrypoint.
Delegates directly to scripts/scan_repo.py so the scanner can be invoked as either:
    python scan_repo.py [args]
or:
    python scripts/scan_repo.py [args]
"""
import os
import sys
import runpy

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    target_script = os.path.join(project_root, "scripts", "scan_repo.py")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    runpy.run_path(target_script, run_name="__main__")

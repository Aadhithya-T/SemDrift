"""
SemDrift Dataset Expansion - Step 1: Repository Discovery & Cloning
Discovers and clones high-quality target repositories into data/expansion_repos/
and records provenance, licenses, and commit SHAs into repository_manifest.json.
"""

import os
import json
import subprocess
from pathlib import Path

EXPANSION_REPOS = [
    {
        "name": "scipy",
        "url": "https://github.com/scipy/scipy.git",
        "domain": "Scientific Computing",
        "license": "BSD-3-Clause",
        "description": "Scientific library for mathematics, science, and engineering."
    },
    {
        "name": "matplotlib",
        "url": "https://github.com/matplotlib/matplotlib.git",
        "domain": "Data Visualization",
        "license": "PSF/BSD-compatible",
        "description": "Comprehensive library for creating static, animated, and interactive visualizations."
    },
    {
        "name": "seaborn",
        "url": "https://github.com/mwaskom/seaborn.git",
        "domain": "Statistical Visualization",
        "license": "BSD-3-Clause",
        "description": "Statistical data visualization based on matplotlib."
    },
    {
        "name": "sqlalchemy",
        "url": "https://github.com/sqlalchemy/sqlalchemy.git",
        "domain": "Database & ORM",
        "license": "MIT",
        "description": "Database toolkit and Object Relational Mapper for Python."
    },
    {
        "name": "aiohttp",
        "url": "https://github.com/aio-libs/aiohttp.git",
        "domain": "Async Networking & Web",
        "license": "Apache-2.0",
        "description": "Asynchronous HTTP client/server framework for asyncio."
    },
    {
        "name": "celery",
        "url": "https://github.com/celery/celery.git",
        "domain": "Distributed Systems & Tasks",
        "license": "BSD-3-Clause",
        "description": "Distributed task queue for real-time operation and scheduling."
    },
    {
        "name": "networkx",
        "url": "https://github.com/networkx/networkx.git",
        "domain": "Graph Algorithms & Networks",
        "license": "BSD-3-Clause",
        "description": "Software package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks."
    },
    {
        "name": "pillow",
        "url": "https://github.com/python-pillow/Pillow.git",
        "domain": "Image Processing",
        "license": "HPND",
        "description": "The friendly PIL fork for Python image processing."
    },
    {
        "name": "starlette",
        "url": "https://github.com/encode/starlette.git",
        "domain": "ASGI Web Framework",
        "license": "BSD-3-Clause",
        "description": "Lightweight ASGI framework/toolkit for async web services."
    },
    {
        "name": "uvicorn",
        "url": "https://github.com/encode/uvicorn.git",
        "domain": "ASGI Web Server",
        "license": "BSD-3-Clause",
        "description": "An ASGI web server implementation for Python."
    },
    {
        "name": "scrapy",
        "url": "https://github.com/scrapy/scrapy.git",
        "domain": "Web Scraping & Crawling",
        "license": "BSD-3-Clause",
        "description": "High-level web crawling and scraping framework."
    },
    {
        "name": "peewee",
        "url": "https://github.com/coleifer/peewee.git",
        "domain": "Lightweight ORM",
        "license": "MIT",
        "description": "A small, expressive ORM for Python."
    },
    {
        "name": "attrs",
        "url": "https://github.com/python-attrs/attrs.git",
        "domain": "Classes & Data Modeling",
        "license": "MIT",
        "description": "Classes Without Boilerplate."
    },
    {
        "name": "loguru",
        "url": "https://github.com/Delgan/loguru.git",
        "domain": "Logging & Diagnostics",
        "license": "MIT",
        "description": "Python logging made stupidly simple."
    },
    {
        "name": "mypy",
        "url": "https://github.com/python/mypy.git",
        "domain": "Static Type Checking",
        "license": "MIT",
        "description": "Optional static typing for Python."
    },
    {
        "name": "sympy",
        "url": "https://github.com/sympy/sympy.git",
        "domain": "Symbolic Mathematics",
        "license": "BSD-3-Clause",
        "description": "A Python library for symbolic mathematics."
    }
]

def main():
    base_dir = Path(__file__).resolve().parents[2]
    repos_dir = base_dir / "data" / "expansion_repos"
    manifest_dir = base_dir / "data" / "real_dataset_expansion_v1" / "manifests"
    
    repos_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest = []

    print("[*] Starting repository discovery and acquisition for dataset expansion...", flush=True)

    for item in EXPANSION_REPOS:
        name = item["name"]
        url = item["url"]
        dest = repos_dir / name

        if not dest.exists():
            print(f"[*] Cloning {name} (depth 300) from {url}...", flush=True)
            try:
                subprocess.run(
                    ["git", "clone", "--depth", "300", url, str(dest)],
                    check=True, timeout=120, capture_output=True, text=True,
                    encoding="utf-8", errors="replace"
                )
                print(f"    [+] Successfully cloned {name}", flush=True)
            except Exception as e:
                print(f"    [-] Failed to clone {name}: {e}", flush=True)
                continue
        else:
            print(f"[*] {name} already cloned at {dest}", flush=True)

        # Retrieve exact immutable commit SHA
        try:
            rev_res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=dest, capture_output=True, text=True,
                encoding="utf-8", errors="replace"
            )
            sha = rev_res.stdout.strip() if rev_res.returncode == 0 else "UNKNOWN"
        except Exception:
            sha = "UNKNOWN"

        manifest.append({
            "repository": name,
            "url": url,
            "domain": item["domain"],
            "license": item["license"],
            "description": item["description"],
            "immutable_commit_sha": sha,
            "local_path": str(dest)
        })

    manifest_file = manifest_dir / "repository_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[+] Repository discovery complete. Manifest saved to {manifest_file}", flush=True)
    print(f"[+] Total expansion repositories ready: {len(manifest)}", flush=True)

if __name__ == "__main__":
    main()

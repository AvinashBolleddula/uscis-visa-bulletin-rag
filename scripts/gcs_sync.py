# gcs_sync.py is a thin wrapper around gsutil rsync that lets you:
# Push a local Chroma directory → GCS
# Pull a GCS directory → local filesystem
# This makes your vector store reproducible, portable, cloud-friendly and decoupled from notebooks or local machines


#!/usr/bin/env python3
"""
Sync a local folder to/from GCS using gsutil.

Examples:
  # upload local chroma -> gcs
  uv run python scripts/gcs_sync.py --mode push --local ./chroma --gcs gs://BUCKET/PREFIX

  # download gcs -> local chroma
  uv run python scripts/gcs_sync.py --mode pull --local ./chroma --gcs gs://BUCKET/PREFIX
"""
# __future__ annotations → cleaner typing (Python ≥3.10 best practice)
from __future__ import annotations

# argparse → parse CLI arguments (--mode, --local, --gcs)
import argparse
# subprocess → execute gsutil commands
import subprocess
# Path → safe, OS-independent path handling
from pathlib import Path

# Helper function: run
# Prints the exact command being run (great for CI logs)
# Executes the command
# If the command fails → workflow fails immediately
def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.check_call(cmd)

# main entry point
def main() -> int:
    # Argument parsing
    ap = argparse.ArgumentParser()
    # Direction of sync (push or pull)
    ap.add_argument("--mode", choices=["push", "pull"], required=True)
    # Local folder path (./chroma)
    ap.add_argument("--local", required=True)
    # GCS path (gs://bucket/prefix)
    ap.add_argument("--gcs", required=True)  # gs://bucket/prefix
    args = ap.parse_args()
   
    # Ensures the local directory exists before syncing
    # Prevents gsutil rsync from failing due to missing path
    # Works for both push & pull
    local = Path(args.local)
    local.mkdir(parents=True, exist_ok=True)

    # push mode
    # Syncs local → GCS
    # -m = multi-threaded (faster)
    # -r = recursive
    # Trailing behavior copies contents, not parent folder
    # ./chroma/
        # ├── index/
        # ├── collections/
        # ├── sqlite.db
    # to gs://bucket/chroma/
    if args.mode == "push":
        # ensure trailing slash so we copy contents
        run(["gsutil", "-m", "rsync", "-r", str(local), args.gcs])
    # Pull mode
    # Syncs GCS → local
    # used by streamlit app startup, local dev wnvironment, cloud run container etc.
    else:
        run(["gsutil", "-m", "rsync", "-r", args.gcs, str(local)])

    return 0

# Script execution guard
# Allows importing this file without executing it
# 
if __name__ == "__main__":
    raise SystemExit(main())


# How this fits into your system (important)
# In GitHub Actions : uv run python scripts/gcs_sync.py --mode push --local ./chroma --gcs gs://...
# Builds vectors → pushes to GCS


# In Streamlit / Cloud Run
# python scripts/gcs_sync.py --mode pull --local ./chroma --gcs gs://...
# Pulls vectors → loads Chroma locally → QA works
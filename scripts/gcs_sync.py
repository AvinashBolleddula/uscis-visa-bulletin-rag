#!/usr/bin/env python3
"""
Sync a local folder to/from GCS using google-cloud-storage (no gsutil needed).

Examples:
  uv run python scripts/gcs_sync.py --mode pull --local ./chroma --gcs gs://BUCKET/PREFIX
  uv run python scripts/gcs_sync.py --mode push --local ./chroma --gcs gs://BUCKET/PREFIX
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

from google.cloud import storage


def parse_gs_uri(gs_uri: str) -> Tuple[str, str]:
    if not gs_uri.startswith("gs://"):
        raise ValueError("Expected gs://bucket/prefix")
    rest = gs_uri[len("gs://") :]
    parts = rest.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    prefix = prefix.strip("/")
    if prefix:
        prefix += "/"
    return bucket, prefix


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def pull_folder(bucket_name: str, prefix: str, local_dir: Path) -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = list(client.list_blobs(bucket_name, prefix=prefix))
    if not blobs:
        raise RuntimeError(f"No objects found at gs://{bucket_name}/{prefix}")

    for b in blobs:
        # skip "directory markers"
        if b.name.endswith("/"):
            continue
        rel = b.name[len(prefix) :] if b.name.startswith(prefix) else b.name
        out_path = local_dir / rel
        ensure_parent(out_path)
        b.download_to_filename(str(out_path))


def push_folder(bucket_name: str, prefix: str, local_dir: Path) -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for p in local_dir.rglob("*"):
        if p.is_dir():
            continue
        rel = p.relative_to(local_dir).as_posix()
        blob = bucket.blob(prefix + rel)
        blob.upload_from_filename(str(p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["push", "pull"], required=True)
    ap.add_argument("--local", required=True)
    ap.add_argument("--gcs", required=True)  # gs://bucket/prefix
    args = ap.parse_args()

    local = Path(args.local)
    local.mkdir(parents=True, exist_ok=True)

    bucket, prefix = parse_gs_uri(args.gcs)

    if args.mode == "pull":
        pull_folder(bucket, prefix, local)
    else:
        if not local.exists():
            raise RuntimeError(f"Local dir not found: {local}")
        push_folder(bucket, prefix, local)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
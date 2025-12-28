#!/usr/bin/env python3
"""
Download USCIS Visa Bulletin PDFs into ./data

Usage:
  uv run python scripts/download_pdfs.py --year 2025 --months 1 2 3 4 5
  uv run python scripts/download_pdfs.py --year 2025 --all

Env (optional):
  DATA_DIR=data
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import requests


BASE_URL = "https://travel.state.gov/content/dam/visas/Bulletins/"

MONTH_MAP = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}


def download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()

    with out_path.open("wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 64):
            if chunk:
                f.write(chunk)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--months", nargs="*", type=int, default=None, help="Month numbers like 1 2 3 ...")
    parser.add_argument("--all", action="store_true", help="Download all 12 months")
    args = parser.parse_args()

    data_dir = Path(os.getenv("DATA_DIR", "data"))
    year = args.year

    if args.all:
        months = list(range(1, 13))
    else:
        months = args.months or list(range(1, 12))  # default similar to your notebook (Jan–Nov)

    for m in months:
        if m not in MONTH_MAP:
            raise ValueError(f"Invalid month: {m}")
        month_name = MONTH_MAP[m]
        filename = f"visabulletin_{month_name}{year}.pdf"
        url = f"{BASE_URL}{filename}"
        out_path = data_dir / filename

        print(f"⬇️  {filename}")
        try:
            download_file(url, out_path)
            print(f"✅ Saved to {out_path}")
        except Exception as e:
            print(f"❌ Failed {filename}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Download product images listed in the processed image manifest."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd


USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)


def download_one(item_id: str, image_url: str, output_path: Path, timeout: int) -> tuple[str, str]:
    """Download one image and return its status."""
    if output_path.exists() and output_path.stat().st_size > 0:
        return item_id, "downloaded"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(image_url, headers={"User-Agent": USER_AGENT})

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
        if not data:
            return item_id, "empty"
        output_path.write_bytes(data)
        return item_id, "downloaded"
    except (urllib.error.URLError, TimeoutError, ValueError):
        return item_id, "failed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download product images")
    parser.add_argument("--data_dir", type=str, default="data/processed/all_beauty")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    manifest_path = data_dir / "image_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Image manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    working_manifest = manifest.head(args.limit).copy() if args.limit else manifest.copy()

    if "download_status" not in manifest.columns:
        manifest["download_status"] = "pending"
    if "download_status" not in working_manifest.columns:
        working_manifest["download_status"] = "pending"

    tasks = []
    for idx, row in working_manifest.iterrows():
        image_url = str(row.get("image_url") or "").strip()
        local_path = str(row.get("local_image_path") or "").strip()
        if not image_url or not local_path:
            manifest.at[idx, "download_status"] = "missing_url"
            continue

        output_path = data_dir / local_path
        if args.overwrite and output_path.exists():
            output_path.unlink()

        tasks.append((idx, row["item_id"], image_url, output_path))

    downloaded = 0
    failed = 0

    with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_idx = {
            executor.submit(download_one, item_id, image_url, output_path, args.timeout): idx
            for idx, item_id, image_url, output_path in tasks
        }
        for future in futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            item_id, status = future.result()
            manifest.at[idx, "download_status"] = status
            if status == "downloaded":
                downloaded += 1
            elif status != "missing_url":
                failed += 1
            print(f"{item_id}: {status}")

    manifest.to_csv(manifest_path, index=False)
    print(f"Downloaded: {downloaded}")
    print(f"Failed: {failed}")
    print(f"Manifest updated: {manifest_path}")


if __name__ == "__main__":
    main()

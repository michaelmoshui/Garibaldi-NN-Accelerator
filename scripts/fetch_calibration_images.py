#!/usr/bin/env python3
"""Fetch exactly NUM_IMAGES calibration images for PTQ.

Downloads individual 224×224 JPEGs from Picsum Photos (a free public photo CDN).
Total transfer: ~1–2 MB. No archive, no extra dependencies.

Output: scripts/data/activation_calibration/

Usage:
    python scripts/fetch_calibration_images.py
"""

import urllib.request
from pathlib import Path

NUM_IMAGES = 100   # number of calibration images to fetch

SCRIPTS_DIR = Path(__file__).resolve().parent
OUT_DIR     = SCRIPTS_DIR / "data" / "activation_calibration"

# Picsum serves a deterministic photo for each seed value.
# URL pattern: https://picsum.photos/seed/<seed>/224/224
PICSUM_URL  = "https://picsum.photos/seed/{seed}/224/224"


def main() -> None:
    if OUT_DIR.exists():
        for f in OUT_DIR.iterdir():
            f.unlink()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[info] fetching {NUM_IMAGES} images from picsum.photos ...")
    for i in range(NUM_IMAGES):
        url = PICSUM_URL.format(seed=i)
        dst = OUT_DIR / f"{i:03d}.jpg"
        urllib.request.urlretrieve(url, dst)
        print(f"\r  {i + 1}/{NUM_IMAGES}", end="", flush=True)

    print(f"\n[ok] saved {NUM_IMAGES} calibration images -> {OUT_DIR}")
    print("     point convert_to_tflite.py CALIB_DIR at this folder.")


if __name__ == "__main__":
    main()

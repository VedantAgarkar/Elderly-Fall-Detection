"""
URFD ADL Downloader and Extractor
Downloads and unzips genuine Activities of Daily Living (ADL) sequences
from the official University of Rzeszów benchmark repository.
"""

import os
import sys
import time
import zipfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_URL = "https://fenix.ur.edu.pl/mkepski/ds/data"
TARGET_DIR = Path("/home/endurance/Projects/Elderly Fall Detection/Training data")
TEMP_DIR = Path("/tmp/urfd_downloads")

def download_and_extract_sequence(seq_id: int):
    seq_name = f"adl-{seq_id:02d}-cam0-rgb"
    target_seq_dir = TARGET_DIR / seq_name

    # Skip if already extracted and has frames
    if target_seq_dir.exists() and len(list(target_seq_dir.glob("*.png"))) > 50:
        print(f"[{seq_name}] Already present with {len(list(target_seq_dir.glob('*.png')))} frames. Skipping.")
        return seq_name, True, "Already present"

    zip_filename = f"{seq_name}.zip"
    url = f"{BASE_URL}/{zip_filename}"
    temp_zip_path = TEMP_DIR / zip_filename

    t0 = time.time()
    try:
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        # Download
        urllib.request.urlretrieve(url, temp_zip_path)
        dl_time = time.time() - t0
        file_size_mb = temp_zip_path.stat().st_size / (1024 * 1024)

        # Unpack directly to TARGET_DIR
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(TARGET_DIR)

        # Cleanup zip to save disk space
        temp_zip_path.unlink()

        n_frames = len(list(target_seq_dir.glob("*.png")))
        msg = f"Downloaded {file_size_mb:.1f} MB in {dl_time:.1f}s ({file_size_mb/dl_time:.2f} MB/s) -> {n_frames} frames extracted."
        print(f"[{seq_name}] Success: {msg}")
        return seq_name, True, msg
    except Exception as e:
        if temp_zip_path.exists():
            temp_zip_path.unlink()
        print(f"[{seq_name}] Failed: {e}")
        return seq_name, False, str(e)

def main():
    # We download ADL sequences 01 through 20 (20 distinct ADL activities: sitting, bending, walking, picking up, lying down)
    sequences = list(range(1, 21))
    print(f"Starting parallel acquisition for ADL sequences {sequences[0]} to {sequences[-1]}...")
    t_start = time.time()

    max_workers = 3
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_and_extract_sequence, sid): sid for sid in sequences}
        for fut in as_completed(futures):
            seq_name, ok, msg = fut.result()
            results[seq_name] = (ok, msg)

    success_count = sum(1 for ok, _ in results.values() if ok)
    total_time = time.time() - t_start
    print(f"\nFinished: {success_count}/{len(sequences)} sequences processed successfully in {total_time:.1f}s.")

if __name__ == "__main__":
    main()

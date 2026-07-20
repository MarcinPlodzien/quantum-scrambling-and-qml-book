#!/usr/bin/env python3
"""
download_santafe.py
===================
Fetch and *verify* Data Set A of the Santa Fe Time Series Prediction Competition
(far-infrared laser), saving it as ``santafe_laser.npy`` next to this script.

The competition data has been public for decades and ships with several
reservoir-computing packages, so this script does not depend on any single URL:
it obtains a copy however it can and then checks it against the fixed fingerprint
in ``santafe_provenance.md`` (length, first samples, SHA-256).  If the check fails
it refuses to overwrite the file, so you can never silently end up with a
different series than the one used for the book figure.

Usage:
    python download_santafe.py            # verify existing file, or fetch+verify
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "santafe_laser.npy"

# Fingerprint of the exact series used in the book (see santafe_provenance.md).
N_EXPECTED = 10093
FIRST5 = [86, 141, 95, 41, 22]
SHA32 = "cc4ace0cb515b4feb5f70ac3998ebe0d"


def verify(arr):
    arr = np.asarray(arr).astype(np.int64).reshape(-1, 1)
    if arr.size != N_EXPECTED:
        return False, f"length {arr.size} != {N_EXPECTED}"
    if arr.ravel()[:5].tolist() != FIRST5:
        return False, f"first samples {arr.ravel()[:5].tolist()} != {FIRST5}"
    got = hashlib.sha256(arr.tobytes()).hexdigest()[:32]
    if got != SHA32:
        return False, f"sha256 {got} != {SHA32}"
    return True, "ok"


def from_reservoirpy():
    """Canonical, maintained source: the reservoirpy dataset loader."""
    import reservoirpy.datasets as ds
    return np.asarray(ds.santafe_laser()).reshape(-1, 1)


def from_url():
    """Fallback: a public raw text mirror (whitespace-separated 8-bit integers)."""
    import urllib.request
    url = ("https://raw.githubusercontent.com/reservoirpy/reservoirpy/"
           "master/reservoirpy/datasets/_santafe_laser.txt")
    with urllib.request.urlopen(url, timeout=30) as r:
        text = r.read().decode()
    return np.array([int(t) for t in text.split()], dtype=np.int64).reshape(-1, 1)


def main():
    if OUT.exists():
        ok, msg = verify(np.load(OUT))
        if ok:
            print(f"{OUT.name}: present and verified ({N_EXPECTED} samples).")
            return 0
        print(f"WARNING: existing {OUT.name} failed verification ({msg}); refetching.")

    for name, fetch in (("reservoirpy", from_reservoirpy), ("url mirror", from_url)):
        try:
            arr = fetch()
        except Exception as e:                              # noqa: BLE001
            print(f"  {name}: unavailable ({e})")
            continue
        ok, msg = verify(arr)
        if ok:
            np.save(OUT, arr.astype(np.int64))
            print(f"Fetched from {name} and verified -> {OUT.name}")
            return 0
        print(f"  {name}: fetched but FAILED verification ({msg}) -- not saved")

    print("ERROR: could not obtain a verified copy. See santafe_provenance.md.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

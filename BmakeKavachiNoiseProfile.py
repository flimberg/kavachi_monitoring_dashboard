#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
import pandas as pd
from obspy import read, UTCDateTime
from obspy.signal.invsim import cosine_taper
from scipy.fft import rfft, rfftfreq

# -----------------------
# Config
# -----------------------
INPUT_DIR = Path("./shake_data")     # directory containing daily .mseed files
DAYPLOT_DIR = Path("./dayplots")     # directory containing existing PNG dayplots
OUTPUT_CSV = "mean_amplitudes.csv"   # CSV file (time, amplitude)
CHUNK_DURATION_SEC = 60              # segment length for mean amplitude calculation
FREQ_RANGE = (1.0, 50.0)             # Hz
CHANNEL = "EHZ"                      # process this channel
AMP_THRESHOLD = 10e6                 # if mean amplitude exceeds this, replace via interpolation

# File name patterns
DAYPLOT_PATTERN = "AM_RF90E_EHZ_{date}_5-40Hz.png"
# -----------------------

def calculate_mean_amplitude(tr, freq_range=(1.0, 50.0)):
    fs = float(tr.stats.sampling_rate)
    data = tr.data.astype(np.float64, copy=False)
    if data.size == 0 or not np.any(np.isfinite(data)):
        return np.nan

    n = data.size
    win = cosine_taper(n, 0.10)
    x = np.nan_to_num(data * win, nan=0.0, posinf=0.0, neginf=0.0)

    X = rfft(x)
    freqs = rfftfreq(n, d=1.0 / fs)

    fmin, fmax = freq_range
    fmin = max(0.0, fmin)
    fmax = min(fmax, fs / 2.0)
    if fmax <= fmin:
        return np.nan

    sel = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(sel):
        return np.nan

    return float(np.mean(np.abs(X[sel])))


def process_miniseed_file_in_chunks(file_path, chunk_duration=CHUNK_DURATION_SEC,
                                    freq_range=FREQ_RANGE):
    rows = []
    try:
        st = read(str(file_path))
        if len(st) == 0:
            return rows

        try:
            st.merge(method=1, fill_value=0)
        except Exception:
            st.merge(method=0)

        st = st.select(channel=CHANNEL)
        if len(st) == 0:
            return rows

        tr = st[0]
        fs = float(tr.stats.sampling_rate)
        npts = int(tr.stats.npts)
        start_t = tr.stats.starttime

        chunk_samples = int(chunk_duration * fs) + 1
        if chunk_samples <= 0:
            return rows

        n_full = npts // chunk_samples
        if n_full == 0:
            return rows

        for k in range(n_full):
            t0 = start_t + k * chunk_duration
            t1 = t0 + chunk_duration
            chunk_tr = tr.slice(starttime=t0, endtime=t1, nearest_sample=False)

            if int(chunk_tr.stats.npts) != chunk_samples:
                try:
                    chunk_tr = chunk_tr.copy()
                    chunk_tr.trim(t0, t1, pad=True, fill_value=0)
                except Exception:
                    continue
                if int(chunk_tr.stats.npts) != chunk_samples:
                    continue

            amp = calculate_mean_amplitude(chunk_tr, freq_range=freq_range)
            ts = f"{UTCDateTime(t0).isoformat()}Z"
            rows.append({"time": ts, "amplitude": amp})

    except Exception as e:
        print(f"[ERROR] {file_path}: {e}")

    return rows


def process_all_miniseed(input_dir: Path, dayplot_dir: Path):
    """Process only MiniSEED files that do NOT already have a PNG dayplot."""
    all_rows = []
    for root, _, files in os.walk(input_dir):
        for name in sorted(files):
            if not name.lower().endswith(".mseed"):
                continue
            fp = Path(root) / name

            # Derive date from filename, e.g., AM.RF90E.00.EHZ.2025-06-02.mseed
            try:
                date_part = name.split(".")[-2]
            except Exception:
                print(f"[WARN] Skipping malformed file: {name}")
                continue

            expected_png = DAYPLOT_PATTERN.format(date=date_part)
            png_path = dayplot_dir / expected_png

            # Skip if dayplot exists
            if png_path.exists():
                print(f"[SKIP] {name} — dayplot exists ({png_path.name})")
                continue

            print(f"Processing {name} (no dayplot found)...")
            all_rows.extend(process_miniseed_file_in_chunks(fp))
    return all_rows


def main():
    # ----------------------------------------------------------
    # Step 1: Load existing CSV (if any)
    # ----------------------------------------------------------
    existing_times = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            df_existing = pd.read_csv(OUTPUT_CSV, usecols=["time"])
            existing_times = set(df_existing["time"].astype(str))
            print(f"[INFO] Loaded existing CSV with {len(existing_times)} timestamps.")
        except Exception as e:
            print(f"[WARN] Could not read existing CSV ({e}); starting fresh.")
            existing_times = set()

    # ----------------------------------------------------------
    # Step 2: Process MiniSEED files without PNGs
    # ----------------------------------------------------------
    rows = process_all_miniseed(INPUT_DIR, DAYPLOT_DIR)
    if not rows:
        print("[INFO] No new data found (all days already have PNGs).")
        return

    new_df = pd.DataFrame(rows, columns=["time", "amplitude"])
    new_df.sort_values("time", inplace=True)

    # Filter out already existing timestamps
    before = len(new_df)
    new_df = new_df[~new_df["time"].astype(str).isin(existing_times)]
    after = len(new_df)

    if after == 0:
        print("[INFO] No new timestamps to append.")
        return

    print(f"[INFO] Found {before} total rows, {after} new unique timestamps to add.")

    # Apply amplitude threshold
    new_df.loc[new_df["amplitude"] > AMP_THRESHOLD, "amplitude"] = np.nan

    # ----------------------------------------------------------
    # Step 3: Append or create CSV
    # ----------------------------------------------------------
    if os.path.exists(OUTPUT_CSV):
        new_df.to_csv(OUTPUT_CSV, mode="a", index=False, header=False)
        print(f"[OK] Appended {len(new_df)} new rows to {OUTPUT_CSV}")
    else:
        new_df.to_csv(OUTPUT_CSV, index=False)
        print(f"[OK] Created {OUTPUT_CSV} with {len(new_df)} rows")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Create 5–40 Hz dayplots for each MiniSEED file in ./shake_data/,
without parallel processing. Each file is handled independently
and plotted in black lines for clarity. If a plot already exists,
it is skipped. After processing, the shake_data folder is emptied.

Author: GPT-5
"""

import os
from pathlib import Path
from obspy import read
import shutil

# ---------------- CONFIG ----------------
INPUT_DIR = Path("./shake_data")     # directory with MiniSEED files
OUTPUT_DIR = Path("./dayplots")      # output directory for PNGs
CHANNEL = "EHZ"                      # channel to plot
FREQMIN, FREQMAX = 5.0, 40.0         # filter band (Hz)
DPI = 150                            # image resolution
# ----------------------------------------


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def process_file(file_path):
    """Read, filter (5–40 Hz), and plot one MiniSEED file."""
    try:
        # Load MiniSEED and merge if necessary
        st = read(str(file_path))
        st.merge(method=1, fill_value=0)
        st = st.select(channel=CHANNEL)
        st = st.detrend('demean')
        if len(st) == 0:
            return f"[WARN] No {CHANNEL} in {file_path.name}"

        # Build output filename
        tr = st[0]
        base_name = f"{tr.stats.network}_{tr.stats.station}_{CHANNEL}_{tr.stats.starttime.date}"
        out_file = OUTPUT_DIR / f"{base_name}_5-40Hz.png"

        # Skip if already exists
        if out_file.exists():
            return f"[SKIP] {out_file.name} already exists."

        # Apply bandpass filter
        st.filter("bandpass", freqmin=FREQMIN, freqmax=FREQMAX,
                  corners=4, zerophase=True)

        # Plot in black only
        title = f"{tr.id} — 5–40 Hz"
        st.plot(
            type="dayplot",
            interval=60,                   # 1 hour per row
            right_vertical_labels=True,
            one_tick_per_line=True,
            show_y_UTC_label=True,
            vertical_scaling_range=6000,
            title=title,
            color='k',                     # black lines only
            linewidth=0.4,
            outfile=str(out_file),
            dpi=DPI,
            show=False
        )

        return f"[OK] {file_path.name} → {out_file.name}"

    except Exception as e:
        return f"[ERROR] {file_path.name}: {e}"


def clear_shake_data():
    """Remove all contents from shake_data folder after processing."""
    if not INPUT_DIR.exists():
        return

    for item in INPUT_DIR.iterdir():
        try:
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        except Exception as e:
            print(f"[WARN] Could not remove {item}: {e}")
    print(f"\n🧹 shake_data folder cleared.")


def main():
    ensure_output_dir()

    # Collect all .mseed files
    mseed_files = sorted([
        Path(root) / name
        for root, _, files in os.walk(INPUT_DIR)
        for name in files if name.lower().endswith(".mseed")
    ])

    if not mseed_files:
        print("No MiniSEED files found.")
        return

    print(f"[INFO] Found {len(mseed_files)} files.\n")

    # Process sequentially
    for file_path in mseed_files:
        print(process_file(file_path))

    print(f"\n✅ Finished. Dayplots saved in: {OUTPUT_DIR.resolve()}")

    # Clear input directory
    clear_shake_data()


if __name__ == "__main__":
    main()

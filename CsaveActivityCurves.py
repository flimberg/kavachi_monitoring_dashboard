#!/usr/bin/env python3
"""
Tremor / Activity Processor (Headless)
--------------------------------------
- Uses 1-hour and 1-day smoothed energy curves
- Computes Hilbert envelope of their difference
- Produces tremor (blue) and activity (red) lines
- Saves hourly-sampled values to processed_activity.csv
- No plotting, no tidal calculations
"""

import pandas as pd
import numpy as np
from scipy.signal import hilbert
import time
import os

start = time.time()

# ---------------- Parameters ----------------
MINUTES_PER_DAY = 24 * 60
WINDOW_DAY = MINUTES_PER_DAY         # 1-day smoothing window
WINDOW_HOUR = 60                     # 1-hour smoothing window
WINDOW_ENVELOPE_SMOOTH = 1200        # smoothing window for envelope
CSV_INPUT = "mean_amplitudes.csv"
CSV_OUTPUT = "processed_activity2.csv"

# ---------------- Helper function ----------------
def normalize(series):
    """Normalize to [0, 1], safely."""
    smin, smax = np.nanmin(series), np.nanmax(series)
    if np.isnan(smin) or np.isnan(smax) or smax <= smin:
        return np.zeros_like(series)
    return (series - smin) / (smax - smin)

# ---------------- Load and preprocess ----------------
if not os.path.exists(CSV_INPUT):
    raise FileNotFoundError(f"❌ Input file not found: {CSV_INPUT}")

print(f"[INFO] Loading {CSV_INPUT} ...")
df = pd.read_csv(CSV_INPUT)
df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
df['amplitude'] /= 2_500_000  # scale amplitudes

# ---------------- Rolling windows ----------------
df['smooth_day'] = df['amplitude'].rolling(window=WINDOW_DAY, center=True).mean()
df['smooth_hour'] = df['amplitude'].rolling(window=WINDOW_HOUR, center=True).mean()

# Interpolate daily smooth to match timestamps
valid_mask = df['smooth_day'].notna()
if valid_mask.sum() > 2:
    df['smooth_day_interp'] = np.interp(
        df['time'].view('int64'),
        df.loc[valid_mask, 'time'].view('int64'),
        df.loc[valid_mask, 'smooth_day']
    )
else:
    df['smooth_day_interp'] = np.nan

# ---------------- Difference and normalization ----------------
df['difference'] = df['smooth_hour'] - df['smooth_day_interp']
df['difference_demean'] = df['difference'] - df['difference'].mean()
max_abs = np.nanmax(np.abs(df['difference_demean']))
if np.isfinite(max_abs) and max_abs > 0:
    df['difference_demean'] /= max_abs
else:
    df['difference_demean'] = 0

# ---------------- Hilbert envelope (tremor indicator) ----------------
analytic_signal = hilbert(df['difference_demean'].fillna(0))
envelope_raw = np.abs(analytic_signal)
df['envelope_smooth'] = (
    pd.Series(envelope_raw)
    .rolling(window=WINDOW_ENVELOPE_SMOOTH, center=True, min_periods=1)
    .mean()
)

# ---------------- Normalize indicators ----------------
df['D'] = normalize(df['envelope_smooth'])   # Tremor (blue)
df['E'] = normalize(df['smooth_day'])        # Activity (red)

# ---------------- Resample to hourly sampling ----------------
df_final = df[['time', 'D', 'E']].copy()
df_final.rename(columns={'D': 'tremor_score_blue', 'E': 'activity_score_red'}, inplace=True)
df_final.set_index('time', inplace=True)
df_hourly = df_final.resample('1H').mean().reset_index()

# ---------------- Save output ----------------
df_hourly.to_csv(CSV_OUTPUT, index=False)
print(f"✅ Hourly-sampled tremor (blue) and activity (red) lines saved to: {CSV_OUTPUT}")
print(f"Rows written: {len(df_hourly)}")
print("Execution time:", round(time.time() - start, 2), "s")

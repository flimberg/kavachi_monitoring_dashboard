#!/usr/bin/env python3
"""
Download Raspberry Shake waveforms day-by-day starting 2025-06-01 (UTC),
saving one MiniSEED file per day and skipping any that already exist
or that already have a corresponding dayplot PNG like:
    AM_RF90E_EHZ_2025-06-02_5-40Hz.png

Improvements:
- Skips download if dayplot already exists.
- Gracefully handles corrupt MiniSEED records (Steim-2).
- Exponential backoff for rate limits and transient errors.
"""

from obspy import UTCDateTime, Stream, read
from obspy.clients.fdsn import Client
from obspy.io.mseed import InternalMSEEDError
from pathlib import Path
from datetime import datetime, timedelta, timezone
from io import BytesIO
import time
import random
import re

# -----------------------------
# User configuration
# -----------------------------
NETWORK = "AM"
STATION = "RF90E"
LOCATION = "00"
CHANNEL = "EHZ"

START_DATE_UTC = "2025-06-01"  # inclusive YYYY-MM-DD (UTC)
END_DATE_UTC = datetime.now(timezone.utc).date().isoformat()  # exclusive upper bound

OUT_DIR = Path("shake_data")
DAYPLOT_DIR = Path("dayplots")  # directory where PNGs are stored
FILENAME_PATTERN = "{network}.{station}.{location}.{channel}.{date}.mseed"
DAYPLOT_PATTERN = "{network}_{station}_{channel}_{date}_5-40Hz.png"

# Request tuning
CHUNK_HOURS = 4
REQUEST_PAUSE_SECONDS = 5
MAX_RETRIES = 6
BACKOFF_INITIAL = 10
BACKOFF_MAX = 600
JITTER_MAX = 1.5
# -----------------------------


def daterange(start_date, end_date):
    """Yield datetime.date objects from start_date to end_date (exclusive)."""
    cur = start_date
    while cur < end_date:
        yield cur
        cur += timedelta(days=1)


def chunk_bounds_for_day(date_obj, hours_per_chunk=4):
    """Return (t0, t1) pairs covering a UTC day."""
    assert 24 % hours_per_chunk == 0
    day_start = UTCDateTime(datetime(date_obj.year, date_obj.month, date_obj.day, tzinfo=timezone.utc))
    bounds = []
    for k in range(0, 24, hours_per_chunk):
        t0 = day_start + 3600 * k
        t1 = day_start + 3600 * (k + hours_per_chunk)
        bounds.append((t0, t1))
    return bounds


def parse_retry_after_from_exception(exc_msg: str):
    """Parse a Retry-After duration from an error string, if present."""
    m = re.search(r"[Rr]etry-After[:\s]+(\d+)", exc_msg)
    if m:
        return int(m.group(1))
    m = re.search(r"retry(?:\s+in|\s+after)?\s+(\d+)\s*seconds?", exc_msg, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def polite_sleep(seconds):
    time.sleep(seconds + random.random() * JITTER_MAX)


def fetch_chunk_normal(client, net, sta, loc, cha, t0, t1) -> Stream:
    return client.get_waveforms(net, sta, loc, cha, t0, t1, attach_response=False)


def fetch_chunk_fallback_ignore_errors(client, net, sta, loc, cha, t0, t1) -> Stream:
    resp = client._download(
        service="waveform",
        network=net, station=sta, location=loc, channel=cha,
        starttime=t0, endtime=t1
    )
    data = resp.read()
    if not data:
        return Stream()
    return read(BytesIO(data), format="MSEED", ignore_data_errors=True)


def fetch_with_retries(client, net, sta, loc, cha, t0, t1):
    """Fetch one chunk with exponential backoff and corruption recovery."""
    attempt = 0
    backoff = BACKOFF_INITIAL

    while True:
        try:
            return fetch_chunk_normal(client, net, sta, loc, cha, t0, t1)

        except InternalMSEEDError as e:
            print(f"  - Corrupt MiniSEED {t0}–{t1}: {e}")
            try:
                st = fetch_chunk_fallback_ignore_errors(client, net, sta, loc, cha, t0, t1)
                if len(st) > 0:
                    print(f"    -> Fallback recovered {len(st)} trace(s).")
                else:
                    print("    -> Fallback yielded no usable data.")
                return st
            except Exception as e2:
                print(f"    -> Fallback failed; skipping chunk {t0}–{t1}: {e2}")
                return Stream()

        except Exception as e:
            attempt += 1
            msg = str(e)
            is_429 = ("429" in msg) or ("Too Many Requests" in msg) or ("rate limit" in msg.lower())

            if attempt > MAX_RETRIES:
                print(f"  - Giving up on chunk {t0}–{t1}: {msg[:180]}")
                return Stream()

            wait_sec = parse_retry_after_from_exception(msg) if is_429 else None
            if wait_sec is None:
                wait_sec = min(backoff, BACKOFF_MAX)
                backoff = min(backoff * (2 if is_429 else 1.5), BACKOFF_MAX)

            print(f"  - Retry {attempt}/{MAX_RETRIES} after {wait_sec}s ({'429' if is_429 else 'error'}).")
            polite_sleep(wait_sec)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DAYPLOT_DIR.mkdir(parents=True, exist_ok=True)
    client = Client("RASPISHAKE", timeout=120)

    start_date = datetime.fromisoformat(START_DATE_UTC).date()
    end_date = datetime.fromisoformat(END_DATE_UTC).date()

    print(f"Downloading {NETWORK}.{STATION}.{LOCATION}.{CHANNEL}")
    print(f"Range: {start_date} → {end_date} (exclusive)\n")

    for day in daterange(start_date, end_date):
        # Build expected file names
        mseed_name = FILENAME_PATTERN.format(
            network=NETWORK, station=STATION, location=LOCATION, channel=CHANNEL, date=day.isoformat()
        )
        dayplot_name = DAYPLOT_PATTERN.format(
            network=NETWORK, station=STATION, channel=CHANNEL, date=day.isoformat()
        )

        mseed_path = OUT_DIR / mseed_name
        dayplot_path = DAYPLOT_DIR / dayplot_name

        # Skip if MiniSEED or dayplot already exist
        if dayplot_path.exists():
            print(f"[SKIP] {day} — dayplot already exists: {dayplot_path.name}")
            continue
        if mseed_path.exists():
            print(f"[SKIP] {day} — waveform already exists: {mseed_path.name}")
            continue

        print(f"[FETCH] {day} — downloading 24h ({24 // CHUNK_HOURS} chunks)...")
        s_all = Stream()
        chunks = chunk_bounds_for_day(day, CHUNK_HOURS)

        for (t0, t1) in chunks:
            polite_sleep(REQUEST_PAUSE_SECONDS)
            st = fetch_with_retries(client, NETWORK, STATION, LOCATION, CHANNEL, t0, t1)
            if len(st) > 0:
                s_all += st
            else:
                print(f"  - No data kept for {t0}–{t1}")

        if len(s_all) == 0:
            print(f"[NONE] {day} — no data retrieved.\n")
            continue

        # Merge and trim
        day_start = UTCDateTime(datetime(day.year, day.month, day.day, tzinfo=timezone.utc))
        day_end = day_start + 24 * 3600
        try:
            s_all.merge(method=1, fill_value=0)
        except Exception:
            s_all.merge(method=0)
        s_all.trim(day_start, day_end, pad=True, fill_value=0)
        s_all = s_all.select(network=NETWORK, station=STATION, location=LOCATION, channel=CHANNEL)

        if len(s_all) == 0:
            print(f"[NONE] {day} — empty after trim.\n")
            continue

        try:
            s_all.write(str(mseed_path), format="MSEED")
            print(f"[OK] {day} — saved {mseed_path.name}\n")
        except Exception as e:
            print(f"[FAIL] {day} — write error: {e}\n")


if __name__ == "__main__":
    main()

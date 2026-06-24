#!/usr/bin/env python3
"""
Download XAUUSD M1 historical bars from Dukascopy (2020–2026).

Output CSV format matches MT5 ExportBars.mq5 and signal_server replay input:
    datetime,open,high,low,close,tick_volume
    2020.01.02 00:00,1517.13,1517.55,1516.41,1516.97,342

Data source: Dukascopy public datafeed (BID candles, 1-minute)
  https://datafeed.dukascopy.com/datafeed/XAUUSD/{year}/{month:02d}/{day:02d}/BID_candles_min_1.bi5
  Note: month is 0-indexed (Jan=00, Dec=11)

File format: LZMA-compressed binary records, 24 bytes each (big-endian):
  [ms_from_midnight:uint32, open:uint32, high:uint32, low:uint32, close:uint32, volume:float32]
  Price divisor for XAUUSD: 1000.0  (raw 1516970 → 1516.970 USD/oz)

Usage:
    python download_dukascopy.py                              # 2020-01-01 to today
    python download_dukascopy.py --from 2020.01.01 --to 2026.06.23 --out bars_2020_2026.csv
    python download_dukascopy.py --validate-price             # test one day, print sample prices
    python download_dukascopy.py --price-div 100              # try if default gives wrong prices

Expected download time: ~45–90 min for 6 years (2190 days × ~0.3s/request).
"""

import argparse
import csv
import struct
import lzma
import sys
import time
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

DUKA_BASE    = "https://datafeed.dukascopy.com/datafeed"
INSTRUMENT   = "XAUUSD"
PRICE_DIV    = 1000.0        # confirmed for XAUUSD (2dp + 1 extra = int/1000)
RECORD_FMT   = ">IIIIIf"    # big-endian: ms, open, high, low, close, vol_float
RECORD_SIZE  = struct.calcsize(RECORD_FMT)   # 24 bytes
UA           = "Mozilla/5.0 (compatible; xauusd-downloader/1.0)"
REQUEST_DELAY = 0.25         # seconds between requests (be polite)
MAX_RETRIES   = 2


# ── Core download / parse ─────────────────────────────────────────────────────

def bi5_url(d: date, price_div: float) -> str:
    # Dukascopy month index is 0-based
    return (f"{DUKA_BASE}/{INSTRUMENT}/{d.year}/{d.month - 1:02d}"
            f"/{d.day:02d}/BID_candles_min_1.bi5")


def download_day(d: date, price_div: float, retries: int = MAX_RETRIES) -> list[dict]:
    url = bi5_url(d, price_div)
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": UA})
            with urlopen(req, timeout=15) as resp:
                compressed = resp.read()
        except HTTPError as e:
            if e.code == 404:
                return []           # weekend / holiday — normal
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            return []
        except (URLError, OSError):
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            return []
        break
    else:
        return []

    try:
        raw = lzma.decompress(compressed)
    except Exception:
        return []

    if len(raw) % RECORD_SIZE != 0:
        return []

    bars = []
    n_recs = len(raw) // RECORD_SIZE
    for i in range(n_recs):
        ms, o, h, l, c, vol = struct.unpack_from(RECORD_FMT, raw, i * RECORD_SIZE)
        if c == 0:
            continue
        # BID_candles_min_1.bi5 encodes time as seconds from midnight (0–86399),
        # not milliseconds and not minute index.
        minute     = ms // 60
        hour_part  = minute // 60
        min_part   = minute % 60
        dt_str     = f"{d.year}.{d.month:02d}.{d.day:02d} {hour_part:02d}:{min_part:02d}"
        tick_vol   = max(1, int(round(vol * 1000)))  # float lot volume → integer tick proxy
        bars.append({
            'datetime':    dt_str,
            'open':        round(o / price_div, 2),
            'high':        round(h / price_div, 2),
            'low':         round(l / price_div, 2),
            'close':       round(c / price_div, 2),
            'tick_volume': tick_vol,
        })
    return bars


# ── Price divisor auto-detect ─────────────────────────────────────────────────
# Gold price range 2020–2026: ~1450–4200 USD/oz.  We pick the first Saturday
# after from_date that has data and test divisors [100, 1000, 10000].

def detect_price_div(from_date: date) -> float:
    test_date = from_date
    for _ in range(60):        # search up to 60 days for a non-empty day
        if test_date.weekday() < 5:   # Mon–Fri
            for div in [1000.0, 100.0, 10000.0]:
                bars = download_day(test_date, div, retries=2)
                if bars:
                    sample = bars[len(bars) // 2]['close']
                    if 400.0 < sample < 15000.0:
                        print(f"  Auto-detected price divisor: {div}  "
                              f"(sample close={sample} USD/oz on {test_date})")
                        return div
        test_date += timedelta(days=1)
    print("  WARNING: could not auto-detect price divisor — using default 1000.0")
    return 1000.0


# ── Main download loop ────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    today = date.today().strftime('%Y.%m.%d')
    ap.add_argument('--from',       dest='from_dt',    default='2020.01.01')
    ap.add_argument('--to',         dest='to_dt',      default=today)
    ap.add_argument('--out',        dest='out',        default='bars_2020_2026.csv')
    ap.add_argument('--price-div',  dest='price_div',  type=float, default=0.0,
                    help='price divisor (0=auto-detect)')
    ap.add_argument('--validate-price', action='store_true',
                    help='download 1 day, print sample prices, and exit')
    ap.add_argument('--delay',      dest='delay',      type=float, default=REQUEST_DELAY,
                    help='seconds between requests (default 0.25)')
    args = ap.parse_args()

    from_date = datetime.strptime(args.from_dt, '%Y.%m.%d').date()
    to_date   = datetime.strptime(args.to_dt,   '%Y.%m.%d').date()
    out_path  = Path(args.out)

    if from_date > to_date:
        print("ERROR: --from must be before --to"); sys.exit(1)

    print(f"XAUUSD M1 downloader — Dukascopy BID candles")
    print(f"  Range  : {from_date} → {to_date}  ({(to_date-from_date).days+1} days)")
    print(f"  Output : {out_path}")

    # ── Price divisor ─────────────────────────────────────────────────────────
    price_div = args.price_div if args.price_div > 0 else 0.0

    if args.validate_price:
        print("\n[validate-price mode — downloading 1 trading day]")
        d = from_date
        while d.weekday() >= 5:   # skip weekends
            d += timedelta(days=1)
        for div in [1000.0, 100.0, 10000.0]:
            bars = download_day(d, div)
            if bars:
                mid = bars[len(bars)//2]
                print(f"  divisor={div:8.0f}  close={mid['close']:.2f}  "
                      f"high={mid['high']:.2f}  low={mid['low']:.2f}  "
                      f"vol={mid['tick_volume']}  ({mid['datetime']})")
        print("\nSelect the divisor that gives gold prices in ~1400–5000 USD/oz range.")
        print("Re-run with: --price-div <correct_value>")
        return

    if price_div == 0.0:
        print("Auto-detecting price divisor...")
        price_div = detect_price_div(from_date)
    else:
        print(f"  Price divisor: {price_div}")

    # ── Download ──────────────────────────────────────────────────────────────
    total_days = (to_date - from_date).days + 1
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Support resume: if file exists, find last written datetime and skip ahead
    resume_date = None
    if out_path.exists() and out_path.stat().st_size > 0:
        with open(out_path, newline='', encoding='utf-8') as f:
            last_row = None
            for last_row in csv.DictReader(f):
                pass
        if last_row:
            last_dt = datetime.strptime(last_row['datetime'][:10], '%Y.%m.%d').date()
            resume_date = last_dt + timedelta(days=1)
            print(f"  Resuming from {resume_date} (last row: {last_row['datetime']})")

    mode = 'a' if resume_date else 'w'
    with open(out_path, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['datetime','open','high','low','close','tick_volume'])
        if mode == 'w':
            writer.writeheader()

        total_bars = 0
        day_count  = 0
        price_ok   = False

        d = resume_date if resume_date else from_date
        while d <= to_date:
            bars = download_day(d, price_div)
            day_count += 1

            if bars:
                # Validate first batch of prices
                if not price_ok:
                    sample = bars[0]['close']
                    if not (400 < sample < 15000):
                        print(f"\n[WARNING] close={sample} is out of expected range [400, 15000].")
                        print(f"  Current price divisor: {price_div}")
                        print(f"  Try: --validate-price  to see all divisor options.")
                    else:
                        price_ok = True

                writer.writerows(bars)
                f.flush()
                total_bars += len(bars)

            pct    = day_count / total_days * 100
            status = f"{len(bars):4d} bars" if bars else "  no data "
            print(f"\r  {d.isoformat()}  {status}  total={total_bars:>9,}  {pct:5.1f}%",
                  end='', flush=True)

            d += timedelta(days=1)
            time.sleep(args.delay)

    print(f"\n\nDownload complete: {total_bars:,} bars → {out_path}")
    print()
    print("Next steps:")
    print(f"  1. Replay (parallel, ~2–5 min on 8 cores):")
    print(f"     rust_signal_server\\target\\release\\signal_server.exe replay \\")
    print(f"         --bars {out_path} \\")
    print(f"         --out signals_full.csv \\")
    print(f"         --actor models\\actor_weights.json \\")
    print(f"         --parallel")
    print()
    print(f"  2. Retrain actor on expanded dataset:")
    print(f"     python train_offline.py")
    print()
    print(f"  3. Restart server:")
    print(f"     .\\start_live.ps1")


if __name__ == '__main__':
    main()

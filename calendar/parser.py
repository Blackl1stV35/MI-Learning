#!/usr/bin/env python3
"""
Investing.com economic calendar parser.
Fetches ec_events_sitemap.xml, filters gold/USD relevant events,
classifies event_risk deterministically by keyword.
Writes session_context.json consumed by MQL5 indicator.

Schedule: run twice daily (before London 07:45 UTC, before NY 12:45 UTC).

Usage:
    python parser.py --out data/session_context.json
    python parser.py --schedule  # runs on schedule indefinitely
"""
from __future__ import annotations
import argparse, json, time, re
from datetime import datetime, timezone, timedelta
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET
from loguru import logger

SITEMAP_URL = "https://www.investing.com/ec_events_sitemap.xml"

# High-impact USD/Gold events → event_risk = 1.0 (hard gate in meta-policy)
HIGH_IMPACT = {
    "federal funds rate", "fomc", "non-farm payrolls", "nfp",
    "cpi", "consumer price index", "ppi", "producer price",
    "gdp", "gross domestic product", "retail sales",
    "powell", "fed chair", "jackson hole",
    "unemployment rate", "initial jobless",
    "ism manufacturing", "ism services",
    "treasury", "debt ceiling", "government shutdown",
}

# Medium-impact → event_risk = 0.5
MEDIUM_IMPACT = {
    "pce", "personal consumption", "durable goods",
    "housing starts", "building permits", "existing home",
    "consumer confidence", "michigan sentiment",
    "trade balance", "current account",
    "factory orders", "industrial production",
    "crude oil inventories",   # DXY correlated
    "ecb", "boe", "boj",       # G4 central banks
    "gold", "xauusd", "comex",
}

# Session boundaries (UTC)
SESSIONS = {
    "asian":  (0,  8),
    "london": (8,  13),
    "ny":     (13, 22),
}


def fetch_sitemap(url: str, timeout: int = 15) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8")


def parse_event_urls(xml_text: str) -> list[str]:
    root = ET.fromstring(xml_text)
    ns   = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    return [loc.text.strip()
            for loc in root.findall(".//sm:loc", ns)
            if loc.text]


def classify_event(url: str, title: str = "") -> tuple[float, str]:
    text = (url + " " + title).lower()
    for kw in HIGH_IMPACT:
        if kw in text:
            return 1.0, kw
    for kw in MEDIUM_IMPACT:
        if kw in text:
            return 0.5, kw
    return 0.0, ""


def current_session(now_utc: datetime) -> str:
    h = now_utc.hour
    if SESSIONS["asian"][0] <= h < SESSIONS["asian"][1]:
        return "asian"
    elif SESSIONS["london"][0] <= h < SESSIONS["london"][1]:
        return "london"
    elif SESSIONS["ny"][0] <= h < SESSIONS["ny"][1]:
        return "ny"
    return "off"


def upcoming_events(urls: list[str], window_hours: int = 8) -> list[dict]:
    """Filter events scheduled within the next window_hours.
    Investing.com URLs encode dates: /economic-calendar/YYYY-MM-DD-event-name
    """
    now    = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=window_hours)
    events = []
    date_pattern = re.compile(r"/(\d{4}-\d{2}-\d{2})-")
    for url in urls:
        m = date_pattern.search(url)
        if not m: continue
        try:
            event_date = datetime.strptime(m.group(1), "%Y-%m-%d").replace(
                tzinfo=timezone.utc)
        except ValueError:
            continue
        if now.date() <= event_date.date() <= cutoff.date():
            risk, kw = classify_event(url)
            if risk > 0:
                events.append({
                    "url":        url,
                    "date":       m.group(1),
                    "event_risk": risk,
                    "keyword":    kw,
                })
    return sorted(events, key=lambda e: e["event_risk"], reverse=True)


def build_session_context(events: list[dict], now: datetime) -> dict:
    session = current_session(now)
    max_risk = max((e["event_risk"] for e in events), default=0.0)
    top_event = events[0] if events else None
    return {
        "timestamp":    now.isoformat(),
        "session":      session,
        "event_risk":   max_risk,
        "top_keyword":  top_event["keyword"] if top_event else "",
        "n_upcoming":   len(events),
        "events":       events[:5],   # top 5 for logging
    }


def run_once(out_path: Path) -> dict:
    logger.info("Fetching calendar sitemap...")
    try:
        xml_text = fetch_sitemap(SITEMAP_URL)
        urls     = parse_event_urls(xml_text)
        logger.info(f"  {len(urls):,} event URLs parsed")
    except Exception as e:
        logger.warning(f"Sitemap fetch failed: {e} — using zero event_risk")
        urls = []

    now    = datetime.now(timezone.utc)
    events = upcoming_events(urls, window_hours=8)
    ctx    = build_session_context(events, now)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(ctx, f, indent=2)

    logger.info(f"session_context.json written: "
                f"session={ctx['session']} "
                f"event_risk={ctx['event_risk']} "
                f"top={ctx['top_keyword'] or 'none'}")
    return ctx


def schedule_loop(out_path: Path) -> None:
    """Run before London open (07:45 UTC) and NY open (12:45 UTC)."""
    run_hours = {7, 12}   # UTC hours to run
    last_run  = -1
    logger.info("Calendar parser running on schedule...")
    while True:
        now = datetime.now(timezone.utc)
        if now.hour in run_hours and now.hour != last_run:
            run_once(out_path)
            last_run = now.hour
        time.sleep(60)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out",      default="data/session_context.json")
    p.add_argument("--schedule", action="store_true")
    args = p.parse_args()
    out = Path(args.out)
    if args.schedule:
        schedule_loop(out)
    else:
        run_once(out)


if __name__ == "__main__":
    main()

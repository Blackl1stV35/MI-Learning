#!/usr/bin/env python3
"""
XAUUSD backtest signal visualizer  --  http://localhost:8080

Serves bars + pre-computed signals as an interactive Plotly chart.
Simulates trade outcomes (TP/SL/timeout) for P&L overlay.

Usage:
    python backtest_viz.py [bars.csv] [signals.csv]
    Defaults to MT5 Common\\Files\\ for both files.
"""

import csv, json, os, sys, webbrowser, threading, math
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

MT5_COMMON   = Path(os.environ.get("APPDATA","")) / "MetaQuotes/Terminal/Common/Files"
DEFAULT_BARS = MT5_COMMON / "XAUUSD_M1_bars.csv"
DEFAULT_SIGS = MT5_COMMON / "signals.csv"
PORT = 8080
MAX_HOLD_BARS = 80   # matches indicator max_hold

# ── CSV loaders ───────────────────────────────────────────────────────────────

def load_bars(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append({
                't': r['datetime'],
                'o': float(r['open']),
                'h': float(r['high']),
                'l': float(r['low']),
                'c': float(r['close']),
                'v': int(r['tick_volume']),
            })
    return rows

def load_signals(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append({
                't':   r['datetime'],
                'dir': float(r['direction_bias']),
                'fd':  float(r['final_dir']),
                'str': float(r['signal_strength']),
                'h':   float(r['hurst']),
                'tda': float(r['tda_wasserstein']),
                'reg': float(r['regime']),
                'ad':  float(r['actor_dir']),
                'ac':  float(r['actor_confidence']),
                'sl':  float(r['sl_pips']),
                'tp':  float(r['tp_pips']),
                'lot': float(r['lot_suggestion']),
                'ex':  r['should_exit'] == '1',
            })
    return rows

# ── Join close price onto signals ─────────────────────────────────────────────

def join_close(bars, sigs):
    idx = {b['t']: (i, b['c']) for i, b in enumerate(bars)}
    for s in sigs:
        entry = idx.get(s['t'])
        s['c']  = entry[1] if entry else None
        s['bi'] = entry[0] if entry else None  # bar index for trade scan

# ── Trade simulation ──────────────────────────────────────────────────────────
# sl and tp are in price units (USD/oz).  direction: +1=buy, -1=sell.

def simulate_trades(bars, sigs):
    trades = []
    for s in sigs:
        if s['str'] == 0 or s['bi'] is None or s['c'] is None:
            continue
        di  = s['dir']
        if di == 0:
            continue
        sl  = s['sl']   # price distance for SL
        tp  = s['tp']   # price distance for TP
        if sl <= 0 or tp <= 0:
            continue
        entry   = s['c']
        sl_px   = entry - di * sl
        tp_px   = entry + di * tp
        bi_start = s['bi']
        outcome = 'timeout'
        exit_t  = s['t']
        exit_px = entry
        bars_held = 0
        for step in range(1, MAX_HOLD_BARS + 1):
            bi = bi_start + step
            if bi >= len(bars):
                break
            bar = bars[bi]
            bars_held = step
            # Check SL first (pessimistic)
            if di > 0:  # BUY: SL below, TP above
                if bar['l'] <= sl_px:
                    outcome = 'sl'; exit_t = bar['t']; exit_px = sl_px; break
                if bar['h'] >= tp_px:
                    outcome = 'tp'; exit_t = bar['t']; exit_px = tp_px; break
            else:       # SELL: SL above, TP below
                if bar['h'] >= sl_px:
                    outcome = 'sl'; exit_t = bar['t']; exit_px = sl_px; break
                if bar['l'] <= tp_px:
                    outcome = 'tp'; exit_t = bar['t']; exit_px = tp_px; break
        else:
            exit_px = bars[min(bi_start + MAX_HOLD_BARS, len(bars)-1)]['c']
            exit_t  = bars[min(bi_start + MAX_HOLD_BARS, len(bars)-1)]['t']

        pnl_px = di * (exit_px - entry)
        trades.append({
            'entry_t': s['t'],  'exit_t': exit_t,
            'entry_px': entry,  'exit_px': exit_px,
            'dir': di,  'outcome': outcome,
            'pnl_px': round(pnl_px, 2),
            'bars_held': bars_held,
            'lot': s['lot'],
        })
    return trades

# ── Date filter ───────────────────────────────────────────────────────────────

def filter_range(rows, from_dt, to_dt, key='t'):
    return [r for r in rows if from_dt <= r[key] <= to_dt]

# ── HTTP server ───────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        from_dt = qs.get('from', ['2026.06.01 00:00'])[0]
        to_dt   = qs.get('to',   ['2026.06.23 23:59'])[0]

        srv = self.server
        if parsed.path == '/':
            html = (Path(__file__).with_name('backtest_viz.html')
                    .read_text(encoding='utf-8'))
            self._send(200, 'text/html; charset=utf-8', html.encode())
        elif parsed.path == '/api/bars':
            self._json(filter_range(srv.bars, from_dt, to_dt))
        elif parsed.path == '/api/signals':
            self._json(filter_range(srv.sigs, from_dt, to_dt))
        elif parsed.path == '/api/trades':
            self._json(filter_range(srv.trades, from_dt, to_dt, key='entry_t'))
        elif parsed.path == '/api/meta':
            self._json({
                'bar_count': len(srv.bars),
                'sig_count': len(srv.sigs),
                'trade_count': len(srv.trades),
                'first': srv.bars[0]['t']  if srv.bars else '',
                'last':  srv.bars[-1]['t'] if srv.bars else '',
                'tp_count':  sum(1 for t in srv.trades if t['outcome']=='tp'),
                'sl_count':  sum(1 for t in srv.trades if t['outcome']=='sl'),
                'to_count':  sum(1 for t in srv.trades if t['outcome']=='timeout'),
                'total_pnl': round(sum(t['pnl_px'] for t in srv.trades), 2),
            })
        else:
            self.send_error(404)

    def _json(self, data):
        body = json.dumps(data, separators=(',', ':')).encode()
        self._send(200, 'application/json', body)

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    bars_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_BARS
    sigs_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SIGS

    print(f"Loading bars   : {bars_path}")
    bars = load_bars(bars_path)
    print(f"  {len(bars):,} bars")

    print(f"Loading signals: {sigs_path}")
    sigs = load_signals(sigs_path)
    print(f"  {len(sigs):,} signals")

    print("Joining price + simulating trades...", end=' ', flush=True)
    join_close(bars, sigs)
    trades = simulate_trades(bars, sigs)
    tp = sum(1 for t in trades if t['outcome']=='tp')
    sl = sum(1 for t in trades if t['outcome']=='sl')
    to = sum(1 for t in trades if t['outcome']=='timeout')
    total_pnl = sum(t['pnl_px'] for t in trades)
    print(f"{len(trades):,} trades  TP={tp} SL={sl} Timeout={to}  "
          f"Net P&L={total_pnl:+.1f} USD/oz")

    server = HTTPServer(('localhost', PORT), Handler)
    server.bars   = bars
    server.sigs   = sigs
    server.trades = trades

    url = f"http://localhost:{PORT}"
    print(f"\nVisualizer -> {url}  (Ctrl+C to stop)\n")
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == '__main__':
    main()

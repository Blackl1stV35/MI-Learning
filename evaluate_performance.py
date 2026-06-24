#!/usr/bin/env python3
"""
XAUUSD backtest performance evaluation.

Loads signals.csv + XAUUSD_M1_bars.csv, simulates SL/TP outcomes, and
prints a comprehensive metrics report. Saves results to logs/performance_metrics.json.

Usage:
    python evaluate_performance.py [bars.csv] [signals.csv]
    python evaluate_performance.py --final-dir     # use final_dir instead of direction_bias

Defaults to MT5 Common\\Files\\ for both files.
"""

import csv
import json
import sys
import os
import math
import statistics
import argparse
from pathlib import Path
from datetime import datetime

MT5_COMMON   = Path(os.environ.get("APPDATA", "")) / "MetaQuotes/Terminal/Common/Files"
DEFAULT_BARS = MT5_COMMON / "XAUUSD_M1_bars.csv"
DEFAULT_SIGS = MT5_COMMON / "signals.csv"
LOG_DIR      = Path("logs")
MAX_HOLD     = 80   # bars before timeout exit


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_bars(path):
    bars = []
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            bars.append({
                't': r['datetime'],
                'o': float(r['open']),
                'h': float(r['high']),
                'l': float(r['low']),
                'c': float(r['close']),
                'v': float(r.get('tick_volume', 1)),
            })
    return bars


def load_signals(path):
    sigs = []
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            sigs.append({
                't':   r['datetime'],
                'dir': float(r['direction_bias']),
                'fd':  float(r['final_dir']),
                'str': float(r['signal_strength']),
                'h':   float(r['hurst']),
                'tda': float(r['tda_wasserstein']),
                'reg': float(r['regime']),
                'ad':  float(r['actor_dir']),
                'sl':  float(r['sl_pips']),
                'tp':  float(r['tp_pips']),
                'lot': float(r['lot_suggestion']),
            })
    return sigs


def join_close(bars, sigs):
    """Attach bar index and close price to each signal by datetime match."""
    idx = {b['t']: (i, b['c']) for i, b in enumerate(bars)}
    matched = 0
    for s in sigs:
        entry = idx.get(s['t'])
        s['c']  = entry[1] if entry else None
        s['bi'] = entry[0] if entry else None
        if entry:
            matched += 1
    return matched


# ── Trade simulation ──────────────────────────────────────────────────────────

def simulate_trades(bars, sigs, use_final_dir=False):
    trades = []
    for s in sigs:
        di = s['fd'] if use_final_dir else s['dir']
        if di == 0 or s['str'] == 0:
            continue
        if s['bi'] is None or s['c'] is None:
            continue
        sl, tp = s['sl'], s['tp']
        if sl <= 0 or tp <= 0:
            continue

        entry    = s['c']
        sl_px    = entry - di * sl
        tp_px    = entry + di * tp
        bi_start = s['bi']
        outcome  = 'timeout'
        exit_px  = entry
        exit_t   = s['t']
        bars_held = 0

        for step in range(1, MAX_HOLD + 1):
            bi = bi_start + step
            if bi >= len(bars):
                break
            bar = bars[bi]
            bars_held = step
            if di > 0:
                if bar['l'] <= sl_px:
                    outcome = 'sl'; exit_px = sl_px; exit_t = bar['t']; break
                if bar['h'] >= tp_px:
                    outcome = 'tp'; exit_px = tp_px; exit_t = bar['t']; break
            else:
                if bar['h'] >= sl_px:
                    outcome = 'sl'; exit_px = sl_px; exit_t = bar['t']; break
                if bar['l'] <= tp_px:
                    outcome = 'tp'; exit_px = tp_px; exit_t = bar['t']; break
        else:
            last_bi = min(bi_start + MAX_HOLD, len(bars) - 1)
            exit_px = bars[last_bi]['c']
            exit_t  = bars[last_bi]['t']

        pnl_px = di * (exit_px - entry)
        trades.append({
            'entry_t':  s['t'],
            'exit_t':   exit_t,
            'entry_px': entry,
            'exit_px':  exit_px,
            'dir':      di,
            'outcome':  outcome,
            'pnl_px':   round(pnl_px, 3),
            'bars_held': bars_held,
            'lot':      s['lot'],
            'str':      s['str'],
            'regime':   s['reg'],
            'hurst':    s['h'],
        })
    return trades


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(trades, n_sigs):
    if not trades:
        return {'error': 'no trades'}

    pnls   = [t['pnl_px'] for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    total  = len(trades)

    tp_c  = sum(1 for t in trades if t['outcome'] == 'tp')
    sl_c  = sum(1 for t in trades if t['outcome'] == 'sl')
    to_c  = sum(1 for t in trades if t['outcome'] == 'timeout')

    win_rate      = len(wins) / total
    gross_profit  = sum(wins)   if wins   else 0.0
    gross_loss    = abs(sum(losses)) if losses else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-8 else float('inf')
    total_pnl     = sum(pnls)
    avg_pnl       = total_pnl / total
    avg_win       = statistics.mean(wins)   if wins   else 0.0
    avg_loss      = statistics.mean(losses) if losses else 0.0

    # Sharpe — treat each trade as one observation, annualize by M1 bars/year
    if len(pnls) > 1:
        std_pnl = statistics.stdev(pnls)
        # trades are distributed over ~99,762 M1 bars (~69 days of signal data)
        n_bars  = n_sigs
        trades_per_bar = total / n_bars if n_bars else 1
        bars_per_year  = 252 * 24 * 60
        trades_per_year = trades_per_bar * bars_per_year
        sharpe = (avg_pnl / std_pnl) * math.sqrt(trades_per_year) if std_pnl > 1e-8 else 0.0
    else:
        sharpe = 0.0

    # Max drawdown in price terms
    cum, peak, max_dd = 0.0, 0.0, 0.0
    dd_trades, dd_start = 0, ''
    for t in trades:
        cum += t['pnl_px']
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
            dd_start = t['entry_t']

    avg_bars   = statistics.mean(t['bars_held'] for t in trades)
    expectancy = avg_pnl

    # Direction breakdown
    buy_trades  = [t for t in trades if t['dir'] > 0]
    sell_trades = [t for t in trades if t['dir'] < 0]
    buy_wr  = len([t for t in buy_trades  if t['pnl_px'] > 0]) / len(buy_trades)  if buy_trades  else 0.0
    sell_wr = len([t for t in sell_trades if t['pnl_px'] > 0]) / len(sell_trades) if sell_trades else 0.0

    # Regime breakdown
    bull_trades = [t for t in trades if t['regime'] > 0.5]
    bear_trades = [t for t in trades if t['regime'] <= 0.5]
    bull_pnl = sum(t['pnl_px'] for t in bull_trades)
    bear_pnl = sum(t['pnl_px'] for t in bear_trades)

    return {
        'total_trades':    total,
        'tp_count':        tp_c,
        'sl_count':        sl_c,
        'timeout_count':   to_c,
        'win_rate':        round(win_rate, 4),
        'profit_factor':   round(profit_factor, 3) if profit_factor != float('inf') else 999.0,
        'total_pnl_px':    round(total_pnl, 2),
        'avg_pnl_px':      round(avg_pnl, 3),
        'avg_win_px':      round(avg_win, 3),
        'avg_loss_px':     round(avg_loss, 3),
        'gross_profit_px': round(gross_profit, 2),
        'gross_loss_px':   round(gross_loss, 2),
        'sharpe_ratio':    round(sharpe, 3),
        'max_drawdown_px': round(max_dd, 2),
        'max_dd_from':     dd_start,
        'avg_bars_held':   round(avg_bars, 1),
        'expectancy_px':   round(expectancy, 3),
        'buy_trades':      len(buy_trades),
        'sell_trades':     len(sell_trades),
        'buy_win_rate':    round(buy_wr, 4),
        'sell_win_rate':   round(sell_wr, 4),
        'bull_regime_pnl': round(bull_pnl, 2),
        'bear_regime_pnl': round(bear_pnl, 2),
        'signal_count':    n_sigs,
        'trade_rate':      round(total / n_sigs, 4) if n_sigs else 0,
    }


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(m, sig_label, use_fd):
    dir_label = "final_dir" if use_fd else "direction_bias"
    print()
    print("=" * 60)
    print(f"  XAUUSD M1 Backtest  [{dir_label}]")
    print("=" * 60)
    print(f"  Signals total       : {m['signal_count']:>10,}")
    print(f"  Trades entered      : {m['total_trades']:>10,}  ({m['trade_rate']*100:.1f}%)")
    print(f"  TP / SL / Timeout   : {m['tp_count']} / {m['sl_count']} / {m['timeout_count']}")
    print(f"  BUY  trades / WR    : {m['buy_trades']} / {m['buy_win_rate']*100:.1f}%")
    print(f"  SELL trades / WR    : {m['sell_trades']} / {m['sell_win_rate']*100:.1f}%")
    print("-" * 60)
    print(f"  Win rate            : {m['win_rate']*100:>9.1f}%")
    print(f"  Profit factor       : {m['profit_factor']:>10.3f}")
    print(f"  Sharpe (annualized) : {m['sharpe_ratio']:>10.3f}")
    print(f"  Total P&L           : {m['total_pnl_px']:>10.2f}  USD/oz")
    print(f"  Avg P&L / trade     : {m['avg_pnl_px']:>10.3f}  USD/oz")
    print(f"  Avg win             : {m['avg_win_px']:>10.3f}  USD/oz")
    print(f"  Avg loss            : {m['avg_loss_px']:>10.3f}  USD/oz")
    print(f"  Gross profit        : {m['gross_profit_px']:>10.2f}  USD/oz")
    print(f"  Gross loss          : {m['gross_loss_px']:>10.2f}  USD/oz")
    print(f"  Max drawdown        : {m['max_drawdown_px']:>10.2f}  USD/oz  (from {m['max_dd_from']})")
    print(f"  Avg bars held       : {m['avg_bars_held']:>10.1f}  bars")
    print("-" * 60)
    print(f"  Bull regime P&L     : {m['bull_regime_pnl']:>10.2f}  USD/oz")
    print(f"  Bear regime P&L     : {m['bear_regime_pnl']:>10.2f}  USD/oz")
    print("=" * 60)


# ── Entry ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bars',    nargs='?', default=str(DEFAULT_BARS))
    ap.add_argument('signals', nargs='?', default=str(DEFAULT_SIGS))
    ap.add_argument('--final-dir', action='store_true',
                    help='use final_dir column instead of direction_bias')
    args = ap.parse_args()

    bars_path = Path(args.bars)
    sigs_path = Path(args.signals)
    use_fd    = args.final_dir

    if not bars_path.exists():
        print(f"ERROR: bars file not found: {bars_path}")
        print(f"  Run ExportBars.mq5 in MT5 to create: {DEFAULT_BARS}")
        sys.exit(1)
    if not sigs_path.exists():
        print(f"ERROR: signals file not found: {sigs_path}")
        print(f"  Run: signal_server.exe replay --bars ... --out {DEFAULT_SIGS}")
        sys.exit(1)

    print(f"Loading bars   : {bars_path}")
    bars = load_bars(bars_path)
    print(f"  {len(bars):,} bars  [{bars[0]['t']} → {bars[-1]['t']}]")

    print(f"Loading signals: {sigs_path}")
    sigs = load_signals(sigs_path)
    print(f"  {len(sigs):,} signals  [{sigs[0]['t']} → {sigs[-1]['t']}]")

    print("Joining by datetime...", end=' ', flush=True)
    matched = join_close(bars, sigs)
    print(f"{matched:,} matched ({100*matched/len(sigs):.1f}%)")

    dir_col = 'final_dir' if use_fd else 'direction_bias'
    print(f"Simulating trades (SL/TP scan, {dir_col})...", end=' ', flush=True)
    trades = simulate_trades(bars, sigs, use_final_dir=use_fd)
    print(f"{len(trades):,} trades")

    metrics = compute_metrics(trades, len(sigs))
    print_report(metrics, sigs_path.name, use_fd)

    LOG_DIR.mkdir(exist_ok=True)
    out_path = LOG_DIR / "performance_metrics.json"
    metrics['generated_utc'] = datetime.utcnow().isoformat()
    metrics['bars_file']     = str(bars_path)
    metrics['signals_file']  = str(sigs_path)
    metrics['direction_col'] = dir_col
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved: {out_path}")


if __name__ == '__main__':
    main()

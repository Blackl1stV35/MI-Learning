#!/usr/bin/env python3
# XAUUSD Meta-Policy Live Dashboard  —  http://localhost:8080
# Data: logs/signal_server_live.log + MT5 Common\Files\position_log.csv + override_log.csv
# Usage: python live_dashboard.py  →  opens http://localhost:8080 (auto-refresh 30s)

import re, csv, os, io, subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import json

# ── Paths ─────────────────────────────────────────────────────────────────────
LOG_FILE = Path(r"d:\xauusd_system\logs\signal_server_live.log")
COMMON   = Path(os.environ.get("APPDATA", "")) / "MetaQuotes/Terminal/Common/Files"
POS_LOG  = COMMON / "position_log.csv"
OVR_LOG  = COMMON / "override_log.csv"
TZ_LOCAL = timezone(timedelta(hours=7))   # UTC+7 (Bangkok)

# ── Signal log parser ─────────────────────────────────────────────────────────
_SIG_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z\s+INFO.*?"
    r"HTTP signal: final_dir=([+-]?\d+\.\d+) actor=([+-]?\d+\.\d+) "
    r"strength=([+-]?\d+\.\d+) regime=([+-]?\d+\.\d+)"
)

def parse_signals(n=120):
    rows = []
    try:
        text = LOG_FILE.read_text(encoding="utf-8", errors="replace")
        for m in _SIG_RE.finditer(text):
            utc = datetime.fromisoformat(m.group(1) + "+00:00")
            local = utc.astimezone(TZ_LOCAL)
            fd  = float(m.group(2))
            act = float(m.group(3))
            rows.append({
                "ts":        local.strftime("%H:%M"),
                "ts_full":   local.strftime("%Y-%m-%d %H:%M:%S"),
                "final_dir": fd,
                "actor":     act,
                "strength":  float(m.group(4)),
                "regime":    float(m.group(5)),
                "label":     "BUY" if fd > 0.5 else ("SELL" if fd < -0.5 else "HOLD"),
            })
    except Exception:
        pass
    return rows[-n:]

def server_pid():
    try:
        r = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq signal_server.exe", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=3
        )
        for line in r.stdout.splitlines():
            if "signal_server.exe" in line:
                parts = line.strip('"').split('","')
                return int(parts[1]) if len(parts) > 1 else "?"
    except Exception:
        pass
    return None

def _safe_read_csv(path):
    """Read a CSV that may be open/locked by MT5 (Windows sharing issue)."""
    try:
        raw = path.read_bytes()
        text = raw.decode("ansi", errors="replace")
        return list(csv.DictReader(io.StringIO(text)))
    except Exception:
        return []

def parse_positions():
    return _safe_read_csv(POS_LOG)

def parse_overrides():
    rows = []
    try:
        raw  = OVR_LOG.read_bytes().decode("ansi", errors="replace")
        rdr  = csv.reader(io.StringIO(raw))
        hdr  = next(rdr, None)
        for row in rdr:
            if len(row) >= 7:
                rows.append(row)
    except Exception:
        pass
    return rows

def compute_stats(signals):
    if not signals:
        return {"total": 0, "buy": 0, "sell": 0, "hold": 0,
                "buy_pct": 0, "sell_pct": 0, "hold_pct": 0,
                "actor_mean": 0, "saturated_pct": 0}
    n = len(signals)
    buy  = sum(1 for s in signals if s["final_dir"] >  0.5)
    sell = sum(1 for s in signals if s["final_dir"] < -0.5)
    hold = n - buy - sell
    actors = [s["actor"] for s in signals]
    saturated = sum(1 for a in actors if abs(a) > 0.99)
    return {
        "total":          n,
        "buy":            buy,  "buy_pct":  round(buy  / n * 100),
        "sell":           sell, "sell_pct": round(sell / n * 100),
        "hold":           hold, "hold_pct": round(hold / n * 100),
        "actor_mean":     round(sum(actors) / n, 4),
        "actor_min":      round(min(actors), 4),
        "actor_max":      round(max(actors), 4),
        "saturated_pct":  round(saturated / n * 100),
    }

def current_position(pos_rows):
    """Return the most recent position state from position_log."""
    if not pos_rows:
        return None
    last = pos_rows[-1]
    # find last entry trade (BUY/SELL action)
    entry_price = float(last.get("entry_price", 0) or 0)
    pos_dir     = float(last.get("pos_dir", 0) or 0)
    hold_frac   = float(last.get("hold_frac", 0) or 0)
    unreal      = float(last.get("unrealized_norm", 0) or 0)
    sl          = float(last.get("sl_price", 0) or 0)
    tp          = float(last.get("tp_price", 0) or 0)
    if pos_dir == 0:
        return None
    return {
        "dir":        "SELL" if pos_dir < 0 else "BUY",
        "entry":      entry_price,
        "sl":         sl,
        "tp":         tp,
        "hold_frac":  hold_frac,
        "hold_bars":  round(hold_frac * 80),
        "unreal":     unreal,
        "last_dt":    last.get("datetime", ""),
    }

# ── API data bundle ───────────────────────────────────────────────────────────
def build_data():
    signals  = parse_signals()
    pos_rows = parse_positions()
    stats    = compute_stats(signals)
    position = current_position(pos_rows)
    pid      = server_pid()
    current  = signals[-1] if signals else None
    overrides = parse_overrides()

    # minutes since last signal
    stale = None
    if current:
        try:
            last_ts = datetime.strptime(
                signals[-1]["ts_full"], "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=TZ_LOCAL)
            stale = int((datetime.now(TZ_LOCAL) - last_ts).total_seconds() / 60)
        except Exception:
            pass

    return {
        "server_pid":  pid,
        "stale_min":   stale,
        "current":     current,
        "stats":       stats,
        "position":    position,
        "signals":     signals,
        "pos_rows":    pos_rows[-50:],
        "overrides":   overrides[-10:],
        "now":         datetime.now(TZ_LOCAL).strftime("%Y-%m-%d %H:%M:%S"),
    }

# ── HTML template ─────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>XAUUSD Meta-Policy Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0d1117; --card: #161b22; --border: #30363d;
    --green: #3fb950; --red: #f85149; --yellow: #d29922;
    --blue: #58a6ff; --silver: #8b949e; --white: #e6edf3;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--white); font-family: 'Segoe UI', monospace; font-size: 13px; }
  a { color: var(--blue); text-decoration: none; }
  .container { max-width: 1200px; margin: 0 auto; padding: 12px 16px; }

  /* Header */
  .header { display: flex; align-items: center; justify-content: space-between;
             border-bottom: 1px solid var(--border); padding-bottom: 10px; margin-bottom: 14px; }
  .header h1 { font-size: 16px; font-weight: 600; letter-spacing: .5px; }
  .status-bar { display: flex; gap: 16px; align-items: center; font-size: 12px; color: var(--silver); }
  .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 4px; }
  .dot-green { background: var(--green); box-shadow: 0 0 6px var(--green); }
  .dot-red   { background: var(--red); }
  .dot-gray  { background: var(--silver); }
  .refresh-btn { background: var(--card); border: 1px solid var(--border); color: var(--silver);
                  padding: 4px 10px; border-radius: 6px; cursor: pointer; font-size: 12px; }
  .refresh-btn:hover { color: var(--white); border-color: var(--blue); }

  /* Grid */
  .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 14px; }
  .grid-2 { display: grid; grid-template-columns: 2fr 1fr; gap: 12px; margin-bottom: 14px; }
  @media(max-width: 800px) { .grid-3, .grid-2 { grid-template-columns: 1fr; } }

  /* Card */
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 14px; }
  .card-title { font-size: 10px; text-transform: uppercase; letter-spacing: 1px;
                 color: var(--silver); margin-bottom: 10px; }

  /* Signal card */
  .sig-dir { font-size: 36px; font-weight: 700; letter-spacing: 1px; margin-bottom: 8px; }
  .sig-dir.buy  { color: var(--green); }
  .sig-dir.sell { color: var(--red); }
  .sig-dir.hold { color: var(--silver); }
  .sig-rows { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 16px; }
  .sig-row { display: flex; justify-content: space-between; padding: 3px 0;
              border-bottom: 1px solid var(--border); }
  .sig-label { color: var(--silver); }
  .sig-val   { font-weight: 600; font-family: monospace; }

  /* Position card */
  .pos-dir { font-size: 22px; font-weight: 700; margin-bottom: 10px; }
  .pos-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 5px 12px; }
  .pos-item { display: flex; flex-direction: column; }
  .pos-key  { font-size: 10px; color: var(--silver); text-transform: uppercase; margin-bottom: 1px; }
  .pos-val  { font-family: monospace; font-size: 13px; font-weight: 600; }
  .hold-bar-wrap { margin-top: 10px; }
  .hold-bar-bg  { background: var(--border); border-radius: 4px; height: 6px; margin-top: 3px; }
  .hold-bar-fill { background: var(--yellow); border-radius: 4px; height: 6px; transition: width .3s; }

  /* Stats card */
  .stat-row { display: flex; justify-content: space-between; align-items: center;
               padding: 5px 0; border-bottom: 1px solid var(--border); }
  .stat-row:last-child { border-bottom: none; }
  .stat-bar { height: 4px; border-radius: 2px; margin-top: 3px; }
  .bar-buy  { background: var(--green); }
  .bar-sell { background: var(--red); }
  .bar-hold { background: var(--silver); }

  /* Chart */
  .chart-wrap { position: relative; height: 180px; }

  /* Table */
  .log-table { width: 100%; border-collapse: collapse; font-size: 12px; }
  .log-table th { color: var(--silver); font-weight: 500; text-align: left;
                   padding: 4px 8px; border-bottom: 1px solid var(--border); font-size: 10px;
                   text-transform: uppercase; letter-spacing: .5px; }
  .log-table td { padding: 4px 8px; border-bottom: 1px solid #21262d; font-family: monospace; }
  .log-table tr:hover td { background: #1c2128; }
  .tag { display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 11px; font-weight: 600; }
  .tag-buy  { background: rgba(63,185,80,.15); color: var(--green); }
  .tag-sell { background: rgba(248,81,73,.15); color: var(--red); }
  .tag-hold { background: rgba(139,148,158,.12); color: var(--silver); }
  .tag-act  { background: rgba(88,166,255,.12); color: var(--blue); }

  /* Misc */
  .no-data { color: var(--silver); text-align: center; padding: 24px; font-size: 12px; }
  .footer  { text-align: center; color: #444; font-size: 11px; padding: 12px 0 4px; }
  .countdown { font-size: 11px; color: var(--silver); }
</style>
</head>
<body>
<div class="container">

<!-- Header -->
<div class="header">
  <h1>⚡ XAUUSD Meta-Policy — Live Dashboard</h1>
  <div style="display:flex;align-items:center;gap:12px">
    <span class="countdown" id="cdTimer"></span>
    <button class="refresh-btn" onclick="location.reload()">⟳ Refresh</button>
  </div>
</div>

<!-- Status bar -->
<div class="status-bar" style="margin-bottom:14px;padding:8px 12px;background:var(--card);border:1px solid var(--border);border-radius:6px">
  {% if data.server_pid %}
    <span><span class="dot dot-green"></span>Server LIVE (PID {{ data.server_pid }})</span>
  {% else %}
    <span><span class="dot dot-red"></span>Server OFFLINE</span>
  {% endif %}
  <span>Last signal: {{ data.current.ts_full if data.current else "—" }}</span>
  {% if data.stale_min is not none %}
    <span {% if data.stale_min > 3 %}style="color:var(--yellow)"{% endif %}>
      {{ data.stale_min }} min ago
    </span>
  {% endif %}
  <span>Mode: HTTP (broker-blocked TCP)</span>
  <span style="margin-left:auto;color:#444">{{ data.now }}</span>
</div>

<!-- Top row: Signal / Position / Stats -->
<div class="grid-3">

  <!-- Current Signal -->
  <div class="card">
    <div class="card-title">Current Signal</div>
    {% if data.current %}
      {% set d = data.current %}
      {% set dc = "buy" if d.final_dir > 0.5 else ("sell" if d.final_dir < -0.5 else "hold") %}
      <div class="sig-dir {{ dc }}">{{ d.label }}</div>
      <div class="sig-rows">
        <div class="sig-row"><span class="sig-label">Final Dir</span>
          <span class="sig-val">{{ "%.2f"|format(d.final_dir) }}</span></div>
        <div class="sig-row"><span class="sig-label">Actor</span>
          <span class="sig-val">{{ "%.4f"|format(d.actor) }}</span></div>
        <div class="sig-row"><span class="sig-label">Strength</span>
          <span class="sig-val">{{ (d.strength * 100)|round|int }}%</span></div>
        <div class="sig-row"><span class="sig-label">Regime</span>
          <span class="sig-val">{{ "Bull" if d.regime > 0.5 else "Bear" }}</span></div>
      </div>
    {% else %}
      <div class="no-data">No signals yet<br>Waiting for M1 bar…</div>
    {% endif %}
  </div>

  <!-- Open Position -->
  <div class="card">
    <div class="card-title">Open Position</div>
    {% if data.position %}
      {% set p = data.position %}
      {% set pcol = "color:var(--red)" if p.dir == "SELL" else "color:var(--green)" %}
      <div class="pos-dir" style="{{ pcol }}">{{ p.dir }}</div>
      <div class="pos-grid">
        <div class="pos-item">
          <span class="pos-key">Entry</span>
          <span class="pos-val">{{ "%.2f"|format(p.entry) }}</span>
        </div>
        <div class="pos-item">
          <span class="pos-key">SL</span>
          <span class="pos-val" style="color:var(--red)">{{ "%.2f"|format(p.sl) if p.sl else "—" }}</span>
        </div>
        <div class="pos-item">
          <span class="pos-key">TP</span>
          <span class="pos-val" style="color:var(--green)">{{ "%.2f"|format(p.tp) if p.tp else "—" }}</span>
        </div>
        <div class="pos-item">
          <span class="pos-key">Bars held</span>
          <span class="pos-val">{{ p.hold_bars }} / 80</span>
        </div>
      </div>
      <div class="hold-bar-wrap">
        <span style="font-size:10px;color:var(--silver)">Hold progress</span>
        <div class="hold-bar-bg">
          <div class="hold-bar-fill" style="width:{{ (p.hold_frac * 100)|round|int }}%"></div>
        </div>
      </div>
    {% else %}
      <div class="no-data">No open position</div>
    {% endif %}
  </div>

  <!-- Session Stats -->
  <div class="card">
    <div class="card-title">Session Stats ({{ data.stats.total }} bars)</div>
    {% set s = data.stats %}
    <div class="stat-row">
      <span>BUY</span>
      <div style="flex:1;margin:0 10px">
        <div class="stat-bar bar-buy" style="width:{{ s.buy_pct }}%"></div>
      </div>
      <span style="color:var(--green);font-weight:600;min-width:55px;text-align:right">
        {{ s.buy }} ({{ s.buy_pct }}%)</span>
    </div>
    <div class="stat-row">
      <span>SELL</span>
      <div style="flex:1;margin:0 10px">
        <div class="stat-bar bar-sell" style="width:{{ s.sell_pct }}%"></div>
      </div>
      <span style="color:var(--red);font-weight:600;min-width:55px;text-align:right">
        {{ s.sell }} ({{ s.sell_pct }}%)</span>
    </div>
    <div class="stat-row">
      <span>HOLD</span>
      <div style="flex:1;margin:0 10px">
        <div class="stat-bar bar-hold" style="width:{{ s.hold_pct }}%"></div>
      </div>
      <span style="color:var(--silver);font-weight:600;min-width:55px;text-align:right">
        {{ s.hold }} ({{ s.hold_pct }}%)</span>
    </div>
    <div class="stat-row" style="margin-top:6px">
      <span style="color:var(--silver)">Actor mean</span>
      <span style="font-family:monospace;font-weight:600">{{ s.actor_mean }}</span>
    </div>
    <div class="stat-row">
      <span style="color:var(--silver)">Saturation</span>
      <span style="font-weight:600;color:{{ 'var(--red)' if s.saturated_pct > 5 else 'var(--green)' }}">
        {{ s.saturated_pct }}%</span>
    </div>
  </div>

</div><!-- /grid-3 -->

<!-- Chart row -->
<div class="card" style="margin-bottom:14px">
  <div class="card-title">Signal Timeline — final_dir &amp; actor (last {{ data.signals|length }} bars)</div>
  <div class="chart-wrap">
    <canvas id="sigChart"></canvas>
  </div>
</div>

<!-- Bottom row: bar log + overrides -->
<div class="grid-2">

  <!-- Bar log -->
  <div class="card">
    <div class="card-title">Recent Signal Log</div>
    {% if data.signals %}
    <div style="max-height:220px;overflow-y:auto">
    <table class="log-table">
      <thead><tr>
        <th>Time</th><th>Signal</th><th>Actor</th><th>Strength</th><th>Regime</th>
      </tr></thead>
      <tbody>
      {% for s in data.signals[-30:]|reverse %}
      <tr>
        <td>{{ s.ts }}</td>
        <td><span class="tag tag-{{ s.label.lower() }}">{{ s.label }}</span></td>
        <td style="{{ 'color:var(--red)' if s.actor < -0.5 else ('color:var(--green)' if s.actor > 0.5 else '') }}">
          {{ "%.4f"|format(s.actor) }}</td>
        <td>{{ (s.strength * 100)|round|int }}%</td>
        <td>{{ "Bull" if s.regime > 0.5 else "Bear" }}</td>
      </tr>
      {% endfor %}
      </tbody>
    </table>
    </div>
    {% else %}
      <div class="no-data">No signal history yet</div>
    {% endif %}
  </div>

  <!-- Position log + overrides -->
  <div style="display:flex;flex-direction:column;gap:12px">
    <div class="card">
      <div class="card-title">Position Log (last {{ data.pos_rows|length }} rows)</div>
      {% if data.pos_rows %}
      <div style="max-height:100px;overflow-y:auto">
      <table class="log-table">
        <thead><tr><th>Time</th><th>Dir</th><th>Action</th><th>Hold</th></tr></thead>
        <tbody>
        {% for r in data.pos_rows[-10:]|reverse %}
        <tr>
          <td>{{ r.get("datetime","")[-5:] }}</td>
          <td>{{ r.get("pos_dir","") }}</td>
          <td><span class="tag tag-act">{{ r.get("action_taken","") }}</span></td>
          <td>{{ (r.get("hold_frac","0")|float * 80)|round|int }}b</td>
        </tr>
        {% endfor %}
        </tbody>
      </table>
      </div>
      {% else %}
        <div class="no-data" style="padding:12px">Waiting for first bar flush…</div>
      {% endif %}
    </div>

    <div class="card">
      <div class="card-title">Override Log ({{ data.overrides|length }} events)</div>
      {% if data.overrides %}
      <div style="max-height:100px;overflow-y:auto">
      <table class="log-table">
        <thead><tr><th>Time</th><th>Dir</th><th>Lot</th><th>Event</th></tr></thead>
        <tbody>
        {% for r in data.overrides|reverse %}
        <tr>
          <td>{{ r[0][-8:] if r|length > 0 else "" }}</td>
          <td style="{{ 'color:var(--red)' if (r[6]|default('0'))|int < 0 else 'color:var(--green)' }}">
            {{ "SELL" if (r[6]|default("0"))|int < 0 else "BUY" }}</td>
          <td>{{ r[7] if r|length > 7 else "" }}</td>
          <td><span class="tag tag-act">{{ r[-1] if r else "" }}</span></td>
        </tr>
        {% endfor %}
        </tbody>
      </table>
      </div>
      {% else %}
        <div class="no-data" style="padding:12px">No overrides detected</div>
      {% endif %}
    </div>
  </div>

</div><!-- /grid-2 -->

<div class="footer">
  XAUUSD Meta-Policy v0.4 · BC-trained actor (99,761 bars) · obs normalisation ON ·
  <a href="http://localhost:8080">localhost:8080</a>
</div>

</div><!-- /container -->

<!-- Chart.js -->
<script>
const sigs = {{ signals_json }};
const labels = sigs.map(s => s.ts);
const finalDirs = sigs.map(s => s.final_dir);
const actors    = sigs.map(s => s.actor);

const ctx = document.getElementById('sigChart').getContext('2d');
new Chart(ctx, {
  type: 'line',
  data: {
    labels,
    datasets: [
      {
        label: 'final_dir',
        data: finalDirs,
        borderColor: finalDirs.map(d => d > 0.5 ? '#3fb950' : d < -0.5 ? '#f85149' : '#8b949e'),
        backgroundColor: 'transparent',
        stepped: 'before',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0,
      },
      {
        label: 'actor',
        data: actors,
        borderColor: '#58a6ff',
        backgroundColor: 'rgba(88,166,255,0.06)',
        fill: true,
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.3,
      }
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    animation: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { labels: { color: '#8b949e', font: { size: 11 } } },
      tooltip: {
        backgroundColor: '#161b22',
        borderColor: '#30363d',
        borderWidth: 1,
        titleColor: '#e6edf3',
        bodyColor: '#8b949e',
      }
    },
    scales: {
      x: { ticks: { color: '#8b949e', maxTicksLimit: 12, font: { size: 10 } },
           grid: { color: '#21262d' } },
      y: { min: -1.2, max: 1.2,
           ticks: { color: '#8b949e', font: { size: 10 } },
           grid: { color: '#21262d' } }
    }
  }
});

// Auto-refresh countdown
let sec = 30;
const cd = document.getElementById('cdTimer');
setInterval(() => {
  sec--;
  if (sec <= 0) { location.reload(); return; }
  cd.textContent = 'Refresh in ' + sec + 's';
}, 1000);
</script>
</body>
</html>"""

# ── Jinja2-lite: simple template render without Flask dep ─────────────────────
try:
    from jinja2 import Environment
    _jinja = True
except ImportError:
    _jinja = False

def render(data):
    if _jinja:
        from jinja2 import Environment
        env = Environment(autoescape=True)
        env.filters['round'] = round
        tmpl = env.from_string(HTML)
        return tmpl.render(
            data=data,
            signals_json=json.dumps(data["signals"])
        )
    return "<h1>Install jinja2: pip install jinja2</h1>"

# ── HTTP server ───────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress per-request console noise

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/data":
            body = json.dumps(build_data(), default=str).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        else:
            body = render(build_data()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)

if __name__ == "__main__":
    import webbrowser, threading
    port = 8080
    server = HTTPServer(("127.0.0.1", port), Handler)
    url = f"http://localhost:{port}"
    print(f"XAUUSD Dashboard → {url}")
    print("Ctrl+C to stop")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")

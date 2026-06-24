//! XAUUSD signal server v0.3 — std::net::TcpListener (no ZMQ).
//!
//! Modes:
//!   serve   (default)  ./signal_server [--bind 127.0.0.1:5555] [--actor models/actor_weights.json]
//!   replay             ./signal_server replay --bars bars.csv --out signals.csv [--actor ...] [--parallel]
//!
//! Protocol: accepts raw newline-JSON (TCP) and HTTP POST (WebRequest) on the same port.
//! Auto-detected by first line of each connection.

use anyhow::Result;
use log::{info, warn};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::env;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::time::Duration;

mod actor;
mod indicators;
mod scatter;
mod tda;
mod regime;

use actor::{actor_decision, DsacActor};
use indicators::Indicators;
use scatter::ScatterBlock;
use tda::WassersteinTracker;

const SEQ_LEN: usize = 240;
const OBS_DIM: usize = 118;

// ── Wire types ────────────────────────────────────────────────────────────────
#[derive(Deserialize)]
struct SignalRequest {
    bars:          Vec<[f32; 8]>,
    pos_dir:       f32,
    unrealized:    f32,
    hold_fraction: f32,
}

#[derive(Serialize)]
struct SignalResponse {
    obs:              Vec<f32>,
    direction_bias:   f32,
    signal_strength:  f32,
    sl_pips:          f32,
    tp_pips:          f32,
    lot_suggestion:   f32,
    hurst:            f32,
    tda_wasserstein:  f32,
    event_risk:       f32,
    regime:           f32,
    actor_dir:        f32,
    actor_exit:       f32,
    actor_confidence: f32,
    final_dir:        f32,
    should_exit:      bool,
    momentum_exit:    bool,
}

// ── Entry point ───────────────────────────────────────────────────────────────
fn main() -> Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")
    ).init();

    let args: Vec<String> = env::args().collect();

    if args.get(1).map(|s| s.as_str()) == Some("replay") {
        return run_replay(&args);
    }
    run_server(&args)
}

// ── Actor loader ──────────────────────────────────────────────────────────────
fn load_actor(path: Option<String>) -> Option<DsacActor> {
    match path {
        Some(p) => match DsacActor::load(&p) {
            Ok(a)  => { info!("DSAC actor loaded: {}", p); Some(a) }
            Err(e) => { warn!("Actor load failed: {} — rule-only", e); None }
        }
        None => { info!("No --actor provided — rule-based signal only"); None }
    }
}

// ── Live server — accepts both raw-JSON (MT5 TCP) and HTTP POST (MT5 WebRequest) ──
fn run_server(args: &[String]) -> Result<()> {
    let bind_raw = arg_value(args, "--bind")
        .unwrap_or_else(|| "127.0.0.1:5555".into());
    let bind_addr = bind_raw
        .strip_prefix("tcp://")
        .unwrap_or(&bind_raw)
        .to_string();

    let actor = load_actor(arg_value(args, "--actor"));

    let listener = TcpListener::bind(&bind_addr)?;
    info!("Signal server listening on {} (TCP + HTTP POST on same port)", bind_addr);
    info!("Actor: {}", if actor.is_some() { "loaded" } else { "rule-only" });

    let scatter  = ScatterBlock::new();
    let mut tda  = WassersteinTracker::new(60);

    'accept: loop {
        let (stream, addr) = listener.accept()?;
        stream.set_read_timeout(Some(Duration::from_secs(30)))?;

        let mut writer = stream.try_clone()?;
        let mut reader = BufReader::new(stream);

        // Read first line to auto-detect protocol
        let mut first_line = String::new();
        match reader.read_line(&mut first_line) {
            Ok(0)  => continue 'accept,
            Ok(_)  => {}
            Err(e) => { warn!("Read error from {}: {}", addr, e); continue 'accept; }
        }

        if first_line.starts_with("POST") || first_line.starts_with("GET") {
            // HTTP protocol (from MT5 WebRequest) — single request / response / close
            serve_http(&first_line, &mut reader, &mut writer, &scatter, &mut tda, &actor);
            continue 'accept;
        }

        // Raw newline-delimited JSON (persistent connection)
        info!("MT5 connected from {} (raw JSON)", addr);
        let mut line = first_line;
        loop {
            if !line.trim().is_empty() {
                match handle_request(line.trim().as_bytes(), &scatter, &mut tda, &actor) {
                    Ok(resp) => {
                        let mut out = serde_json::to_vec(&resp)?;
                        out.push(b'\n');
                        if let Err(e) = writer.write_all(&out) {
                            warn!("Write error: {}", e);
                            break;
                        }
                    }
                    Err(e) => {
                        warn!("Request error: {}", e);
                        let mut err = serde_json::to_vec(
                            &serde_json::json!({"error": e.to_string()})
                        )?;
                        err.push(b'\n');
                        let _ = writer.write_all(&err);
                    }
                }
            }
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0)  => break,
                Ok(_)  => {}
                Err(e) => { warn!("Read error: {}", e); break; }
            }
        }
        info!("MT5 disconnected");
    }
}

// ── HTTP handler ──────────────────────────────────────────────────────────────
fn serve_http<R: Read>(
    _request_line: &str,
    reader:        &mut BufReader<R>,
    writer:        &mut impl Write,
    scatter:       &ScatterBlock,
    tda:           &mut WassersteinTracker,
    actor:         &Option<DsacActor>,
) {
    let mut content_length = 0usize;
    loop {
        let mut hdr = String::new();
        match reader.read_line(&mut hdr) {
            Ok(0) | Err(_) => return,
            _ => {}
        }
        let t = hdr.trim();
        if t.is_empty() { break; }
        if t.len() > 15 && t[..15].eq_ignore_ascii_case("content-length:") {
            content_length = t[15..].trim().parse().unwrap_or(0);
        }
    }

    if content_length == 0 {
        let _ = writer.write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n");
        return;
    }

    let mut body = vec![0u8; content_length];
    if reader.read_exact(&mut body).is_err() { return; }

    let resp_bytes = match handle_request(&body, scatter, tda, actor) {
        Ok(resp)  => {
            info!("HTTP signal: final_dir={:.2} actor={:.4} strength={:.2} regime={:.2}",
                  resp.final_dir, resp.actor_dir, resp.signal_strength, resp.regime);
            serde_json::to_vec(&resp).unwrap_or_default()
        }
        Err(e)    => {
            warn!("HTTP request error: {}", e);
            serde_json::to_vec(&serde_json::json!({"error": e.to_string()}))
                .unwrap_or_default()
        }
    };

    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        resp_bytes.len()
    );
    let _ = writer.write_all(header.as_bytes());
    let _ = writer.write_all(&resp_bytes);
}

// ── Core signal pipeline — JSON wrapper (live path) ───────────────────────────
fn handle_request(
    raw:     &[u8],
    scatter: &ScatterBlock,
    tda:     &mut WassersteinTracker,
    actor:   &Option<DsacActor>,
) -> Result<SignalResponse> {
    let req: SignalRequest = serde_json::from_slice(raw)?;
    anyhow::ensure!(req.bars.len() == SEQ_LEN,
        "Expected {} bars, got {}", SEQ_LEN, req.bars.len());
    process_bars(&req.bars, req.pos_dir, req.unrealized, req.hold_fraction,
                 scatter, tda, actor)
}

// ── Core signal pipeline — direct (replay path, no JSON overhead) ─────────────
fn process_bars(
    bars:      &[[f32; 8]],
    pos_dir:   f32,
    unrealized: f32,
    hold_frac: f32,
    scatter:   &ScatterBlock,
    tda:       &mut WassersteinTracker,
    actor:     &Option<DsacActor>,
) -> Result<SignalResponse> {
    let close:   Vec<f32> = bars.iter().map(|b| b[3]).collect();
    let high:    Vec<f32> = bars.iter().map(|b| b[1]).collect();
    let low:     Vec<f32> = bars.iter().map(|b| b[2]).collect();
    let volume:  Vec<f32> = bars.iter().map(|b| b[4]).collect();
    let session: Vec<f32> = bars.iter().map(|b| b[6]).collect();
    let event_risk = bars.last().map(|b| b[7]).unwrap_or(0.0);

    let scatter_feats = scatter.forward(bars);
    let ind = Indicators::compute(&close, &high, &low, &volume, &session, event_risk);

    let log_ret: Vec<f32> = close.windows(2)
        .map(|w| (w[1] / w[0].max(1e-8)).ln())
        .collect();
    let tda_w  = tda.update(&log_ret);
    let regime_v = regime::bear_bull_proxy(&close, &volume);

    let (rule_dir, strength) = meta_policy_signal(&ind, tda_w, event_risk, regime_v);
    let atr_last = ind.atr;
    let (sl_pips, tp_pips, lot) = size_trade(rule_dir, strength, atr_last, event_risk);

    let unreal_norm = (unrealized / atr_last.max(1e-8)).tanh();
    let mut obs = Vec::with_capacity(OBS_DIM);
    obs.extend_from_slice(&scatter_feats);
    obs.push(ind.atr_ratio);
    obs.push(ind.vwap_dev);
    obs.push(ind.rsi_63);
    obs.push(ind.bb_pctb);
    obs.push(ind.vol_zscore);
    obs.push(ind.hurst);
    obs.push(tda_w);
    obs.push(session.last().copied().unwrap_or(0.5));
    obs.push(event_risk);
    obs.push(regime_v);
    obs.push(pos_dir);
    obs.push(unreal_norm);
    obs.push(hold_frac);
    obs.push(strength);

    let mom_exit = momentum_exit_rule(pos_dir, &ind);

    let (actor_dir, actor_exit, actor_confidence, final_dir, actor_should_exit) =
        match actor {
            Some(a) => match a.forward(&obs) {
                Ok((ad, ae)) => {
                    let (fd, se, conf) = actor_decision(rule_dir, ad, ae, strength);
                    (ad, ae, conf, fd, se)
                }
                Err(e) => { warn!("Actor forward failed: {}", e); (0.0, 0.0, strength, rule_dir, false) }
            }
            None => (0.0, 0.0, strength, rule_dir, false),
        };

    let should_exit = actor_should_exit || mom_exit;

    Ok(SignalResponse {
        obs,
        direction_bias:  rule_dir,
        signal_strength: strength,
        sl_pips, tp_pips,
        lot_suggestion:  lot,
        hurst:           ind.hurst,
        tda_wasserstein: tda_w,
        event_risk,
        regime:          regime_v,
        actor_dir,
        actor_exit,
        actor_confidence,
        final_dir,
        should_exit,
        momentum_exit:   mom_exit,
    })
}

// ── Offline replay ────────────────────────────────────────────────────────────
//
// Input CSV columns  : datetime,open,high,low,close,tick_volume
// Output CSV columns : datetime,direction_bias,signal_strength,final_dir,
//                      should_exit,hurst,tda_wasserstein,regime,actor_dir,
//                      actor_confidence,sl_pips,tp_pips,lot_suggestion
//
// --obs-out <path>   : writes flat binary of 118D obs vectors
//                      format: [n_rows:u32LE][n_cols:u32LE][n_rows × n_cols × f32LE]
//                      load in Python: np.frombuffer(...).reshape(n, 118)
//
// --parallel         : process in Rayon thread pool (ideal for 2.5M+ bar datasets).
//                      TDA is warmed up with 120-bar lead-in per chunk — output is
//                      equivalent to serial for all but the very first chunk.
fn run_replay(args: &[String]) -> Result<()> {
    let bars_path = arg_value(args, "--bars")
        .ok_or_else(|| anyhow::anyhow!("--bars <path> required for replay"))?;
    let out_path  = arg_value(args, "--out")
        .ok_or_else(|| anyhow::anyhow!("--out <path> required for replay"))?;
    let obs_path  = arg_value(args, "--obs-out");
    let parallel  = args.iter().any(|a| a == "--parallel");

    let actor   = load_actor(arg_value(args, "--actor"));
    let scatter = ScatterBlock::new();

    // ── Parse all bars upfront ────────────────────────────────────────────────
    let content = std::fs::read_to_string(&bars_path)?;
    let mut lines = content.lines();
    let _header = lines.next();

    let mut all_times: Vec<String>   = Vec::new();
    let mut all_bars:  Vec<[f32; 8]> = Vec::new();

    for raw in lines {
        let f: Vec<&str> = raw.splitn(7, ',').collect();
        if f.len() < 6 { continue; }
        let datetime = f[0].trim().to_string();
        let open  = f[1].trim().parse::<f32>().unwrap_or(0.0);
        let high  = f[2].trim().parse::<f32>().unwrap_or(0.0);
        let low   = f[3].trim().parse::<f32>().unwrap_or(0.0);
        let close = f[4].trim().parse::<f32>().unwrap_or(0.0);
        let vol   = f[5].trim().parse::<f32>().unwrap_or(0.0);
        if close <= 0.0 { continue; }
        let session_ph = session_phase_from_dt(&datetime);
        all_times.push(datetime);
        all_bars.push([open, high, low, close, vol, 0.0, session_ph, 0.0]);
    }

    let n = all_bars.len();
    info!("Parsed {} bars from {}", n, bars_path);

    // ── Dispatch to serial or parallel ────────────────────────────────────────
    if parallel {
        return run_replay_parallel(&all_times, &all_bars, &out_path, obs_path, &scatter, &actor);
    }

    // ── Serial replay — zero JSON serialisation overhead ─────────────────────
    let mut tda = WassersteinTracker::new(60);
    let mut out = std::fs::File::create(&out_path)?;
    let mut all_obs: Vec<Vec<f32>> = if obs_path.is_some() { Vec::with_capacity(n) } else { Vec::new() };

    writeln!(out,
        "datetime,direction_bias,signal_strength,final_dir,should_exit,\
         hurst,tda_wasserstein,regime,actor_dir,actor_confidence,\
         sl_pips,tp_pips,lot_suggestion")?;

    let first_sig = SEQ_LEN - 1;
    let mut sig_n = 0u64;

    for i in first_sig..n {
        let window = &all_bars[i + 1 - SEQ_LEN..=i];
        let dt = &all_times[i];

        match process_bars(window, 0.0, 0.0, 0.0, &scatter, &mut tda, &actor) {
            Ok(r) => {
                writeln!(out,
                    "{},{:.4},{:.4},{:.4},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.2},{:.2},{:.3}",
                    dt,
                    r.direction_bias, r.signal_strength, r.final_dir,
                    r.should_exit as u8,
                    r.hurst, r.tda_wasserstein, r.regime,
                    r.actor_dir, r.actor_confidence,
                    r.sl_pips, r.tp_pips, r.lot_suggestion,
                )?;
                if obs_path.is_some() { all_obs.push(r.obs); }
                sig_n += 1;
            }
            Err(e) => warn!("Replay bar {}: {}", dt, e),
        }
    }

    info!("Replay done -- {} bars in, {} rows out -> {}", n, sig_n, out_path);
    write_obs_binary(obs_path, &all_obs)
}

// ── Parallel replay via Rayon ─────────────────────────────────────────────────
//
// Each chunk of CHUNK_SIZE signal bars is processed independently.
// LEAD_IN warm-up calls before each chunk stabilise the TDA Wasserstein tracker.
// Since TDA only looks at the last 60 log-returns (within the 240-bar window),
// 60 warm-up calls are sufficient for exact convergence; we use 120 for safety.
// Rayon's par_iter preserves chunk order, so no sort step is needed.
fn run_replay_parallel(
    all_times: &[String],
    all_bars:  &[[f32; 8]],
    out_path:  &str,
    obs_path:  Option<String>,
    scatter:   &ScatterBlock,
    actor:     &Option<DsacActor>,
) -> Result<()> {
    let n = all_bars.len();
    let first_sig = SEQ_LEN - 1;

    const CHUNK_SIZE: usize = 4096;
    const LEAD_IN:    usize = 120;

    let chunks: Vec<(usize, usize)> = (first_sig..n)
        .step_by(CHUNK_SIZE)
        .map(|s| (s, (s + CHUNK_SIZE).min(n)))
        .collect();

    let n_threads = rayon::current_num_threads();
    info!("Parallel replay: {} bars, {} chunks of {}, {} threads",
          n, chunks.len(), CHUNK_SIZE, n_threads);

    // Process each chunk independently; Rayon preserves ordering in collect()
    let chunk_results: Vec<Vec<(usize, SignalResponse)>> = chunks
        .par_iter()
        .map(|&(sig_start, sig_end)| {
            // Extend backward for TDA warm-up; clamped to first valid signal index
            let warmup_start = sig_start.saturating_sub(LEAD_IN).max(first_sig);
            let mut tda = WassersteinTracker::new(60);
            let mut out = Vec::with_capacity(sig_end - sig_start);

            for i in warmup_start..sig_end {
                let window = &all_bars[i + 1 - SEQ_LEN..=i];
                match process_bars(window, 0.0, 0.0, 0.0, scatter, &mut tda, actor) {
                    Ok(r) if i >= sig_start => out.push((i, r)),
                    _ => {}
                }
            }
            out
        })
        .collect();

    // Write results in chunk order (order is preserved by Rayon)
    let mut out_file = std::fs::File::create(out_path)?;
    writeln!(out_file,
        "datetime,direction_bias,signal_strength,final_dir,should_exit,\
         hurst,tda_wasserstein,regime,actor_dir,actor_confidence,\
         sl_pips,tp_pips,lot_suggestion")?;

    let store_obs = obs_path.is_some();
    let mut all_obs: Vec<Vec<f32>> = if store_obs { Vec::with_capacity(n) } else { Vec::new() };
    let mut sig_n = 0u64;

    for chunk in &chunk_results {
        for (i, r) in chunk {
            writeln!(out_file,
                "{},{:.4},{:.4},{:.4},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.2},{:.2},{:.3}",
                all_times[*i],
                r.direction_bias, r.signal_strength, r.final_dir,
                r.should_exit as u8,
                r.hurst, r.tda_wasserstein, r.regime,
                r.actor_dir, r.actor_confidence,
                r.sl_pips, r.tp_pips, r.lot_suggestion,
            )?;
            if store_obs { all_obs.push(r.obs.clone()); }
            sig_n += 1;
        }
    }

    info!("Parallel replay done -- {} bars in, {} rows out -> {}", n, sig_n, out_path);
    write_obs_binary(obs_path, &all_obs)
}

// ── Obs binary writer ─────────────────────────────────────────────────────────
fn write_obs_binary(obs_path: Option<String>, all_obs: &[Vec<f32>]) -> Result<()> {
    if let Some(ref path) = obs_path {
        let n = all_obs.len() as u32;
        let mut f = std::fs::File::create(path)?;
        f.write_all(&n.to_le_bytes())?;
        f.write_all(&(OBS_DIM as u32).to_le_bytes())?;
        for obs in all_obs {
            for &v in obs {
                f.write_all(&v.to_le_bytes())?;
            }
        }
        info!("Obs binary written: {} rows x {} cols -> {}", n, OBS_DIM, path);
    }
    Ok(())
}

/// Parse session phase from MT5 datetime string "2026.01.01 00:00" or ISO "2026-01-01 00:00:00".
fn session_phase_from_dt(dt: &str) -> f32 {
    let hour: u32 = dt.splitn(2, ' ')
        .nth(1)
        .and_then(|t| t.splitn(2, ':').next())
        .and_then(|h| h.parse().ok())
        .unwrap_or(0);
    if hour < 8  { 0.0 }
    else if hour < 13 { 0.5 }
    else { 1.0 }
}

// ── Rule-based meta-policy ────────────────────────────────────────────────────
fn meta_policy_signal(
    ind:        &Indicators,
    tda_w:      f32,
    event_risk: f32,
    regime:     f32,
) -> (f32, f32) {
    if event_risk >= 1.0 { return (0.0, 0.0); }
    if tda_w > 0.35      { return (0.0, 0.0); }
    let bull    = regime > 0.5;
    let low_vol = ind.vol_zscore < -0.3;
    if bull && low_vol { return (0.0, 0.0); }

    let sell_votes = [
        ind.rsi_63    < 0.35,
        ind.bb_pctb   < 0.25,
        ind.atr_ratio < -0.1,
        ind.vwap_dev  < -0.15,
        !bull,
    ];
    let buy_votes = [
        ind.rsi_63    > 0.65,
        ind.bb_pctb   > 0.75,
        ind.atr_ratio > 0.1,
        ind.vwap_dev  > 0.15,
        bull,
    ];
    let n_sell: f32 = sell_votes.iter().map(|&v| v as u8 as f32).sum();
    let n_buy:  f32 = buy_votes.iter().map(|&v| v as u8 as f32).sum();

    // Direction-aware thresholds: with-trend trades are easier to trigger,
    // counter-trend trades require near-consensus.  Bull/bear asymmetry
    // directly targets the observed -38K bull-regime P&L vs +130K bear-regime.
    let base       = if ind.hurst > 0.50 { 3.0f32 } else { 4.0f32 };
    let buy_thresh  = if bull  { base } else { base + 1.0 };
    let sell_thresh = if !bull { base } else { base + 1.0 };

    if n_sell >= sell_thresh && n_sell > n_buy {
        (-1.0, n_sell / 5.0)
    } else if n_buy >= buy_thresh && n_buy > n_sell {
        (1.0, n_buy / 5.0)
    } else {
        (0.0, 0.0)
    }
}

// Momentum reversal exit: fires when ≥2 of 4 validated indicators confirm the
// position has turned against us.  Requires pos_dir ≠ 0 (live server only;
// replay always passes pos_dir=0 so this is always false there).
//
// Thresholds are grounded: RSI 35/65 = Wilder oversold/overbought boundaries;
// BB%B 0.20/0.80 = ±2σ breakout zone; VWAP-dev ±0.15 ≈ tanh(0.15×ATR) deviation;
// ATR-ratio ±0.10 = tanh-scaled single-bar momentum sign.
// 2/4 majority prevents whipsaws from a single noisy indicator.
fn momentum_exit_rule(pos_dir: f32, ind: &Indicators) -> bool {
    if pos_dir == 0.0 { return false; }
    if pos_dir > 0.0 {
        let votes: usize = [
            ind.rsi_63    < 0.35,
            ind.vwap_dev  < -0.15,
            ind.bb_pctb   < 0.20,
            ind.atr_ratio < -0.10,
        ].iter().filter(|&&v| v).count();
        votes >= 2
    } else {
        let votes: usize = [
            ind.rsi_63    > 0.65,
            ind.vwap_dev  > 0.15,
            ind.bb_pctb   > 0.80,
            ind.atr_ratio > 0.10,
        ].iter().filter(|&&v| v).count();
        votes >= 2
    }
}

fn size_trade(dir: f32, strength: f32, atr: f32, event_risk: f32) -> (f32, f32, f32) {
    if dir == 0.0 { return (0.0, 0.0, 0.0); }
    let em  = 1.0 - event_risk * 0.5;
    let tp  = atr * 1.5 * (1.0 + (strength - 0.33).max(0.0)) * em;
    let sl  = atr * 0.75 * (1.5 - strength * 0.8) * em;
    let k   = ((strength - 0.333) / (1.0 - strength + 1e-6)).clamp(0.0, 0.25);
    let lot = (0.01 * strength * k).max(0.01);
    (sl * 10.0, tp * 10.0, (lot * 100.0).round() / 100.0)
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .map(|w| w[1].clone())
}

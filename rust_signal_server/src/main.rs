//! XAUUSD signal server v0.3 — std::net::TcpListener (no ZMQ).
//!
//! Newline-delimited JSON over a persistent TCP connection.
//! MT5 connects once; each bar sends one JSON line, receives one JSON line.
//!
//! Modes:
//!   serve   (default)  ./signal_server [--bind 127.0.0.1:5555] [--actor models/actor_weights.json]
//!   replay             ./signal_server replay --bars historical.csv --out signals.csv [--actor ...]

use anyhow::Result;
use log::{info, warn};
use serde::{Deserialize, Serialize};
use std::env;
use std::io::{BufRead, BufReader, Write};
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

// ── Live TCP server ───────────────────────────────────────────────────────────
fn run_server(args: &[String]) -> Result<()> {
    let bind_raw = arg_value(args, "--bind")
        .unwrap_or_else(|| "127.0.0.1:5555".into());
    // Accept both "127.0.0.1:5555" and legacy "tcp://127.0.0.1:5555"
    let bind_addr = bind_raw
        .strip_prefix("tcp://")
        .unwrap_or(&bind_raw)
        .to_string();

    let actor = load_actor(arg_value(args, "--actor"));

    let listener = TcpListener::bind(&bind_addr)?;
    info!("Signal server listening on {} (TCP)", bind_addr);
    info!("Actor: {}", if actor.is_some() { "loaded" } else { "rule-only" });

    let scatter  = ScatterBlock::new();
    let mut tda  = WassersteinTracker::new(60);

    'accept: loop {
        let (stream, addr) = listener.accept()?;
        info!("MT5 connected from {}", addr);
        stream.set_read_timeout(Some(Duration::from_secs(30)))?;

        let mut writer = stream.try_clone()?;
        let mut reader = BufReader::new(stream);
        let mut line   = String::new();

        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0)  => { info!("MT5 disconnected"); continue 'accept; }
                Ok(_)  => {}
                Err(e) => { warn!("Read error: {}", e); continue 'accept; }
            }

            match handle_request(line.trim().as_bytes(), &scatter, &mut tda, &actor) {
                Ok(resp) => {
                    let mut out = serde_json::to_vec(&resp)?;
                    out.push(b'\n');
                    if let Err(e) = writer.write_all(&out) {
                        warn!("Write error: {}", e);
                        continue 'accept;
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
    }
}

// ── Offline replay ────────────────────────────────────────────────────────────
//
// Input CSV columns  : datetime,open,high,low,close,tick_volume
// Output CSV columns : datetime,direction_bias,signal_strength,final_dir,
//                      should_exit,hurst,tda_wasserstein,regime,actor_dir,
//                      actor_confidence,sl_pips,tp_pips,lot_suggestion
//
// pos_dir / unrealized / hold_fraction are held at 0 (flat-position assumption).
// TDA tracker is stateful and processes bars in sequence — do not shuffle input.
fn run_replay(args: &[String]) -> Result<()> {
    let bars_path = arg_value(args, "--bars")
        .ok_or_else(|| anyhow::anyhow!("--bars <path> required for replay"))?;
    let out_path  = arg_value(args, "--out")
        .ok_or_else(|| anyhow::anyhow!("--out <path> required for replay"))?;

    let actor   = load_actor(arg_value(args, "--actor"));
    let scatter = ScatterBlock::new();
    let mut tda = WassersteinTracker::new(60);

    let content = std::fs::read_to_string(&bars_path)?;
    let mut lines = content.lines();
    let _header = lines.next(); // skip header

    let mut window: Vec<[f32; 8]> = Vec::with_capacity(SEQ_LEN);
    let mut out   = std::fs::File::create(&out_path)?;

    writeln!(out,
        "datetime,direction_bias,signal_strength,final_dir,should_exit,\
         hurst,tda_wasserstein,regime,actor_dir,actor_confidence,\
         sl_pips,tp_pips,lot_suggestion")?;

    let (mut bar_n, mut sig_n) = (0u64, 0u64);

    for raw in lines {
        // Columns: datetime,open,high,low,close,tick_volume[,...]
        let f: Vec<&str> = raw.splitn(7, ',').collect();
        if f.len() < 6 { continue; }

        let datetime = f[0].trim();
        let open  = f[1].trim().parse::<f32>().unwrap_or(0.0);
        let high  = f[2].trim().parse::<f32>().unwrap_or(0.0);
        let low   = f[3].trim().parse::<f32>().unwrap_or(0.0);
        let close = f[4].trim().parse::<f32>().unwrap_or(0.0);
        let vol   = f[5].trim().parse::<f32>().unwrap_or(0.0);
        if close <= 0.0 { continue; }

        // Derive session_phase from bar datetime; event_risk = 0.0 (no calendar)
        let session_ph = session_phase_from_dt(datetime);
        let bar: [f32; 8] = [open, high, low, close, vol, 0.0, session_ph, 0.0];

        if window.len() == SEQ_LEN { window.remove(0); }
        window.push(bar);
        bar_n += 1;

        if window.len() < SEQ_LEN { continue; }

        let req_json = serde_json::json!({
            "bars":          window,
            "pos_dir":       0.0_f32,
            "unrealized":    0.0_f32,
            "hold_fraction": 0.0_f32
        });
        let req_bytes = serde_json::to_vec(&req_json)?;

        match handle_request(&req_bytes, &scatter, &mut tda, &actor) {
            Ok(r) => {
                writeln!(out,
                    "{},{:.4},{:.4},{:.4},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.2},{:.2},{:.3}",
                    datetime,
                    r.direction_bias, r.signal_strength, r.final_dir,
                    r.should_exit as u8,
                    r.hurst, r.tda_wasserstein, r.regime,
                    r.actor_dir, r.actor_confidence,
                    r.sl_pips, r.tp_pips, r.lot_suggestion,
                )?;
                sig_n += 1;
            }
            Err(e) => warn!("Replay bar {}: {}", datetime, e),
        }
    }

    info!("Replay done — {} bars in, {} rows out → {}", bar_n, sig_n, out_path);
    Ok(())
}

/// Parse session phase from MT5 datetime string "2026.01.01 00:00" or "2026.01.01 00:00:00".
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

// ── Core signal pipeline ──────────────────────────────────────────────────────
fn handle_request(
    raw:     &[u8],
    scatter: &ScatterBlock,
    tda:     &mut WassersteinTracker,
    actor:   &Option<DsacActor>,
) -> Result<SignalResponse> {
    let req: SignalRequest = serde_json::from_slice(raw)?;
    anyhow::ensure!(req.bars.len() == SEQ_LEN,
        "Expected {} bars, got {}", SEQ_LEN, req.bars.len());

    let close:   Vec<f32> = req.bars.iter().map(|b| b[3]).collect();
    let high:    Vec<f32> = req.bars.iter().map(|b| b[1]).collect();
    let low:     Vec<f32> = req.bars.iter().map(|b| b[2]).collect();
    let volume:  Vec<f32> = req.bars.iter().map(|b| b[4]).collect();
    let session: Vec<f32> = req.bars.iter().map(|b| b[6]).collect();
    let event_risk = req.bars.last().map(|b| b[7]).unwrap_or(0.0);

    let scatter_feats = scatter.forward(&req.bars);
    let ind = Indicators::compute(&close, &high, &low, &volume, &session, event_risk);

    let log_ret: Vec<f32> = close.windows(2)
        .map(|w| (w[1] / w[0].max(1e-8)).ln())
        .collect();
    let tda_w  = tda.update(&log_ret);
    let regime = regime::bear_bull_proxy(&close, &volume);

    let (rule_dir, strength) = meta_policy_signal(&ind, tda_w, event_risk, regime);
    let atr_last = ind.atr;
    let (sl_pips, tp_pips, lot) = size_trade(rule_dir, strength, atr_last, event_risk);

    let unreal_norm = (req.unrealized / atr_last.max(1e-8)).tanh();
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
    obs.push(regime);
    obs.push(req.pos_dir);
    obs.push(unreal_norm);
    obs.push(req.hold_fraction);
    obs.push(strength);

    let (actor_dir, actor_exit, actor_confidence, final_dir, should_exit) =
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

    Ok(SignalResponse {
        obs,
        direction_bias:  rule_dir,
        signal_strength: strength,
        sl_pips, tp_pips,
        lot_suggestion:  lot,
        hurst:           ind.hurst,
        tda_wasserstein: tda_w,
        event_risk,
        regime,
        actor_dir,
        actor_exit,
        actor_confidence,
        final_dir,
        should_exit,
    })
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
    let thresh  = if ind.hurst > 0.50 { 3.0 } else { 4.0 };

    if n_sell >= thresh && n_sell > n_buy {
        (-1.0, n_sell / 5.0)
    } else if n_buy >= thresh && n_buy > n_sell {
        (1.0, n_buy / 5.0)
    } else {
        (0.0, 0.0)
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

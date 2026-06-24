//+------------------------------------------------------------------+
//|  AutoTradeEA.mq5  --  XAUUSD Meta-Policy v0.4 auto-executor     |
//|                                                                    |
//|  PURPOSE:                                                          |
//|    Executes signals from the Rust signal_server automatically,     |
//|    capturing real position-state features (pos_dir, unrealized,    |
//|    hold_fraction) that are always 0 in offline replay mode.        |
//|                                                                    |
//|  MODES:                                                            |
//|    Live  : TCP connect to Rust server, send position context,      |
//|            receive final_dir/sl/tp/lot, execute order.             |
//|    Tester: loads pre-computed SIGNALS_CSV (replay output),         |
//|            executes signals, logs position state per bar.          |
//|                                                                    |
//|  OUTPUT (both modes):                                              |
//|    position_log.csv in MT5 Common\Files -- per-bar log of:        |
//|    datetime, pos_dir, hold_frac, unrealized_norm, signal_dir,     |
//|    action_taken, entry_price, sl_price, tp_price                  |
//|    Join to signals.csv by datetime to get 118D obs with real       |
//|    position context for DSAC buffer enrichment.                    |
//+------------------------------------------------------------------+
#property copyright "XAUUSD Meta-Policy v0.4"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

//-- Inputs
input string SERVER_HOST   = "127.0.0.1";
input int    SERVER_PORT   = 5555;
input string SIGNALS_CSV   = "signals.csv";        // tester: filename in Common\Files
input string POS_LOG_CSV   = "position_log.csv";   // output log (Common\Files)
input double BASE_LOT      = 0.01;
input int    MAX_HOLD_BARS = 80;
input double SL_MULT       = 1.0;   // scale SL from signal (1.0 = use as-is)
input double TP_MULT       = 1.0;   // scale TP from signal

//-- Precomputed signal row
struct PrecompRow {
    string bar_dt;      // renamed: 'datetime' is a reserved keyword in MQL5
    double final_dir;
    double signal_strength;
    double sl_pips;
    double tp_pips;
    double lot_suggestion;
};

//-- Globals
CTrade          g_trade;
bool            g_is_tester     = false;
bool            g_tcp_ok        = false;
bool            g_use_http      = false;   // true when broker blocks SocketConnect (err 4014)
int             g_socket        = INVALID_HANDLE;

PrecompRow      g_precomp[];
int             g_precomp_count = 0;
int             g_precomp_cursor = 0;

int             g_log_handle    = INVALID_HANDLE;
int             g_hold_bars     = 0;
double          g_entry_price   = 0.0;
double          g_pos_dir       = 0.0;   // +1=long, -1=short, 0=flat

datetime        g_prev_bar_time = 0;

//-- Momentum-exit indicator handles (persistent — created once in OnInit)
int             g_rsi_handle    = INVALID_HANDLE;
int             g_bb_handle     = INVALID_HANDLE;
int             g_matr_handle   = INVALID_HANDLE;

//+------------------------------------------------------------------+

int OnInit() {
    g_is_tester = (bool)MQLInfoInteger(MQL_TESTER);
    g_trade.SetExpertMagicNumber(20240001);
    g_trade.SetDeviationInPoints(30);

    // Open position log
    g_log_handle = FileOpen(POS_LOG_CSV,
        FILE_WRITE|FILE_CSV|FILE_ANSI|FILE_COMMON, ',');
    if(g_log_handle == INVALID_HANDLE) {
        Print("AutoTradeEA: cannot open log file ", POS_LOG_CSV, " err:", GetLastError());
        return INIT_FAILED;
    }
    FileWrite(g_log_handle,
        "datetime", "pos_dir", "hold_frac", "unrealized_norm",
        "signal_dir", "action_taken", "entry_price", "sl_price", "tp_price");

    //-- Momentum-exit indicators (used in both live and tester modes)
    g_rsi_handle  = iRSI(Symbol(),   PERIOD_M1, 63, PRICE_CLOSE);
    g_bb_handle   = iBands(Symbol(), PERIOD_M1, 20, 0, 2.0, PRICE_CLOSE);
    g_matr_handle = iATR(Symbol(),   PERIOD_M1, 14);
    if(g_rsi_handle == INVALID_HANDLE || g_bb_handle == INVALID_HANDLE || g_matr_handle == INVALID_HANDLE)
        Print("AutoTradeEA: WARNING — momentum-exit indicator init failed (err=", GetLastError(), ")");

    if(g_is_tester) {
        if(!LoadPrecomp()) {
            Print("AutoTradeEA: failed to load signals CSV");
            return INIT_FAILED;
        }
        Print("AutoTradeEA [TESTER]: loaded ", g_precomp_count, " rows from ", SIGNALS_CSV);
    } else {
        ConnectServer();
        Print("AutoTradeEA [LIVE]: TCP ", g_tcp_ok ? "connected" : "failed — will retry");
    }

    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+

void OnDeinit(const int reason) {
    if(g_log_handle != INVALID_HANDLE) {
        FileClose(g_log_handle);
        g_log_handle = INVALID_HANDLE;
    }
    if(g_socket != INVALID_HANDLE) {
        SocketClose(g_socket);
        g_socket = INVALID_HANDLE;
    }
    if(g_rsi_handle  != INVALID_HANDLE) { IndicatorRelease(g_rsi_handle);  g_rsi_handle  = INVALID_HANDLE; }
    if(g_bb_handle   != INVALID_HANDLE) { IndicatorRelease(g_bb_handle);   g_bb_handle   = INVALID_HANDLE; }
    if(g_matr_handle != INVALID_HANDLE) { IndicatorRelease(g_matr_handle); g_matr_handle = INVALID_HANDLE; }
    Print("AutoTradeEA: stopped. Reason=", reason);
}

//+------------------------------------------------------------------+

void OnTick() {
    // Fire once per new M1 bar
    datetime bar_time = (datetime)SeriesInfoInteger(Symbol(), PERIOD_M1, SERIES_LASTBAR_DATE);
    if(bar_time == g_prev_bar_time) return;
    g_prev_bar_time = bar_time;

    string dt = TimeToString(bar_time, TIME_DATE|TIME_MINUTES);
    StringReplace(dt, ".", ".");  // keep MT5 dot-separated format
    // MT5 format: "2026.06.01 00:00" (TimeToString default)

    // Sync position state from broker
    RefreshPositionState();

    double hold_frac      = (g_hold_bars > 0) ? MathMin((double)g_hold_bars / MAX_HOLD_BARS, 1.0) : 0.0;
    double unrealized_raw = (g_pos_dir != 0) ? GetUnrealizedPnl() : 0.0;
    double atr_val        = GetATR();
    double unrealized_norm = (atr_val > 1e-8) ? MathTanh(unrealized_raw / atr_val) : 0.0;

    // Get signal
    double sig_dir = 0, sl_px = 0, tp_px = 0, lot = BASE_LOT;
    bool   should_exit = false;
    string action_taken = "HOLD";

    if(g_is_tester) {
        // Advance cursor to matching datetime
        while(g_precomp_cursor < g_precomp_count &&
              g_precomp[g_precomp_cursor].bar_dt < dt)
            g_precomp_cursor++;

        if(g_precomp_cursor < g_precomp_count &&
           g_precomp[g_precomp_cursor].bar_dt == dt) {
            int c = g_precomp_cursor;
            sig_dir = g_precomp[c].final_dir;
            sl_px   = g_precomp[c].sl_pips;
            tp_px   = g_precomp[c].tp_pips;
            lot     = (g_precomp[c].lot_suggestion > 0)
                      ? g_precomp[c].lot_suggestion : BASE_LOT;
        }
    } else {
        // Live mode — HTTP (WebRequest) if broker blocked raw socket, else TCP
        if(g_use_http) {
            GetSignalHTTP(g_pos_dir, unrealized_raw, hold_frac,
                          sig_dir, sl_px, tp_px, lot, should_exit);
        } else {
            if(!g_tcp_ok) ConnectServer();
            if(g_tcp_ok) {
                GetSignalTCP(g_pos_dir, unrealized_raw, hold_frac,
                             sig_dir, sl_px, tp_px, lot, should_exit);
            }
        }
    }

    // -- Momentum-exit check (runs in both live and tester modes; uses real bar data
    //    so fires correctly even when server-side should_exit is from pos_dir=0 replay)
    bool mom_exit = MomentumExitSignal(g_pos_dir);

    // -- Execute trade decision
    double entry_price = 0, actual_sl = 0, actual_tp = 0;

    // Check max hold timeout
    if(g_pos_dir != 0 && g_hold_bars >= MAX_HOLD_BARS) {
        ClosePosition();
        action_taken = "TIMEOUT";
    } else if(mom_exit && g_pos_dir != 0) {
        ClosePosition();
        action_taken = "MOM_EXIT";
    } else if(should_exit && g_pos_dir != 0) {
        ClosePosition();
        action_taken = "EXIT_SIGNAL";
    } else if(sig_dir > 0.5 && g_pos_dir == 0) {
        // BUY
        double ask = SymbolInfoDouble(Symbol(), SYMBOL_ASK);
        actual_sl  = (sl_px > 0) ? ask - sl_px * SL_MULT : 0;
        actual_tp  = (tp_px > 0) ? ask + tp_px * TP_MULT : 0;
        if(g_trade.Buy(lot, Symbol(), ask, actual_sl, actual_tp, "AutoTradeEA BUY")) {
            g_entry_price = ask;
            g_hold_bars   = 0;
            action_taken  = "BUY";
            entry_price   = ask;
        }
    } else if(sig_dir < -0.5 && g_pos_dir == 0) {
        // SELL
        double bid = SymbolInfoDouble(Symbol(), SYMBOL_BID);
        actual_sl  = (sl_px > 0) ? bid + sl_px * SL_MULT : 0;
        actual_tp  = (tp_px > 0) ? bid - tp_px * TP_MULT : 0;
        if(g_trade.Sell(lot, Symbol(), bid, actual_sl, actual_tp, "AutoTradeEA SELL")) {
            g_entry_price = bid;
            g_hold_bars   = 0;
            action_taken  = "SELL";
            entry_price   = bid;
        }
    } else if(g_pos_dir != 0) {
        g_hold_bars++;
        action_taken = "HOLD_POS";
    }

    // Log bar
    if(g_log_handle != INVALID_HANDLE) {
        FileWrite(g_log_handle,
            dt,
            DoubleToString(g_pos_dir, 1),
            DoubleToString(hold_frac, 4),
            DoubleToString(unrealized_norm, 4),
            DoubleToString(sig_dir, 1),
            action_taken,
            DoubleToString(entry_price > 0 ? entry_price : g_entry_price, 5),
            DoubleToString(actual_sl, 5),
            DoubleToString(actual_tp, 5)
        );
        FileFlush(g_log_handle);
    }
}

//+------------------------------------------------------------------+
//  Momentum-exit rule (mirrors Rust momentum_exit_rule in main.rs)
//
//  Validated indicator selection: RSI-63 uses Wilder 35/65 boundaries;
//  BB%B uses ±2σ (0.20/0.80); ATR-ratio is tanh-scaled bar momentum.
//  ≥2 of 3 indicators must agree to fire — prevents single-indicator whipsaws.
//+------------------------------------------------------------------+

bool MomentumExitSignal(double pos_dir) {
    if(pos_dir == 0) return false;
    if(g_rsi_handle == INVALID_HANDLE || g_bb_handle == INVALID_HANDLE || g_matr_handle == INVALID_HANDLE)
        return false;

    double rsi_buf[1], bb_upper[1], bb_lower[1], close_buf[2], atr_buf[1];
    if(CopyBuffer(g_rsi_handle,  0, 0, 1, rsi_buf)   < 1) return false;
    if(CopyBuffer(g_bb_handle,   1, 0, 1, bb_upper)  < 1) return false;  // upper band
    if(CopyBuffer(g_bb_handle,   2, 0, 1, bb_lower)  < 1) return false;  // lower band
    if(CopyClose(Symbol(), PERIOD_M1, 0, 2, close_buf) < 2) return false;
    if(CopyBuffer(g_matr_handle, 0, 0, 1, atr_buf)   < 1) return false;

    double rsi     = rsi_buf[0] / 100.0;
    double bb_rng  = bb_upper[0] - bb_lower[0];
    double bb_pctb = (bb_rng > 1e-8) ? (close_buf[0] - bb_lower[0]) / bb_rng : 0.5;
    double atr_r   = (atr_buf[0] > 1e-8) ? MathTanh((close_buf[0] - close_buf[1]) / atr_buf[0]) : 0.0;

    int votes = 0;
    if(pos_dir > 0) {
        if(rsi     < 0.35)  votes++;
        if(bb_pctb < 0.20)  votes++;
        if(atr_r   < -0.10) votes++;
    } else {
        if(rsi     > 0.65) votes++;
        if(bb_pctb > 0.80) votes++;
        if(atr_r   > 0.10) votes++;
    }
    return (votes >= 2);
}

//+------------------------------------------------------------------+
//  Position helpers
//+------------------------------------------------------------------+

void RefreshPositionState() {
    if(PositionSelect(Symbol())) {
        long type = PositionGetInteger(POSITION_TYPE);
        g_pos_dir = (type == POSITION_TYPE_BUY) ? 1.0 : -1.0;
    } else {
        if(g_pos_dir != 0) {
            // Position was closed externally (SL/TP hit)
            g_hold_bars   = 0;
            g_entry_price = 0.0;
        }
        g_pos_dir = 0.0;
    }
}

double GetUnrealizedPnl() {
    if(!PositionSelect(Symbol())) return 0.0;
    return PositionGetDouble(POSITION_PROFIT);
}

double GetATR() {
    int    handle = iATR(Symbol(), PERIOD_M1, 14);
    double buf[1];
    if(handle == INVALID_HANDLE || CopyBuffer(handle, 0, 0, 1, buf) < 1) return 1.0;
    IndicatorRelease(handle);
    return (buf[0] > 0) ? buf[0] : 1.0;
}

void ClosePosition() {
    if(PositionSelect(Symbol())) {
        g_trade.PositionClose(Symbol());
        g_hold_bars   = 0;
        g_entry_price = 0.0;
        g_pos_dir     = 0.0;
    }
}

//+------------------------------------------------------------------+
//  TCP live mode helpers
//+------------------------------------------------------------------+

void ConnectServer() {
    if(g_socket != INVALID_HANDLE) {
        SocketClose(g_socket);
        g_socket = INVALID_HANDLE;
    }
    g_socket = SocketCreate();
    if(g_socket == INVALID_HANDLE) {
        int err = GetLastError();
        Print("AutoTradeEA: SocketCreate FAILED err=", err);
        g_tcp_ok = false;
        return;
    }
    if(!SocketConnect(g_socket, SERVER_HOST, SERVER_PORT, 3000)) {
        int err = GetLastError();
        SocketClose(g_socket);
        g_socket = INVALID_HANDLE;
        g_tcp_ok = false;
        if(err == 4014) {
            // Broker blocks raw socket connections — switch to HTTP (WebRequest) mode.
            // ACTION REQUIRED: Tools → Options → Expert Advisors → Allow WebRequest → add:
            //   http://127.0.0.1:5555
            Print("AutoTradeEA: broker blocks SocketConnect (err 4014) → HTTP mode enabled");
            Print("  ACTION: Tools→Options→Expert Advisors→Allow WebRequest for listed URL");
            Print("  Add: http://127.0.0.1:5555  then click OK and re-attach EA");
            g_use_http = true;
        } else {
            Print("AutoTradeEA: SocketConnect FAILED to ", SERVER_HOST, ":", SERVER_PORT,
                  " err=", err, " — is signal_server.exe running? (run start_live.ps1)");
        }
        return;
    }
    g_tcp_ok = true;
    Print("AutoTradeEA: TCP connected to ", SERVER_HOST, ":", SERVER_PORT);
}

//-- Shared bar-window builder used by both TCP and HTTP paths.
// Returns the full request JSON string.  ok=false if CopyRates failed.
string BuildBarsRequest(double pos_dir, double unrealized, double hold_frac, bool &ok) {
    ok = false;
    MqlRates rates[240];
    int copied = CopyRates(Symbol(), PERIOD_M1, 0, 240, rates);
    if(copied < 240) {
        Print("BuildBarsRequest: CopyRates returned ", copied, " bars (need 240)");
        return "";
    }
    string bars_json = "[";
    for(int i = 0; i < 240; i++) {
        if(i > 0) bars_json += ",";
        bars_json += StringFormat("[%.5f,%.5f,%.5f,%.5f,%.1f,0.0,%.1f,0.0]",
            rates[i].open, rates[i].high, rates[i].low, rates[i].close,
            (double)rates[i].tick_volume, SessionPhase(rates[i].time));
    }
    bars_json += "]";
    string tail = StringFormat(
        ",\"pos_dir\":%.1f,\"unrealized\":%.4f,\"hold_fraction\":%.4f}\n",
        pos_dir, unrealized, hold_frac);
    ok = true;
    return "{\"bars\":" + bars_json + tail;
}

//-- HTTP fallback: uses MT5 WebRequest() instead of raw socket.
// Requires http://127.0.0.1:5555 whitelisted in Tools→Options→Expert Advisors.
bool GetSignalHTTP(double pos_dir, double unrealized, double hold_frac,
                   double &out_dir, double &out_sl, double &out_tp,
                   double &out_lot, bool &out_exit) {
    out_dir  = 0; out_sl = 0; out_tp = 0; out_lot = BASE_LOT; out_exit = false;

    bool ok;
    string req = BuildBarsRequest(pos_dir, unrealized, hold_frac, ok);
    if(!ok) return false;

    uchar req_data[];
    StringToCharArray(req, req_data, 0, StringLen(req));

    uchar  resp_data[];
    string resp_headers;
    int code = WebRequest(
        "POST",
        StringFormat("http://%s:%d/", SERVER_HOST, SERVER_PORT),
        "Content-Type: application/json\r\n",
        5000,
        req_data,
        resp_data,
        resp_headers
    );

    if(code != 200) {
        Print("GetSignalHTTP: HTTP code=", code, " err=", GetLastError(),
              " — ensure http://127.0.0.1:5555 is in Tools→Options→Expert Advisors whitelist");
        return false;
    }

    string resp = CharArrayToString(resp_data);
    out_dir  = JsonDouble(resp, "final_dir");
    out_sl   = JsonDouble(resp, "sl_pips");
    out_tp   = JsonDouble(resp, "tp_pips");
    out_lot  = JsonDouble(resp, "lot_suggestion");
    out_exit = (JsonDouble(resp, "should_exit") > 0.5);
    return true;
}

bool GetSignalTCP(double pos_dir, double unrealized, double hold_frac,
                  double &out_dir, double &out_sl, double &out_tp,
                  double &out_lot, bool &out_exit) {
    out_dir  = 0; out_sl = 0; out_tp = 0; out_lot = BASE_LOT; out_exit = false;

    bool ok;
    string req = BuildBarsRequest(pos_dir, unrealized, hold_frac, ok);
    if(!ok) return false;

    uchar req_bytes[];
    StringToCharArray(req, req_bytes, 0, StringLen(req));
    if(SocketSend(g_socket, req_bytes, ArraySize(req_bytes)) < 0) {
        Print("GetSignalTCP: SocketSend failed");
        g_tcp_ok = false;
        return false;
    }

    string resp = SocketReadLine();
    if(resp == "") {
        Print("GetSignalTCP: no response (timeout or disconnect)");
        g_tcp_ok = false;
        return false;
    }

    out_dir  = JsonDouble(resp, "final_dir");
    out_sl   = JsonDouble(resp, "sl_pips");
    out_tp   = JsonDouble(resp, "tp_pips");
    out_lot  = JsonDouble(resp, "lot_suggestion");
    out_exit = (JsonDouble(resp, "should_exit") > 0.5);
    return true;
}

string SocketReadLine() {
    string result = "";
    uchar  buf[1];
    uint   timeout_ms = 5000;
    ulong  t0 = GetTickCount64();
    while(GetTickCount64() - t0 < timeout_ms) {
        uint avail = SocketIsReadable(g_socket);
        if(avail == 0) { Sleep(5); continue; }
        uchar chunk[2048];
        int n = SocketRead(g_socket, chunk, (uint)MathMin((int)avail, 2048), 1000);
        for(int i = 0; i < n; i++) {
            if(chunk[i] == '\n') return result;
            result += CharToString(chunk[i]);
        }
    }
    return result;
}

double SessionPhase(datetime t) {
    MqlDateTime dt;
    TimeToStruct(t, dt);
    if(dt.hour <  8) return 0.0;
    if(dt.hour < 13) return 0.5;
    return 1.0;
}

double JsonDouble(const string &json, const string &key) {
    string search = "\"" + key + "\":";
    int pos = StringFind(json, search);
    if(pos < 0) return 0.0;
    pos += StringLen(search);
    // skip whitespace
    while(pos < StringLen(json) && StringGetCharacter(json, pos) == ' ') pos++;
    // read until comma, } or end
    string val = "";
    int len = StringLen(json);
    while(pos < len) {
        ushort c = StringGetCharacter(json, pos);
        if(c == ',' || c == '}' || c == ']') break;
        val += ShortToString(c);
        pos++;
    }
    return StringToDouble(val);
}

//+------------------------------------------------------------------+
//  Precomputed CSV loader (tester mode)
//  Same format as signals.csv from Rust replay.
//+------------------------------------------------------------------+

bool LoadPrecomp() {
    int fh = FileOpen(SIGNALS_CSV,
        FILE_READ|FILE_CSV|FILE_ANSI|FILE_COMMON, ',');
    if(fh == INVALID_HANDLE) {
        Print("LoadPrecomp: cannot open ", SIGNALS_CSV, " err:", GetLastError());
        return false;
    }

    // Skip header
    if(!FileIsEnding(fh)) {
        for(int col = 0; col < 13; col++) FileReadString(fh);
    }

    ArrayResize(g_precomp, 200000);
    g_precomp_count = 0;

    while(!FileIsEnding(fh)) {
        // Columns: datetime,direction_bias,signal_strength,final_dir,should_exit,
        //          hurst,tda_wasserstein,regime,actor_dir,actor_confidence,
        //          sl_pips,tp_pips,lot_suggestion
        string dt = FileReadString(fh);
        if(FileIsEnding(fh) || StringLen(dt) == 0) break;

        double dir_bias    = StringToDouble(FileReadString(fh));
        double strength    = StringToDouble(FileReadString(fh));
        double final_dir   = StringToDouble(FileReadString(fh));
        int    should_exit = (int)StringToInteger(FileReadString(fh));
        double hurst       = StringToDouble(FileReadString(fh));
        double tda         = StringToDouble(FileReadString(fh));
        double regime      = StringToDouble(FileReadString(fh));
        double actor_dir   = StringToDouble(FileReadString(fh));
        double actor_conf  = StringToDouble(FileReadString(fh));
        double sl_pips     = StringToDouble(FileReadString(fh));
        double tp_pips     = StringToDouble(FileReadString(fh));
        double lot_sug     = StringToDouble(FileReadString(fh));

        if(g_precomp_count >= ArraySize(g_precomp))
            ArrayResize(g_precomp, ArraySize(g_precomp) + 50000);

        int c = g_precomp_count;
        g_precomp[c].bar_dt         = dt;
        g_precomp[c].final_dir      = final_dir;
        g_precomp[c].signal_strength = strength;
        g_precomp[c].sl_pips        = sl_pips;
        g_precomp[c].tp_pips        = tp_pips;
        g_precomp[c].lot_suggestion = lot_sug;
        g_precomp_count++;
    }

    FileClose(fh);
    ArrayResize(g_precomp, g_precomp_count);
    return (g_precomp_count > 0);
}

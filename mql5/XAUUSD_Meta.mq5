//+------------------------------------------------------------------+
//| XAUUSD Meta-Policy Indicator v0.3                                |
//| Native TCP socket → Rust signal server (no ZMQ library)          |
//| Protocol: newline-delimited JSON over persistent TCP connection   |
//| Displays signal, sizing, SL/TP — no execution                    |
//| Logs user overrides to CSV for DSAC training                     |
//| Backtest mode: reads pre-computed signals from CSV               |
//+------------------------------------------------------------------+
#property indicator_chart_window
#property indicator_buffers 0
#property indicator_plots   0

//--- Connection parameters
input string SERVER_HOST  = "127.0.0.1";
input int    SERVER_PORT  = 5555;
input string LOG_PATH     = "C:\\Program Files\\MetaTrader 5\\override_log.csv";
input string CONTEXT_PATH = "C:\\Program Files\\MetaTrader 5\\session_context.json";
input int    BARS_HISTORY = 240;
input bool   SHOW_PANEL   = true;
//--- Backtest: path to pre-computed signals CSV from Rust replay mode
//    Leave empty for live trading
input string SIGNALS_CSV  = "";

//--- Native socket handle
int  g_socket    = INVALID_HANDLE;
bool g_connected = false;

//--- Signal cache (populated from server or from pre-computed CSV)
double   g_direction   = 0;
double   g_strength    = 0;
double   g_sl_pips     = 0;
double   g_tp_pips     = 0;
double   g_lot         = 0.01;
double   g_hurst       = 0.5;
double   g_tda_w       = 0;
double   g_event_risk  = 0;
double   g_regime      = 0.5;
double   g_actor_dir   = 0;
double   g_final_dir   = 0;
double   g_actor_conf  = 0;
bool     g_should_exit = false;
datetime g_last_bar    = 0;

//--- Override detection
int g_prev_positions = 0;

//--- Backtest pre-computed signal table
struct PrecompRow {
    datetime bar_time;
    double   direction_bias;
    double   signal_strength;
    double   final_dir;
    bool     should_exit;
    double   hurst;
    double   tda_wasserstein;
    double   regime;
    double   actor_dir;
    double   actor_confidence;
    double   sl_pips;
    double   tp_pips;
    double   lot_suggestion;
};

PrecompRow g_precomp[];
int        g_precomp_cursor = 0;
bool       g_backtest_mode  = false;

//+------------------------------------------------------------------+
int OnInit()
{
    //--- Backtest mode: load pre-computed signals if CSV provided
    if((bool)MQLInfoInteger(MQL_TESTER) && StringLen(SIGNALS_CSV) > 0) {
        LoadPrecomp();
        g_backtest_mode = true;
        Print("Backtest mode: using pre-computed signals from ", SIGNALS_CSV);
        return INIT_SUCCEEDED;
    }

    //--- Live mode: open TCP socket to Rust server
    g_socket = SocketCreate();
    if(g_socket == INVALID_HANDLE) {
        Print("SocketCreate failed — check MT5 build (requires 2485+)");
        return INIT_SUCCEEDED; // continue in rule-only / display-only mode
    }

    if(SocketConnect(g_socket, SERVER_HOST, (uint)SERVER_PORT, 3000)) {
        g_connected = true;
        Print("Signal server connected: ", SERVER_HOST, ":", SERVER_PORT);
    } else {
        Print("WARNING: Cannot connect to signal server — rule-only mode");
    }
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    if(g_socket != INVALID_HANDLE) {
        SocketClose(g_socket);
        g_socket = INVALID_HANDLE;
    }
    ObjectsDeleteAll(0, "lbl_");
}

//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double   &open[],
                const double   &high[],
                const double   &low[],
                const double   &close[],
                const long     &tick_volume[],
                const long     &volume[],
                const int      &spread[])
{
    datetime current_bar = time[rates_total - 1];
    if(current_bar == g_last_bar) {
        if(SHOW_PANEL) DrawPanel();
        CheckOverride();
        return rates_total;
    }
    g_last_bar = current_bar;

    if(rates_total < BARS_HISTORY) return rates_total;

    //--- Backtest mode: look up pre-computed signal by bar time
    if(g_backtest_mode) {
        ApplyPrecomp(current_bar);
        if(SHOW_PANEL) DrawPanel();
        return rates_total;
    }

    //--- Live mode: build request and send to Rust server
    double event_risk = ReadEventRisk();
    double session_ph = SessionPhase();

    string bars_json = "[";
    int start = rates_total - BARS_HISTORY;
    for(int i = start; i < rates_total; i++) {
        if(i > start) bars_json += ",";
        bars_json += StringFormat("[%.5f,%.5f,%.5f,%.5f,%.0f,%.1f,%.4f,%.4f]",
            open[i], high[i], low[i], close[i],
            (double)tick_volume[i],
            (double)spread[i] * _Point * 10,
            session_ph, event_risk);
    }
    bars_json += "]";

    double pos_dir = 0, unrealized = 0, hold_frac = 0;
    GetPositionState(pos_dir, unrealized, hold_frac);

    string request = StringFormat(
        "{\"bars\":%s,\"pos_dir\":%.1f,\"unrealized\":%.5f,\"hold_fraction\":%.4f}\n",
        bars_json, pos_dir, unrealized, hold_frac);

    //--- Send to Rust server and parse response
    if(g_connected && g_socket != INVALID_HANDLE) {
        if(!SocketIsConnected(g_socket)) {
            g_connected = false;
            Print("Server disconnected — attempting reconnect");
            if(SocketConnect(g_socket, SERVER_HOST, (uint)SERVER_PORT, 2000))
                g_connected = true;
        }
        if(g_connected) {
            uchar send_buf[];
            int   send_len = StringToCharArray(request, send_buf, 0, StringLen(request));
            if(SocketSend(g_socket, send_buf, send_len) > 0) {
                string resp = SocketReadLine(g_socket, 5000);
                if(StringLen(resp) > 0)
                    ParseResponse(resp);
                else
                    Print("Server timeout — cached signal used");
            }
        }
    }

    if(SHOW_PANEL) DrawPanel();
    return rates_total;
}

//+------------------------------------------------------------------+
// Read one newline-terminated response from the server.
string SocketReadLine(int socket, uint timeout_ms)
{
    string result = "";
    uchar  buf[2048];
    bool   done = false;
    while(!done) {
        uint got = SocketRead(socket, buf, ArraySize(buf), timeout_ms);
        if(got == 0) break;
        for(uint i = 0; i < got; i++) {
            if((char)buf[i] == '\n') { done = true; break; }
            result += CharToString((char)buf[i]);
        }
        timeout_ms = 500; // short timeout for subsequent chunks
    }
    return result;
}

//+------------------------------------------------------------------+
void ParseResponse(string json)
{
    g_direction    = JsonGetDouble(json, "direction_bias");
    g_strength     = JsonGetDouble(json, "signal_strength");
    g_sl_pips      = JsonGetDouble(json, "sl_pips");
    g_tp_pips      = JsonGetDouble(json, "tp_pips");
    g_lot          = JsonGetDouble(json, "lot_suggestion");
    g_hurst        = JsonGetDouble(json, "hurst");
    g_tda_w        = JsonGetDouble(json, "tda_wasserstein");
    g_event_risk   = JsonGetDouble(json, "event_risk");
    g_regime       = JsonGetDouble(json, "regime");
    g_actor_dir    = JsonGetDouble(json, "actor_dir");
    g_final_dir    = JsonGetDouble(json, "final_dir");
    g_actor_conf   = JsonGetDouble(json, "actor_confidence");
    int pos = StringFind(json, "\"should_exit\":");
    if(pos >= 0) {
        pos += 14;
        g_should_exit = (StringSubstr(json, pos, 4) == "true");
    }
}

//+------------------------------------------------------------------+
double JsonGetDouble(const string json, const string key)
{
    string search = "\"" + key + "\":";
    int    pos    = StringFind(json, search);
    if(pos < 0) return 0.0;
    pos += StringLen(search);
    int end = pos;
    while(end < StringLen(json) &&
          StringGetCharacter(json, end) != ',' &&
          StringGetCharacter(json, end) != '}') end++;
    return StringToDouble(StringSubstr(json, pos, end - pos));
}

//+------------------------------------------------------------------+
// ── Backtest pre-computed signal helpers ─────────────────────────────
//+------------------------------------------------------------------+
void LoadPrecomp()
{
    // MT5 sandbox: FileOpen can only access files inside the terminal data folder
    // or the Common\Files folder (FILE_COMMON flag).
    // Pass only the filename in SIGNALS_CSV input (e.g. "signals.csv"),
    // the file must be in Common\Files (same folder ExportBars writes to).
    int fh = FileOpen(SIGNALS_CSV, FILE_READ | FILE_CSV | FILE_ANSI | FILE_COMMON, ',');
    if(fh == INVALID_HANDLE) {
        Print("Cannot open SIGNALS_CSV '", SIGNALS_CSV,
              "' from Common\\Files\\  err:", GetLastError(),
              " — place signals.csv in MT5 Common\\Files and set SIGNALS_CSV=signals.csv");
        return;
    }

    // Skip header row
    while(!FileIsLineEnding(fh) && !FileIsEnding(fh))
        FileReadString(fh);

    // Pre-allocate large block — avoids O(n^2) repeated realloc in the loop.
    // ArrayResize with reserve param keeps capacity without shrinking.
    int capacity = 200000;
    ArrayResize(g_precomp, capacity);
    int count = 0;

    while(!FileIsEnding(fh)) {
        string dt = FileReadString(fh);
        if(StringLen(dt) == 0) {
            while(!FileIsLineEnding(fh) && !FileIsEnding(fh))
                FileReadString(fh);
            continue;
        }
        double dir_bias = FileReadNumber(fh);
        double sig_str  = FileReadNumber(fh);
        double fin_dir  = FileReadNumber(fh);
        double sh_exit  = FileReadNumber(fh);
        double hurst    = FileReadNumber(fh);
        double tda_w    = FileReadNumber(fh);
        double regime   = FileReadNumber(fh);
        double act_dir  = FileReadNumber(fh);
        double act_conf = FileReadNumber(fh);
        double sl_p     = FileReadNumber(fh);
        double tp_p     = FileReadNumber(fh);
        double lot_s    = FileReadNumber(fh);

        if(count >= capacity) {
            capacity += 50000;
            ArrayResize(g_precomp, capacity);
        }
        g_precomp[count].bar_time         = StringToTime(dt);
        g_precomp[count].direction_bias   = dir_bias;
        g_precomp[count].signal_strength  = sig_str;
        g_precomp[count].final_dir        = fin_dir;
        g_precomp[count].should_exit      = sh_exit > 0.5;
        g_precomp[count].hurst            = hurst;
        g_precomp[count].tda_wasserstein  = tda_w;
        g_precomp[count].regime           = regime;
        g_precomp[count].actor_dir        = act_dir;
        g_precomp[count].actor_confidence = act_conf;
        g_precomp[count].sl_pips          = sl_p;
        g_precomp[count].tp_pips          = tp_p;
        g_precomp[count].lot_suggestion   = lot_s;
        count++;
    }
    FileClose(fh);
    ArrayResize(g_precomp, count); // trim to exact size
    g_precomp_cursor = 0;
    Print("Loaded ", count, " pre-computed signal rows");
}

void ApplyPrecomp(datetime current)
{
    int sz = ArraySize(g_precomp);
    // Advance cursor past any rows older than current bar
    while(g_precomp_cursor < sz - 1 &&
          g_precomp[g_precomp_cursor].bar_time < current)
        g_precomp_cursor++;

    if(g_precomp_cursor >= sz) return;
    if(g_precomp[g_precomp_cursor].bar_time != current) return;

    int c          = g_precomp_cursor;
    g_direction    = g_precomp[c].direction_bias;
    g_strength     = g_precomp[c].signal_strength;
    g_final_dir    = g_precomp[c].final_dir;
    g_should_exit  = g_precomp[c].should_exit;
    g_hurst        = g_precomp[c].hurst;
    g_tda_w        = g_precomp[c].tda_wasserstein;
    g_regime       = g_precomp[c].regime;
    g_actor_dir    = g_precomp[c].actor_dir;
    g_actor_conf   = g_precomp[c].actor_confidence;
    g_sl_pips      = g_precomp[c].sl_pips;
    g_tp_pips      = g_precomp[c].tp_pips;
    g_lot          = g_precomp[c].lot_suggestion;
}

//+------------------------------------------------------------------+
// ── Panel & logging (unchanged from v0.2) ────────────────────────────
//+------------------------------------------------------------------+
void DrawPanel()
{
    double display_dir = g_final_dir;
    string dir_str     = display_dir > 0.25  ? "BUY"  :
                         display_dir < -0.25 ? "SELL" : "HOLD";
    color  dir_col     = display_dir > 0.25  ? clrLimeGreen :
                         display_dir < -0.25 ? clrTomato    : clrGray;

    string regime_s = g_regime > 0.5 ? "Bull" : "Bear";
    string risk_s   = g_event_risk >= 1.0 ? "HIGH" :
                      g_event_risk >= 0.5 ? "MED"  : "LOW";
    string exit_s   = g_should_exit ? " [EXIT]" : "";

    int x = 15, y = 30;
    CreateLabel("lbl_dir",    x, y, dir_str + exit_s, dir_col, 14);        y += 24;
    CreateLabel("lbl_str",    x, y,
        StringFormat("Str: %.0f%%  Conf: %.0f%%",
                     g_strength * 100, g_actor_conf * 100),
        clrSilver, 9);  y += 16;
    CreateLabel("lbl_lot",    x, y,
        StringFormat("Lot: %.2f  SL: %.1f  TP: %.1f",
                     g_lot, g_sl_pips, g_tp_pips),
        clrSilver, 9);  y += 16;
    CreateLabel("lbl_regime", x, y,
        StringFormat("H=%.3f  TDA=%.3f  %s", g_hurst, g_tda_w, regime_s),
        clrDarkGray, 8);  y += 14;
    CreateLabel("lbl_risk",   x, y,
        StringFormat("Event: %s  Actor: %.2f", risk_s, g_actor_dir),
        g_event_risk >= 1.0 ? clrTomato : clrDarkGray, 8);
    ChartRedraw();
}

void CreateLabel(string name, int x, int y, string text, color col, int font_size)
{
    if(ObjectFind(0, name) < 0)
        ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
    ObjectSetInteger(0, name, OBJPROP_CORNER,    CORNER_LEFT_UPPER);
    ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
    ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
    ObjectSetString (0, name, OBJPROP_TEXT,      text);
    ObjectSetInteger(0, name, OBJPROP_COLOR,     col);
    ObjectSetInteger(0, name, OBJPROP_FONTSIZE,  font_size);
}

void CheckOverride()
{
    int current_pos = PositionsTotal();
    if(current_pos == g_prev_positions) return;

    if(current_pos > g_prev_positions) {
        for(int i = 0; i < current_pos; i++) {
            if(!PositionSelectByTicket(PositionGetTicket(i))) continue;
            double opened_lot = PositionGetDouble(POSITION_VOLUME);
            int    opened_dir = (PositionGetInteger(POSITION_TYPE) ==
                                 POSITION_TYPE_BUY) ? 1 : -1;
            bool   is_override = ((int)g_final_dir != opened_dir) ||
                                 (g_final_dir == 0);
            if(is_override)
                LogOverride(opened_dir, opened_lot, "open");
        }
    }
    g_prev_positions = current_pos;
}

void LogOverride(int user_dir, double user_lot, string event_type)
{
    int fh = FileOpen(LOG_PATH, FILE_READ | FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
    if(fh == INVALID_HANDLE) {
        fh = FileOpen(LOG_PATH, FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
        if(fh == INVALID_HANDLE) { Print("Cannot open log: ", LOG_PATH); return; }
        FileWrite(fh,
            "timestamp,symbol,rule_dir,final_dir,actor_dir,signal_strength,"
            "user_dir,user_lot,hurst,tda_w,event_risk,regime,event_type");
    }
    FileSeek(fh, 0, SEEK_END);
    FileWrite(fh,
        TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS),
        _Symbol,
        (int)g_direction, (int)g_final_dir, g_actor_dir, g_strength,
        user_dir, user_lot,
        g_hurst, g_tda_w, g_event_risk, g_regime,
        event_type);
    FileClose(fh);
}

double ReadEventRisk()
{
    int fh = FileOpen(CONTEXT_PATH, FILE_READ | FILE_TXT | FILE_ANSI);
    if(fh == INVALID_HANDLE) return 0.0;
    string content = "";
    while(!FileIsEnding(fh)) content += FileReadString(fh);
    FileClose(fh);
    return JsonGetDouble(content, "event_risk");
}

double SessionPhase()
{
    MqlDateTime dt;
    TimeToStruct(TimeGMT(), dt);
    if(dt.hour < 8)  return 0.0;
    if(dt.hour < 13) return 0.5;
    return 1.0;
}

void GetPositionState(double &dir, double &unrealized, double &hold_frac)
{
    dir = 0; unrealized = 0; hold_frac = 0;
    if(PositionsTotal() == 0) return;
    if(!PositionSelect(_Symbol)) return;
    dir        = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? 1.0 : -1.0;
    unrealized = PositionGetDouble(POSITION_PROFIT);
    datetime opened = (datetime)PositionGetInteger(POSITION_TIME);
    int bars_held   = (int)((TimeCurrent() - opened) / PeriodSeconds(PERIOD_M1));
    hold_frac       = MathMin((double)bars_held / 80.0, 1.0);
}

//+------------------------------------------------------------------+
//| XAUUSD Meta-Policy Indicator v0.2                                |
//| ZeroMQ client → Rust signal server                               |
//| Displays signal, sizing, SL/TP — no execution                    |
//| Logs user overrides to CSV for DSAC training                     |
//|                                                                  |
//| Dependencies:                                                    |
//|   ZMQ4MQL5 — https://github.com/dingmaotu/mql-zmq/releases       |
//|   Copy to: C:\Program Files\MetaTrader 5\MQL5\Include\ZMQ\       |
//|            C:\Program Files\MetaTrader 5\MQL5\Libraries\         |
//|            C:\Program Files\MetaTrader 5\  (zmq.dll runtime)     |
//+------------------------------------------------------------------+
#property indicator_chart_window
#property indicator_buffers 0
#property indicator_plots   0

#include <ZMQ\ZMQ.mqh>

//--- Parameters
input string SERVER_ADDR    = "tcp://127.0.0.1:5555";
input string LOG_PATH       = "C:\\Program Files\\MetaTrader 5\\override_log.csv";
input string CONTEXT_PATH   = "C:\\Program Files\\MetaTrader 5\\session_context.json";
input int    BARS_HISTORY   = 240;
input bool   SHOW_PANEL     = true;

//--- ZMQ state
Context g_ctx("xauusd_meta");
Socket  g_sock(g_ctx, ZMQ_REQ);
bool    g_connected = false;

//--- Signal cache
double g_direction    = 0;
double g_strength     = 0;
double g_sl_pips      = 0;
double g_tp_pips      = 0;
double g_lot          = 0.01;
double g_hurst        = 0.5;
double g_tda_w        = 0;
double g_event_risk   = 0;
double g_regime       = 0.5;
double g_actor_dir    = 0;
double g_final_dir    = 0;
bool   g_should_exit  = false;
double g_actor_conf   = 0;
datetime g_last_bar   = 0;

//--- Override detection
int    g_prev_positions = 0;

//+------------------------------------------------------------------+
int OnInit()
{
   g_sock.setReceiveTimeout(2000);
   if(g_sock.connect(SERVER_ADDR))
   {
      g_connected = true;
      Print("Signal server connected: ", SERVER_ADDR);
   }
   else
      Print("WARNING: Cannot connect to signal server — rule-only mode");
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   g_sock.disconnect(SERVER_ADDR);
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
   if(current_bar == g_last_bar)
   {
      if(SHOW_PANEL) DrawPanel();
      CheckOverride();
      return rates_total;
   }
   g_last_bar = current_bar;

   if(rates_total < BARS_HISTORY) return rates_total;

   double event_risk = ReadEventRisk();
   double session_ph = SessionPhase();

   //--- Build bar JSON array
   string bars_json = "[";
   int start = rates_total - BARS_HISTORY;
   for(int i = start; i < rates_total; i++)
   {
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
      "{\"bars\":%s,\"pos_dir\":%.1f,\"unrealized\":%.5f,\"hold_fraction\":%.4f}",
      bars_json, pos_dir, unrealized, hold_frac);

   //--- Send to Rust server
   if(g_connected)
   {
      ZmqMsg req_msg(request);
      if(g_sock.send(req_msg))
      {
         ZmqMsg resp_msg;
         if(g_sock.recv(resp_msg))
            ParseResponse(resp_msg.getData());
         else
            Print("ZMQ recv timeout — cached signal used");
      }
   }

   if(SHOW_PANEL) DrawPanel();
   return rates_total;
}

//+------------------------------------------------------------------+
void ParseResponse(string json)
{
   g_direction   = JsonGetDouble(json, "direction_bias");
   g_strength    = JsonGetDouble(json, "signal_strength");
   g_sl_pips     = JsonGetDouble(json, "sl_pips");
   g_tp_pips     = JsonGetDouble(json, "tp_pips");
   g_lot         = JsonGetDouble(json, "lot_suggestion");
   g_hurst       = JsonGetDouble(json, "hurst");
   g_tda_w       = JsonGetDouble(json, "tda_wasserstein");
   g_event_risk  = JsonGetDouble(json, "event_risk");
   g_regime      = JsonGetDouble(json, "regime");
   g_actor_dir   = JsonGetDouble(json, "actor_dir");
   g_final_dir   = JsonGetDouble(json, "final_dir");
   g_actor_conf  = JsonGetDouble(json, "actor_confidence");
   string exit_s = "";
   int pos = StringFind(json, "\"should_exit\":");
   if(pos >= 0)
   {
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
void DrawPanel()
{
   //--- Use final_dir (blended rule+actor) for display
   double display_dir = g_final_dir;
   string dir_str     = display_dir > 0.25  ? "BUY"  :
                        display_dir < -0.25 ? "SELL" : "HOLD";
   color  dir_col     = display_dir > 0.25  ? clrLimeGreen :
                        display_dir < -0.25 ? clrTomato    : clrGray;

   string regime_s    = g_regime > 0.5 ? "Bull" : "Bear";
   string risk_s      = g_event_risk >= 1.0 ? "HIGH" :
                        g_event_risk >= 0.5 ? "MED"  : "LOW";
   string exit_s      = g_should_exit ? " [EXIT]" : "";

   int x = 15, y = 30;

   CreateLabel("lbl_dir",   x, y, dir_str + exit_s, dir_col, 14);  y += 24;
   CreateLabel("lbl_str",   x, y,
      StringFormat("Str: %.0f%%  Conf: %.0f%%",
                   g_strength * 100, g_actor_conf * 100),
      clrSilver, 9);  y += 16;
   CreateLabel("lbl_lot",   x, y,
      StringFormat("Lot: %.2f  SL: %.1f  TP: %.1f",
                   g_lot, g_sl_pips, g_tp_pips),
      clrSilver, 9);  y += 16;
   CreateLabel("lbl_regime", x, y,
      StringFormat("H=%.3f  TDA=%.3f  %s", g_hurst, g_tda_w, regime_s),
      clrDarkGray, 8);  y += 14;
   CreateLabel("lbl_risk",  x, y,
      StringFormat("Event: %s  Actor: %.2f", risk_s, g_actor_dir),
      g_event_risk >= 1.0 ? clrTomato : clrDarkGray, 8);

   ChartRedraw();
}

//+------------------------------------------------------------------+
void CreateLabel(string name, int x, int y, string text,
                 color col, int font_size)
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

//+------------------------------------------------------------------+
void CheckOverride()
{
   int current_pos = PositionsTotal();
   if(current_pos == g_prev_positions) return;

   if(current_pos > g_prev_positions)
   {
      for(int i = 0; i < current_pos; i++)
      {
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

//+------------------------------------------------------------------+
void LogOverride(int user_dir, double user_lot, string event_type)
{
   int fh = FileOpen(LOG_PATH,
                     FILE_READ | FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
   if(fh == INVALID_HANDLE)
   {
      fh = FileOpen(LOG_PATH,
                    FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
      if(fh == INVALID_HANDLE)
      {
         Print("Cannot open log: ", LOG_PATH);
         return;
      }
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

//+------------------------------------------------------------------+
double ReadEventRisk()
{
   int fh = FileOpen(CONTEXT_PATH,
                     FILE_READ | FILE_TXT | FILE_ANSI);
   if(fh == INVALID_HANDLE) return 0.0;
   string content = "";
   while(!FileIsEnding(fh)) content += FileReadString(fh);
   FileClose(fh);
   return JsonGetDouble(content, "event_risk");
}

//+------------------------------------------------------------------+
double SessionPhase()
{
   MqlDateTime dt;
   TimeToStruct(TimeGMT(), dt);
   if(dt.hour < 8)  return 0.0;
   if(dt.hour < 13) return 0.5;
   return 1.0;
}

//+------------------------------------------------------------------+
void GetPositionState(double &dir, double &unrealized, double &hold_frac)
{
   dir = 0; unrealized = 0; hold_frac = 0;
   if(PositionsTotal() == 0) return;
   if(!PositionSelect(_Symbol)) return;
   dir        = (PositionGetInteger(POSITION_TYPE) ==
                 POSITION_TYPE_BUY) ? 1.0 : -1.0;
   unrealized = PositionGetDouble(POSITION_PROFIT);
   datetime opened  = (datetime)PositionGetInteger(POSITION_TIME);
   int bars_held    = (int)((TimeCurrent() - opened) /
                             PeriodSeconds(PERIOD_M1));
   hold_frac        = MathMin((double)bars_held / 80.0, 1.0);
}
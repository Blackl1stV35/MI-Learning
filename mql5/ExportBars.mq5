//+------------------------------------------------------------------+
//| ExportBars.mq5                                                   |
//| Script: export XAUUSD M1 OHLCV to CSV for Rust replay mode.     |
//|                                                                  |
//| Output columns: datetime,open,high,low,close,tick_volume         |
//| datetime format: "2026.01.01 00:00"  (TIME_DATE|TIME_MINUTES)    |
//|                                                                  |
//| Run in MetaEditor → Scripts → Compile then drag onto chart.     |
//| Output file is written to the MT5 common Files folder.           |
//+------------------------------------------------------------------+
#property script_show_inputs

input string        SYMBOL    = "XAUUSD";
input ENUM_TIMEFRAMES TF      = PERIOD_M1;
input datetime      FROM_DATE = D'2026.01.01';
input datetime      TO_DATE   = D'2026.06.30';
input string        OUT_FILE  = "XAUUSD_M1_bars.csv";

void OnStart()
{
    MqlRates rates[];
    ArraySetAsSeries(rates, false);

    int count = CopyRates(SYMBOL, TF, FROM_DATE, TO_DATE, rates);
    if(count <= 0) {
        Print("CopyRates failed — err:", GetLastError(),
              "  Ensure the symbol has history loaded in MT5.");
        return;
    }

    // Write to MT5 common Files folder (accessible across terminals)
    int fh = FileOpen(OUT_FILE, FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_COMMON, ',');
    if(fh == INVALID_HANDLE) {
        Print("FileOpen failed — err:", GetLastError(), "  path: ", OUT_FILE);
        return;
    }

    FileWrite(fh, "datetime", "open", "high", "low", "close", "tick_volume");

    for(int i = 0; i < count; i++) {
        FileWrite(fh,
            TimeToString(rates[i].time, TIME_DATE | TIME_MINUTES),
            DoubleToString(rates[i].open,  _Digits),
            DoubleToString(rates[i].high,  _Digits),
            DoubleToString(rates[i].low,   _Digits),
            DoubleToString(rates[i].close, _Digits),
            (long)rates[i].tick_volume);
    }
    FileClose(fh);

    Print("Exported ", count, " M1 bars  →  Common\\Files\\", OUT_FILE);
    Print("Next: signal_server replay --bars <path\\", OUT_FILE, "> --out signals.csv [--actor models\\actor_weights.json]");
}

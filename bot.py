# bot.py — покращений сигнал-бот з бектестом (Українська, Купити/Продати)
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import asyncio

# ----------------- Налаштування -----------------
TELEGRAM_TOKEN = "ТУТ_ТВІЙ_ТОКЕН"   # <- встав свій токен
ASSETS_FILE = "assets.json"          # список пар (без payout)
MIN_STRENGTH = 86                    # початковий поріг (підлаштувати через бектест)
EXPIRY_MIN = 3                       # інформаційна експірація
# ------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Простий список пар (заміни/редагуй assets.json файлом)
def ensure_assets():
    p = Path(ASSETS_FILE)
    if not p.exists():
        example = [
            {"symbol":"EURUSD=X","display":"EUR/USD"},
            {"symbol":"GBPUSD=X","display":"GBP/USD"},
            {"symbol":"USDJPY=X","display":"USD/JPY"},
            {"symbol":"BTC-USD","display":"BTC/USD"},
            {"symbol":"ETH-USD","display":"ETH/USD"}
        ]
        p.write_text(json.dumps(example, ensure_ascii=False, indent=2))
    return json.loads(Path(ASSETS_FILE).read_text(encoding='utf-8'))

# ---------- Indicators ----------
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def macd_hist(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - macd_signal

def atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ---------- Fetch data (with simple caching) ----------
_cache = {}
def fetch_ohlcv(symbol, interval='5m', period='3d'):
    key = f"{symbol}_{interval}_{period}"
    now = datetime.utcnow()
    cached = _cache.get(key)
    if cached and (now - cached['ts']).seconds < 60:
        return cached['df']
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            return None
        df = df.dropna()
        _cache[key] = {'ts': now, 'df': df}
        return df
    except Exception as e:
        logger.exception("fetch failed")
        return None

# ---------- Analysis ----------
def analyze_symbol(symbol, use_15m=True):
    # 5m base
    df5 = fetch_ohlcv(symbol, interval='5m', period='7d')
    if df5 is None or len(df5) < 80:
        return None
    close5 = df5['Close']
    ema50_5 = ema(close5,50)
    ema200_5 = ema(close5,200)
    last_close = float(close5.iloc[-1])
    # basic check
    if np.isnan(ema200_5.iloc[-1]):
        return None
    # trend
    trend = "BUY" if ema50_5.iloc[-1] > ema200_5.iloc[-1] else "SELL"
    # indicators
    last_rsi = float(rsi(close5,14).iloc[-1])
    last_macd = float(macd_hist(close5).iloc[-1])
    last_atr = float(atr(df5,14).iloc[-1]) if not np.isnan(atr(df5,14).iloc[-1]) else 0.0
    # spike filter
    recent = df5[-6:]
    for i in range(len(recent)):
        rng = recent['High'].iloc[i] - recent['Low'].iloc[i]
        if rng > max(last_atr*3.0, last_close*0.007):
            return None
    # SR
    window = 120
    support = float(df5['Low'][-window:].min())
    resistance = float(df5['High'][-window:].max())
    near_support = abs(last_close - support) <= max(last_atr*1.2, last_close*0.0015)
    near_resistance = abs(last_close - resistance) <= max(last_atr*1.2, last_close*0.0015)
    # scoring (5 checks *20)
    score = 0
    reasons = []
    score += 20; reasons.append("Trend")
    if trend=="BUY" and last_rsi < 55:
        score += 20; reasons.append(f"RSI{int(last_rsi)}")
    if trend=="SELL" and last_rsi > 45:
        score += 20; reasons.append(f"RSI{int(last_rsi)}")
    if trend=="BUY" and last_macd > 0:
        score += 20; reasons.append("MACD+")
    if trend=="SELL" and last_macd < 0:
        score += 20; reasons.append("MACD-")
    if (trend=="BUY" and near_support) or (trend=="SELL" and near_resistance):
        score += 20; reasons.append("NearSR")
    # volatility requirement
    vol_ok = last_atr >= (last_close * 0.00015)
    if not vol_ok:
        return None
    strength = min(100,int(score))
    # 15m confirmation
    if use_15m:
        df15 = fetch_ohlcv(symbol, interval='15m', period='14d')
        if df15 is None or len(df15)<60:
            return None
        close15 = df15['Close']
        ema50_15 = ema(close15,50)
        ema200_15 = ema(close15,200)
        if np.isnan(ema200_15.iloc[-1]):
            return None
        trend15 = "BUY" if ema50_15.iloc[-1] > ema200_15.iloc[-1] else "SELL"
        if trend15 != trend:
            return None
    return {
        "symbol": symbol,
        "trend": trend,
        "price": round(last_close,6),
        "strength": strength,
        "reasons": reasons,
        "support": round(support,6),
        "resistance": round(resistance,6)
    }

# ---------- Backtest simple: simulate expiry after N minutes ----------
def backtest(symbol, days=7, expiry_min=3, min_strength=MIN_STRENGTH, use_15m=True):
    # We'll use 1m data if possible for simulation accuracy, fallback to 5m
    df1 = fetch_ohlcv(symbol, interval='1m', period=f'{days+2}d')
    if df1 is None or len(df1) < 60:
        df5 = fetch_ohlcv(symbol, interval='5m', period=f'{days+10}d')
        if df5 is None:
            return {"error":"no data for backtest"}
        # approximate expiry using next candle: treat 5m as single candle (less accurate)
        dataframe = df5
        use_1m = False
    else:
        dataframe = df1
        use_1m = True

    # generate signals across history: sliding window approach
    signals = []
    # we will step over time and apply same logic as analyze_symbol at each 5m boundary
    # for speed, downsample 1m->5m if necessary
    if use_1m:
        base = dataframe.resample('5T').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
    else:
        base = dataframe.copy()

    # iterate over base candles skipping initial warmup
    for i in range(200, len(base)-5):
        window = base.iloc[:i+1]
        # recreate minimal df for analyze_symbol: take last 7d worth from window
        recent = window[-200:]  # approx
        # we need to compute indicators locally
        try:
            close = recent['Close']
            ema50 = ema(close,50)
            ema200 = ema(close,200)
            if np.isnan(ema200.iloc[-1]): 
                continue
            trend = "BUY" if ema50.iloc[-1] > ema200.iloc[-1] else "SELL"
            last_close = float(close.iloc[-1])
            last_rsi = float(rsi(close,14).iloc[-1])
            last_macd = float(macd_hist(close).iloc[-1])
            last_atr = float(atr(recent,14).iloc[-1])
            # basic spike/vol checks
            if last_atr < (last_close * 0.00012): 
                continue
            # score
            score = 20
            if trend=="BUY" and last_rsi<55: score+=20
            if trend=="SELL" and last_rsi>45: score+=20
            if trend=="BUY" and last_macd>0: score+=20
            if trend=="SELL" and last_macd<0: score+=20
            # near SR
            support = float(recent['Low'][-120:].min())
            resistance = float(recent['High'][-120:].max())
            near_support = abs(last_close-support) <= max(last_atr*1.2, last_close*0.0015)
            near_res = abs(last_close-resistance) <= max(last_atr*1.2, last_close*0.0015)
            if (trend=="BUY" and near_support) or (trend=="SELL" and near_res): score+=20
            strength = int(min(100,score))
            if strength < min_strength:
                continue
            # simulate result: check price after expiry_min minutes
            # get index of current candle in original (1m) dataframe to find future price
            # find the timestamp to check
            ts = recent.index[-1]
            target_ts = ts + pd.Timedelta(minutes=expiry_min)
            # find nearest later close in dataframe
            future_idx = dataframe.index.get_indexer([target_ts], method='nearest')[0]
            future_price = float(dataframe['Close'].iloc[future_idx])
            result = (future_price > last_close) if trend=="BUY" else (future_price < last_close)
            signals.append({"time":ts,"trend":trend,"strength":strength,"entry":last_close,"exit":future_price,"win":bool(result)})
        except Exception:
            continue

    if not signals:
        return {"error":"no signals in backtest"}
    wins = sum(1 for s in signals if s['win'])
    total = len(signals)
    winrate = wins/total*100
    return {"symbol":symbol,"days":days,"total":total,"wins":wins,"winrate":round(winrate,2)}

# ---------- Telegram handlers ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привіт! Я сигнал-бот. /signal щоб отримати сигнал, /backtest SYMBOL DAYS щоб перевірити результат на історії. /set_threshold N — змінити поріг.")

async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat.id
    await context.bot.send_message(chat, "🔎 Аналізую (5m + підтвердження 15m)...")
    assets = ensure_assets()
    results = []
    for a in assets:
        res = analyze_symbol(a['symbol'], use_15m=True)
        if res and res['strength'] >= MIN_STRENGTH:
            res['display'] = a.get('display', a['symbol'])
            results.append(res)
    if not results:
        await context.bot.send_message(chat, "❌ Сильних сигналів немає. Спробуй пізніше.")
        return
    # відправити найсильніші
    results = sorted(results, key=lambda x: x['strength'], reverse=True)[:5]
    out = []
    for r in results:
        verb = "Купити" if r['trend']=="BUY" else "Продати"
        out.append(
            f"📌 {r['display']} ({r['symbol']})\n"
            f"🔔 {verb}  |  Потужність: {r['strength']}%\n"
            f"💵 Ціна: {r['price']}\n"
            f"📈 Підтвердження: {', '.join(r['reasons'])}\n"
            f"🛑 S: {r['support']}  ▶ R: {r['resistance']}\n"
            f"⏱ Expiry (info): {EXPIRY_MIN} хв\n"
        )
    await context.bot.send_message(chat, "\n\n".join(out))

async def cmd_backtest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat.id
    args = context.args
    if not args or len(args) < 2:
        await context.bot.send_message(chat, "Використання: /backtest SYMBOL DAYS (наприклад: /backtest EURUSD=X 7)")
        return
    symbol = args[0].strip()
    try:
        days = int(args[1])
    except:
        days = 7
    await context.bot.send_message(chat, f"🔁 Запускаю бектест {symbol} за останні {days} днів (може зайняти хвилину)...")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, backtest, symbol, days, EXPIRY_MIN, MIN_STRENGTH, True)
    if "error" in result:
        await context.bot.send_message(chat, f"❌ Backtest error: {result.get('error')}")
        return
    await context.bot.send_message(chat, f"🧾 Backtest {result['symbol']} за {result['days']} днів\nУгод: {result['total']}\nВиграші: {result['wins']}\nWinrate: {result['winrate']}%")

async def cmd_set_threshold(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global MIN_STRENGTH
    chat = update.effective_chat.id
    args = context.args
    if not args:
        await context.bot.send_message(chat, f"Поточний поріг: {MIN_STRENGTH}%")
        return
    try:
        v = int(args[0])
        MIN_STRENGTH = v
        await context.bot.send_message(chat, f"Поріг оновлено: {MIN_STRENGTH}%")
    except:
        await context.bot.send_message(chat, "Помилка: вкажи ціле число порога, наприклад /set_threshold 85")

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signal", cmd_signal))
    app.add_handler(CommandHandler("backtest", cmd_backtest))
    app.add_handler(CommandHandler("set_threshold", cmd_set_threshold))
    logger.info("Bot starting...")
    app.run_polling()

if __name__ == "__main__":
    main()

# bot.py
import os
import time
import math
import requests
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from telegram.ext import Updater, CommandHandler

# ========== CONFIG ==========
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "YOUR_TELEGRAM_TOKEN")
TD_API_KEY = os.environ.get("TWELVE_KEY", "YOUR_TWELVE_KEY")  # Twelve Data or swap to other provider
CANDLES_LIMIT = 200  # how many candles to fetch
# ============================

# ---- helper: fetch 5m candles from Twelve Data (or change provider) ----
def fetch_candles_twelvedata(symbol: str, interval: str = "5min", outputsize: int = CANDLES_LIMIT):
    # symbol: "EUR/USD" for Twelve Data; we'll try common formats
    # Twelve Data endpoint example: https://api.twelvedata.com/time_series?symbol=EUR/USD&interval=5min&outputsize=200&apikey=KEY
    base = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "format": "JSON",
        "apikey": TD_API_KEY
    }
    r = requests.get(base, params=params, timeout=10).json()
    if "values" not in r:
        raise ValueError("No data: " + str(r))
    df = pd.DataFrame(r["values"])
    # TwelveData returns newest first — reverse
    df = df[::-1].reset_index(drop=True)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

# ---- indicators & SR detection ----
def compute_indicators(df):
    df = df.copy()
    df['ema10'] = EMAIndicator(df['close'], window=10).ema_indicator()
    df['ema20'] = EMAIndicator(df['close'], window=20).ema_indicator()
    df['ema50'] = EMAIndicator(df['close'], window=50).ema_indicator()
    df['rsi14'] = RSIIndicator(df['close'], window=14).rsi()
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_hi'] = bb.bollinger_hband()
    df['bb_lo'] = bb.bollinger_lband()
    return df

def local_extrema_levels(df, k=3):
    highs = []
    lows = []
    for i in range(k, len(df)-k):
        window = df['high'].iloc[i-k:i+k+1]
        if df['high'].iloc[i] == window.max():
            highs.append((i, df['high'].iloc[i]))
        window2 = df['low'].iloc[i-k:i+k+1]
        if df['low'].iloc[i] == window2.min():
            lows.append((i, df['low'].iloc[i]))
    # take values only
    high_vals = sorted(list({round(v,6) for _,v in highs}))
    low_vals  = sorted(list({round(v,6) for _,v in lows}))
    return low_vals, high_vals  # support list, resistance list

# ---- signal logic ----
def generate_signal(df):
    # expects df computed
    if len(df) < 60:
        return {"signal":"NO_DATA"}
    a = df.iloc[-2]
    b = df.iloc[-1]

    # Trend filter
    if b['ema20'] is None or b['ema50'] is None:
        return {"signal":"NO_DATA"}

    if b['ema20'] > b['ema50']:
        trend = "BUY"
    elif b['ema20'] < b['ema50']:
        trend = "SELL"
    else:
        trend = "NEUTRAL"

    # EMA crossover short-term
    cross_up = (a['ema10'] < a['ema50']) and (b['ema10'] > b['ema50'])
    cross_down = (a['ema10'] > a['ema50']) and (b['ema10'] < b['ema50'])

    rsi = b['rsi14']

    # Bollinger touch
    touched_lower = (b['low'] <= b['bb_lo'])
    touched_upper = (b['high'] >= b['bb_hi'])

    # Compute local SR
    supports, resistances = local_extrema_levels(df, k=3)
    # nearest levels:
    price = b['close']
    nearest_support = min(supports, key=lambda x: abs(x-price)) if supports else None
    nearest_resist  = min(resistances, key=lambda x: abs(x-price)) if resistances else None

    reasons = []

    # BUY condition
    if trend == "BUY" and cross_up and (rsi < 65) and (touched_lower or (nearest_support and abs(price-nearest_support) <= abs(price)*0.0035)):
        reasons.append("EMA20>EMA50")
        reasons.append("EMA10↗EMA50")
        reasons.append(f"RSI={round(rsi)}")
        if touched_lower:
            reasons.append("Touched lower Bollinger")
        if nearest_support:
            reasons.append("Near support")
        return {"signal":"BUY", "price":price, "stop": nearest_support, "resist": nearest_resist, "reasons": reasons}

    # SELL condition
    if trend == "SELL" and cross_down and (rsi > 35) and (touched_upper or (nearest_resist and abs(price-nearest_resist) <= abs(price)*0.0035)):
        reasons.append("EMA20<EMA50")
        reasons.append("EMA10↘EMA50")
        reasons.append(f"RSI={round(rsi)}")
        if touched_upper:
            reasons.append("Touched upper Bollinger")
        if nearest_resist:
            reasons.append("Near resistance")
        return {"signal":"SELL", "price":price, "stop": nearest_resist, "support": nearest_support, "reasons": reasons}

    return {"signal":"NO_TRADE", "price":price, "support": nearest_support, "resist": nearest_resist, "reasons": reasons}

# ---- formatting message ----
def format_message(symbol, interval, result):
    if result.get("signal") in ["BUY","SELL"]:
        sig = result["signal"]
        emoji = "📈" if sig=="BUY" else "📉"
        reasons = " | ".join(result.get("reasons",[]))
        price = result.get("price")
        stop = result.get("stop") or result.get("support")
        resist = result.get("resist") or result.get("resistance")
        # craft message
        msg = (
            f"{emoji} *{symbol}*  ({interval})\n"
            f"🔔 *Сигнал:* {sig}\n"
            f"⏱ *Експірація:* 5 хв\n"
            f"📈 *Причини:* {reasons}\n"
            f"📍 *Поточна ціна:* {price}\n"
        )
        if stop:
            msg += f"🛑 *Stop (S/R):* {stop}\n"
        if resist:
            msg += f"🎯 *R/S:* {resist}\n"
        msg += "\n⚠️ _Порада:_ тестуй сигнал на демо. Нема гарантій.\n"
        return msg
    else:
        supp = result.get("support")
        resi = result.get("resist")
        msg = f"⚠️ *{symbol}* — Немає якісного входу ({result.get('signal')}).\n"
        if supp or resi:
            msg += f"📌 Ближчі рівні S/R: S={supp}  R={resi}\n"
        msg += "_Спробуйте пізніше або змініть інструмент._"
        return msg

# ---- Telegram handler ----
def handle_signal(update, context):
    text = update.message.text.strip().lower()
    # allow commands like /eurusd or /btc/usd or /btc
    symbol_raw = text.replace("/", "").replace(" ", "").replace("/", "").replace("@","").lstrip("/")
    # map simple short names to TwelveData symbols
    mapping = {
        "eurusd":"EUR/USD",
        "gbpusd":"GBP/USD",
        "audusd":"AUD/USD",
        "usdjpy":"USD/JPY",
        "btc":"BTC/USD",
        "btcusd":"BTC/USD",
        "eth":"ETH/USD",
        "ethusd":"ETH/USD"
    }
    symbol = mapping.get(symbol_raw, symbol_raw.upper())

    chat = update.message.chat_id
    update.message.reply_text("⏳ Аналізую — зачекай кілька секунд...")
    try:
        df5 = fetch_candles_twelvedata(symbol, interval="5min", outputsize=CANDLES_LIMIT)
    except Exception as e:
        update.message.reply_text(f"❗ Помилка при отриманні даних: {e}")
        return

    df5 = compute_indicators(df5)
    result = generate_signal(df5)
    msg = format_message(symbol, "5m", result)
    # send styled message
    update.message.reply_text(msg, parse_mode="Markdown")

# ---- main ----
def main():
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher
    # add handlers for common symbols or fallback handler
    commands = ['eurusd','gbpusd','audusd','usdjpy','btc','btcusd','eth','ethusd']
    for c in commands:
        dp.add_handler(CommandHandler(c, handle_signal))
    # generic (any /something) - optional
    dp.add_handler(CommandHandler('start', lambda u,c: u.message.reply_text("Відправ команду /eurusd або /btc")))
    updater.start_polling()
    print("Bot started")
    updater.idle()

if __name__ == "__main__":
    main()

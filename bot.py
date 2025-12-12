import telebot
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time

TOKEN = "ВСТАВ СВІЙ ТЕЛЕГРАМ ТОКЕН ТУТ"

bot = telebot.TeleBot(TOKEN)

# --- індикатори ---
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

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal).mean()
    return macd_line - macd_signal

# --- список активів з payout >= 85% ---
ASSETS = {
    "EURUSD=X": {"name": "EUR/USD", "payout": 0.90},
    "GBPUSD=X": {"name": "GBP/USD", "payout": 0.88},
    "USDJPY=X": {"name": "USD/JPY", "payout": 0.86},
    "BTC-USD": {"name": "BTC/USD", "payout": 0.92},
    "ETH-USD": {"name": "ETH/USD", "payout": 0.91}
}

def analyze(symbol):
    try:
        df = yf.download(symbol, period="2d", interval="5m", progress=False)

        if df is None or df.empty:
            return None

        close = df["Close"]

        ema20 = ema(close, 20)
        ema50 = ema(close, 50)
        rsi14 = rsi(close, 14)
        macd_hist = macd(close)

        last = close.iloc[-1]
        last_ema20 = ema20.iloc[-1]
        last_ema50 = ema50.iloc[-1]
        last_rsi = rsi14.iloc[-1]
        last_macd = macd_hist.iloc[-1]

        score = 0
        reasons = []

        # Тренд
        if last_ema20 > last_ema50:
            trend = "BUY"
        else:
            trend = "SELL"

        # Підтвердження 1: RSI
        if trend == "BUY" and last_rsi < 60:
            score += 1
            reasons.append("RSI")
        if trend == "SELL" and last_rsi > 40:
            score += 1
            reasons.append("RSI")

        # Підтвердження 2: MACD
        if trend == "BUY" and last_macd > 0:
            score += 1
            reasons.append("MACD")
        if trend == "SELL" and last_macd < 0:
            score += 1
            reasons.append("MACD")

        # Підтвердження 3: EMA кут
        if abs(last_ema20 - last_ema50) > 0:
            score += 1
            reasons.append("EMA")

        if score >= 3:
            return {
                "symbol": symbol,
                "trend": trend,
                "price": round(float(last), 6),
                "score": score,
                "reasons": ", ".join(reasons)
            }
        return None
    except:
        return None

@bot.message_handler(commands=["signal"])
def signal(message):
    bot.send_message(message.chat.id, "🔍 Аналізую ринок з payout ≥ 85%...\nЗачекай 3–5 секунд 🔎")

    results = []

    for symbol in ASSETS:
        payout = ASSETS[symbol]["payout"]

        if payout < 0.85:
            continue

        r = analyze(symbol)
        if r:
            r["name"] = ASSETS[symbol]["name"]
            r["payout"] = payout
            results.append(r)

        time.sleep(1)

    if not results:
        bot.send_message(message.chat.id, "❌ Сильних сигналів немає.\nСпробуй пізніше.")
        return

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    msg = ""
    for r in results:
        msg += (
            f"📌 {r['name']}\n"
            f"🔔 СИГНАЛ: {r['trend']}\n"
            f"💰 Виплата: {int(r['payout']*100)}%\n"
            f"💵 Ціна: {r['price']}\n"
            f"⭐ Потужність сигналу: {r['score']}/3\n"
            f"📈 Підтвердження: {r['reasons']}\n"
            f"⏱ Expiry: 5 хв\n\n"
        )

    bot.send_message(message.chat.id, msg)

bot.polling(none_stop=True)

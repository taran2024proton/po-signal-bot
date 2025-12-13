# ===============================================================
# bot.py — FINAL STABLE (RENDER + TELEGRAM + YFINANCE FIXED)
# ===============================================================

import json
import time
from pathlib import Path
from datetime import datetime, timedelta, UTC

import yfinance as yf
import pandas as pd
import telebot
from flask import Flask, request

# ---------------- CONFIG ----------------
TOKEN = "8517986396:AAENPrASLsQlLu21BxG-jKIYZEaEL-RKxYs"
WEBHOOK_URL = "https://po-signal-bot-gwu0.onrender.com/webhook"

ASSETS_FILE = "assets.json"
CACHE_FILE = "cache.json"
CACHE_SECONDS = 120

PAYOUT_MIN = 0.85
EXPIRY_MIN = 3
MAX_ASSETS = 5

MODE = "conservative"
THRESHOLDS = {
    "conservative": {"MIN_STRENGTH": 80, "USE_15M": True},
    "aggressive": {"MIN_STRENGTH": 70, "USE_15M": False},
}

bot = telebot.TeleBot(TOKEN, parse_mode="HTML", threaded=False)
app = Flask(__name__)

# ---------------- CACHE ----------------
def load_cache():
    try:
        return json.loads(Path(CACHE_FILE).read_text())
    except:
        return {}

def save_cache(c):
    try:
        Path(CACHE_FILE).write_text(json.dumps(c))
    except:
        pass

cache = load_cache()

def cache_get(key):
    if key not in cache:
        return None
    ts = datetime.fromisoformat(cache[key]["ts"])
    if datetime.now(UTC) - ts > timedelta(seconds=CACHE_SECONDS):
        return None
    return cache[key]["data"]

def cache_set(key, data):
    cache[key] = {"ts": datetime.now(UTC).isoformat(), "data": data}
    save_cache(cache)

# ---------------- INDICATORS ----------------
def ema_last(series, period):
    return series.ewm(span=period, adjust=False).mean().iloc[-1]

def rsi_last(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return (100 - (100 / (1 + rs))).iloc[-1]

def macd_hist_last(series):
    fast = series.ewm(span=12, adjust=False).mean()
    slow = series.ewm(span=26, adjust=False).mean()
    macd = fast - slow
    signal = macd.ewm(span=9, adjust=False).mean()
    return (macd - signal).iloc[-1]

def atr_last(df, period=14):
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]

# ---------------- ASSETS ----------------
def get_assets():
    try:
        return json.loads(Path(ASSETS_FILE).read_text())
    except:
        assets = [
            {"symbol":"GBPJPY=X","display":"GBP/JPY","payout":0.87},
            {"symbol":"AUDCAD=X","display":"AUD/CAD","payout":0.86},
            {"symbol":"AUDCHF=X","display":"AUD/CHF","payout":0.86},
            {"symbol":"AUDJPY=X","display":"AUD/JPY","payout":0.87},
            {"symbol":"AUDUSD=X","display":"AUD/USD","payout":0.88},
            {"symbol":"CADCHF=X","display":"CAD/CHF","payout":0.85},
            {"symbol":"CADJPY=X","display":"CAD/JPY","payout":0.86},
            {"symbol":"CHFJPY=X","display":"CHF/JPY","payout":0.86},
            {"symbol":"EURAUD=X","display":"EUR/AUD","payout":0.87},
            {"symbol":"EURCAD=X","display":"EUR/CAD","payout":0.87},
            {"symbol":"EURCHF=X","display":"EUR/CHF","payout":0.88},
            {"symbol":"EURGBP=X","display":"EUR/GBP","payout":0.89},
            {"symbol":"EURUSD=X","display":"EUR/USD","payout":0.90},
            {"symbol":"EURJPY=X","display":"EUR/JPY","payout":0.88},
            {"symbol":"GBPAUD=X","display":"GBP/AUD","payout":0.87},
            {"symbol":"GBPCHF=X","display":"GBP/CHF","payout":0.87},
            {"symbol":"GBPUSD=X","display":"GBP/USD","payout":0.89},
            {"symbol":"GBPCAD=X","display":"GBP/CAD","payout":0.86},
            {"symbol":"USDCAD=X","display":"USD/CAD","payout":0.88},
            {"symbol":"USDCHF=X","display":"USD/CHF","payout":0.88},
            {"symbol":"USDJPY=X","display":"USD/JPY","payout":0.89},
            {"symbol":"BTC-USD","display":"Bitcoin","payout":0.92},
            {"symbol":"ETH-USD","display":"Ethereum","payout":0.91},
            {"symbol":"DASH-USD","display":"Dash","payout":0.90},
            {"symbol":"BCH-EUR","display":"BCH/EUR","payout":0.89},
            {"symbol":"BCH-GBP","display":"BCH/GBP","payout":0.89},
            {"symbol":"BCH-JPY","display":"BCH/JPY","payout":0.89},
            {"symbol":"BTC-GBP","display":"BTC/GBP","payout":0.90},
            {"symbol":"BTC-JPY","display":"BTC/JPY","payout":0.90},
            {"symbol":"LINK-USD","display":"Chainlink","payout":0.90},
            {"symbol":"BZ=F","display":"Brent Oil","payout":0.88},
            {"symbol":"CL=F","display":"WTI Crude Oil","payout":0.88},
            {"symbol":"SI=F","display":"Silver","payout":0.87},
            {"symbol":"GC=F","display":"Gold","payout":0.89},
            {"symbol":"NG=F","display":"Natural Gas","payout":0.85},
            {"symbol":"PA=F","display":"Palladium spot","payout":0.86},
            {"symbol":"PL=F","display":"Platinum spot","payout":0.86},
            {"symbol":"AAPL","display":"Apple","payout":0.88},
            {"symbol":"AXP","display":"American Express","payout":0.87},
            {"symbol":"BA","display":"Boeing Company","payout":0.86},
            {"symbol":"META","display":"Facebook (Meta)","payout":0.88},
            {"symbol":"JNJ","display":"Johnson & Johnson","payout":0.87},
            {"symbol":"JPM","display":"JPMorgan Chase","payout":0.88},
            {"symbol":"MCD","display":"McDonald's","payout":0.87},
            {"symbol":"MSFT","display":"Microsoft","payout":0.89},
            {"symbol":"PFE","display":"Pfizer","payout":0.86},
            {"symbol":"TSLA","display":"Tesla","payout":0.90},
            {"symbol":"BABA","display":"Alibaba","payout":0.87},
            {"symbol":"C","display":"Citigroup","payout":0.86},
            {"symbol":"NFLX","display":"Netflix","payout":0.88},
            {"symbol":"CSCO","display":"Cisco","payout":0.86},
            {"symbol":"XOM","display":"ExxonMobil","payout":0.87},
            {"symbol":"INTC","display":"Intel","payout":0.86}
        ]
        Path(ASSETS_FILE).write_text(json.dumps(assets, indent=2))
        return assets

# ---------------- DATA (🔥 FIXED) ----------------
def fetch(symbol, interval):
    key = f"{symbol}_{interval}"
    cached = cache_get(key)
    if cached:
        return pd.read_json(cached)

    df = yf.download(
        symbol,
        period="3d",
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if df is None or df.empty:
        return None

    # 🔥 FIX MULTIINDEX COLUMNS
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(df.columns):
        print(f"⚠️ Missing OHLC for {symbol} {interval}")
        return None

    df = df.reset_index()
    cache_set(key, df.to_json(date_format="iso"))
    return df

# ---------------- ANALYSIS ----------------
def analyze(symbol, use_15m):
    df5 = fetch(symbol, "5m")
    if df5 is None or len(df5) < 200:
        return None

    close = df5["Close"]
    price = float(close.iloc[-1])

    ema50 = ema_last(close, 50)
    ema200 = ema_last(close, 200)
    trend = "BUY" if ema50 > ema200 else "SELL"

    rsi = rsi_last(close, 5)
    macd = macd_hist_last(close)
    atr = atr_last(df5)

    if atr == 0 or pd.isna(atr):
        return None

    support = float(df5["Low"].tail(60).min())
    resistance = float(df5["High"].tail(60).max())

    score = 20
    if trend == "BUY" and rsi < 55: score += 20
    if trend == "SELL" and rsi > 45: score += 20
    if trend == "BUY" and macd > 0: score += 20
    if trend == "SELL" and macd < 0: score += 20
    if trend == "BUY" and abs(price - support) < atr * 1.2: score += 20
    if trend == "SELL" and abs(price - resistance) < atr * 1.2: score += 20

    strength = min(score, 100)

    if use_15m:
        df15 = fetch(symbol, "15m")
        if df15 is None or len(df15) < 200:
            return None
        t15 = "BUY" if ema_last(df15["Close"], 50) > ema_last(df15["Close"], 200) else "SELL"
        if t15 != trend:
            return None

    return {
        "symbol": symbol,
        "trend": trend,
        "price": price,
        "strength": strength,
        "support": support,
        "resistance": resistance,
    }

# ---------------- COMMANDS ----------------
@bot.message_handler(commands=["signal", "scan"])
def scan_cmd(msg):
    bot.send_message(msg.chat.id, "🔍 Scanning market...")

    assets = get_assets()
    use_15m = THRESHOLDS[MODE]["USE_15M"]
    min_strength = THRESHOLDS[MODE]["MIN_STRENGTH"]

    results = []

    for a in assets[:MAX_ASSETS]:
        if a["payout"] < PAYOUT_MIN:
            continue

        res = analyze(a["symbol"], use_15m)
        if res and res["strength"] >= min_strength:
            res["display"] = a["display"]
            res["payout"] = a["payout"]
            results.append(res)

        time.sleep(1)

    if not results:
        bot.send_message(msg.chat.id, "❌ No strong signals right now")
        return

    results.sort(key=lambda x: x["strength"], reverse=True)

    out = []
    for r in results:
        out.append(
            f"📌 <b>{r['display']}</b>\n"
            f"🔔 {r['trend']} | {r['strength']}%\n"
            f"💰 Payout {int(r['payout']*100)}%\n"
            f"⏱ Expiry {EXPIRY_MIN} min\n"
            f"—"
        )

    bot.send_message(msg.chat.id, "\n".join(out))

@bot.message_handler(commands=["start", "help"])
def help_cmd(msg):
    bot.send_message(
        msg.chat.id,
        "📡 Signal Bot\n"
        "/signal — get signals\n"
        "/scan — same as signal"
    )

# ---------------- WEBHOOK ----------------
@app.route("/webhook", methods=["POST"])
def webhook():
    update = telebot.types.Update.de_json(request.get_data(as_text=True))
    bot.process_new_updates([update])
    return "OK", 200

@app.route("/")
def root():
    return "OK", 200

# ---------------- RUN ----------------
if __name__ == "__main__":
    bot.delete_webhook()
    bot.set_webhook(WEBHOOK_URL)
    app.run(host="0.0.0.0", port=10000)

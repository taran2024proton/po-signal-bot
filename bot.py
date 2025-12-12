# ===============================================================
# main.py — Stable Render Webhook Bot (FINAL VERSION)
# ===============================================================

import json
import time
from pathlib import Path
from datetime import datetime, timedelta, UTC # Added UTC
import yfinance as yf
import pandas as pd
import telebot
from flask import Flask, request

# ---------------- CONFIG ----------------
TOKEN = "8517986396:AAENPrASLsQlLu21BxG-jKIYZEaEL-RKxYs"
WEBHOOK_URL = "po-signal-bot-gwu0.onrender.com"

ASSETS_FILE = "assets.json"
CACHE_FILE = "cache.json"
CACHE_SECONDS = 90

MAX_ASSETS_PER_SCAN = 6
PAYOUT_MIN = 0.85
EXPIRY_MIN = 3

MODE = "conservative"
THRESHOLDS = {
    "conservative": {"MIN_STRENGTH": 80, "USE_15M": True},
    "aggressive": {"MIN_STRENGTH": 70, "USE_15M": False}
}

bot = telebot.TeleBot(TOKEN, parse_mode="HTML")
app = Flask(__name__)

# ---------------- CACHE (FIXED UTC) ----------------
def load_cache():
    try:
        return json.loads(Path(CACHE_FILE).read_text(encoding="utf-8"))
    except:
        return {}

def save_cache(c):
    try:
        Path(CACHE_FILE).write_text(json.dumps(c), encoding="utf-8")
    except:
        pass

cache = load_cache()

def cache_get(key):
    if key not in cache:
        return None
    try:
        ts = datetime.fromisoformat(cache[key]["_ts"])
        if datetime.now(UTC) - ts > timedelta(seconds=CACHE_SECONDS):
            return None
        return cache[key]["data"]
    except:
        return None

def cache_set(key, data):
    cache[key] = {"_ts": datetime.now(UTC).isoformat(), "data": data}
    save_cache(cache)

# ---------------- INDICATORS ----------------
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def macd_hist(series, fast=12, slow=26, signal=9):
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line

def atr(df, period=14):
    hl = df["High"] - df["Low"]
    h_pc = (df["High"] - df["Close"].shift()).abs()
    l_pc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, h_pc, l_pc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ---------------- ASSETS (Fixed single asset for testing) ----------------
def ensure_assets():
    try:
        return json.loads(Path(ASSETS_FILE).read_text(encoding="utf-8"))
    except:
        default = [
            {"symbol": "EURUSD=X", "display": "EUR/USD", "payout": 0.90},
            # Removed other assets to debug yfinance issue
        ]
        Path(ASSETS_FILE).write_text(json.dumps(default, ensure_ascii=False, indent=2))
        return default

# ---------------- FETCH DATA (Fixed auto_adjust) ----------------
def fetch_ohlcv(symbol, interval):
    key = f"{symbol}_{interval}"
    cached = cache_get(key)

    if cached:
        try:
            return pd.read_json(cached).set_index("Datetime")
        except:
            pass

    try:
        # Added auto_adjust=True to fix FutureWarning
        df = yf.download(symbol, period="3d", interval=interval, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        js = df.reset_index().to_json(date_format="iso")
        cache_set(key, js)
        return df.reset_index().set_index("Datetime")
    except Exception as e:
        # Log error if download fails
        print(f"yfinance download error for {symbol}: {e}")
        return None

# ---------------- ANALYSIS ----------------
def analyze(symbol, use_15m=True):
    df5 = fetch_ohlcv(symbol, "5m")
    if df5 is None or len(df5) < 100:
        return None
    # ... (rest of analyze function remains unchanged) ...
    df5 = df5.tail(300)

    atr_val = atr(df5).iloc[-1]
    if pd.isna(atr_val) or atr_val == 0:
        return None

    ema50 = ema(df5["Close"], 50).iloc[-1]
    ema200 = ema(df5["Close"], 200).iloc[-1]
    trend5 = "BUY" if ema50 > ema200 else "SELL"

    rsi5 = rsi(df5["Close"], 5).iloc[-1]
    macd5 = macd_hist(df5["Close"]).iloc[-1]
    price = df5["Close"].iloc[-1]

    support = df5["Low"].tail(60).min()
    resist = df5["High"].tail(60).max()

    near_s = abs(price - support) <= atr_val * 1.2
    near_r = abs(price - resist) <= atr_val * 1.2

    score = 20
    if trend5 == "BUY" and rsi5 < 55: score += 20
    if trend5 == "SELL" and rsi5 > 45: score += 20
    if trend5 == "BUY" and macd5 > 0: score += 20
    if trend5 == "SELL" and macd5 < 0: score += 20
    if trend5 == "BUY" and near_s: score += 20
    if trend5 == "SELL" and near_r: score += 20

    strength = min(100, score)

    if use_15m:
        df15 = fetch_ohlcv(symbol, "15m")
        if df15 is None or len(df15) < 80:
            return None
        trend15 = "BUY" if ema(df15["Close"], 50).iloc[-1] > ema(df15["Close"], 200).iloc[-1] else "SELL"
        if trend15 != trend5:
            return None

    return {
        "symbol": symbol,
        "trend": trend5,
        "price": float(price),
        "strength": strength,
        "support": float(support),
        "resistance": float(resist)
    }

# ---------------- COMMANDS ----------------
@bot.message_handler(commands=["mode"])
def set_mode(msg):
    global MODE
    _, *rest = msg.text.split()
    if rest and rest[0] in THRESHOLDS:
        MODE = rest[0]
        bot.send_message(msg.chat.id, f"Mode set to {MODE}")
    else:
        bot.send_message(msg.chat.id, f"Current mode: {MODE}")

@bot.message_handler(commands=["signal", "scan"])
def scan(msg):
    # Removed DEBUG prints as they are not appearing in Render logs anyway
    bot.send_message(msg.chat.id, f"🔍 Scanning ({MODE})...")

    assets = ensure_assets()
    use_15m = THRESHOLDS[MODE]["USE_15M"]

    valid = [a for a in assets if a["payout"] >= PAYOUT_MIN][:MAX_ASSETS_PER_SCAN]
    if not valid:
        bot.send_message(msg.chat.id, "No assets with high payout.")
        return

    results = []

    for a in valid:
        key = f"result_{a['symbol']}_{MODE}"
        cached = cache_get(key)

        if cached:
            results.append(cached)
            continue

        res = analyze(a["symbol"], use_15m)
        if res:
            res["display"] = a["display"]
            res["payout"] = a["payout"]
            results.append(res)
            cache_set(key, res)
        else:
            cache_set(key, None)

        time.sleep(1)

    if not results:
        bot.send_message(msg.chat.id, "❌ No strong signals right now.")
        return
    
    results = sorted(results, key=lambda x: (x["strength"], x["payout"]), reverse=True)

    text = []
    for r in results[:5]:
        text.append(
            f"📌 {r['display']} ({r['symbol']})\n"
            f"🔔 {r['trend']} | Strength {r['strength']}%\n"
            f"💰 Payout {int(r['payout']*100)}% | Price {r['price']}\n"
            f"📈 S {r['support']} → R {r['resistance']}\n"
            f"⏱ Expiry {EXPIRY_MIN} min\n—"
        )

    bot.send_message(msg.chat.id, "\n\n".join(text))

@bot.message_handler(commands=["start", "help"])
def help_cmd(msg):
    bot.send_message(msg.chat.id,
        "📡 Trading Signal Bot\n"
        "/signal — scan signals\n"
        "/mode <aggressive|conservative> — set mode\n"
    )

# ---------------- WEBHOOK (Alternative handling) ----------------
@app.route("/webhook", methods=["POST"])
def webhook():
    if request.headers.get("content-type") == "application/json":
        json_string = request.get_data(as_text=True)
        update = telebot.types.Update.de_json(json_string)
        bot.process_new_updates([update])
        return "OK", 200
    else:
        return "Bad Request", 403

@app.route("/")
def root():
    return "OK", 200

# ---------------- RUN ----------------
if __name__ == "__main__":
    bot.delete_webhook()
    bot.set_webhook(WEBHOOK_URL)
    app.run(host="0.0.0.0", port=10000)

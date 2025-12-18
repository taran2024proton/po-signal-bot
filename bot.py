# ===============================================================
# bot.py — FINAL STABLE (MARKET UNCHANGED + OTC SCREEN ANALYSIS)
# ===============================================================

import json
import threading
from pathlib import Path
from datetime import datetime, timedelta, timezone
import io

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

UTC = timezone.utc

bot = telebot.TeleBot(TOKEN, parse_mode="HTML", threaded=False)
app = Flask(__name__)

USER_MODE = {}  # chat_id -> MARKET | OTC

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
    try:
        ts = datetime.fromisoformat(cache[key]["ts"])
    except Exception:
        return None
    if datetime.now(UTC) - ts > timedelta(seconds=CACHE_SECONDS):
        return None
    return cache[key]["data"]

def cache_set(key, data):
    cache[key] = {"ts": datetime.now(UTC).isoformat(), "data": data}
    save_cache(cache)

# ---------------- INDICATORS (MARKET) ----------------
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
    except Exception:
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

# ---------------- DATA (MARKET) ----------------
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

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(df.columns):
        return None

    df = df.reset_index()
    cache_set(key, df.to_json(date_format="iso"))
    return df

# ---------------- MARKET ANALYSIS ----------------
def analyze(symbol, use_15m):
    df5 = fetch(symbol, "5m")
    if df5 is None or len(df5) < 200:
        return None

    close = df5["Close"]
    price = float(close.iloc[-1])

    ema50 = ema_last(close, 50)
    ema200 = ema_last(close, 200)
    trend = "КУПИТИ" if ema50 > ema200 else "ПРОДАТИ"

    rsi = rsi_last(close, 5)
    macd = macd_hist_last(close)
    atr = atr_last(df5)

    if atr == 0 or pd.isna(atr): 
        return None

    support = float(df5["Low"].tail(60).min())
    resistance = float(df5["High"].tail(60).max())

    score = 20
    if trend == "КУПИТИ" and rsi < 55: score += 20
    if trend == "ПРОДАТИ" and rsi > 45: score += 20
    if trend == "КУПИТИ" and macd > 0: score += 20
    if trend == "ПРОДАТИ" and macd < 0: score += 20
    if trend == "КУПИТИ" and abs(price - support) < atr * 1.2: score += 20
    if trend == "ПРОДАТИ" and abs(price - resistance) < atr * 1.2: score += 20

    strength = min(score, 100)

    if use_15m:
        df15 = fetch(symbol, "15m")
        if df15 is None or len(df15) < 200:
            return None
        t15 = "КУПИТИ" if ema_last(df15["Close"], 50) > ema_last(df15["Close"], 200) else "ПРОДАТИ"
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

# ================= OTC SCREEN ANALYSIS =================

import cv2
import numpy as np
from PIL import Image
import io


# ------------------------------------------------------
# 1. ВИТЯГ СВІЧОК ЗІ СКРІНУ
# ------------------------------------------------------

def extract_candles_from_image(image_bytes, count=25):
    """
    Повертає список свічок у форматі:
    open, close, high, low
    (адаптовано під OTC screen-анализ)
    """

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = np.array(img)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candles = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # фільтр "свічка, а не шум"
        if h > w * 2 and h > 25:
            candles.append((x, y, w, h))

    candles = sorted(candles, key=lambda x: x[0])[-count:]

    out = []
    for x, y, w, h in candles:
        out.append({
            "open": y + h * 0.30,
            "close": y + h * 0.70,
            "high": y,
            "low": y + h
        })

    return out

# ------------------------------------------------------
# OTC ANALYZE — ADAPTIVE (2m / 3m)
# ------------------------------------------------------

def otc_analyze(candles):
    """
    Returns:
    {
        direction: "CALL" | "PUT",
        exp: 2 | 3,
        type: "OTC_SOFT_REJECTION" | "OTC_STRONG_REJECTION"
    }
    or None
    """

    if len(candles) < 20:
        return None

    last = candles[-1]
    recent = candles[-20:]

    # ---------- helpers ----------

    def body(c):
        return abs(c["close"] - c["open"])

    def rng(c):
        return c["high"] - c["low"]

    def upper_shadow(c):
        return c["high"] - max(c["open"], c["close"])

    def lower_shadow(c):
        return min(c["open"], c["close"]) - c["low"]

    avg_body = sum(body(c) for c in recent) / 20

    # ---------- 1. OTC FLAT CHECK (SOFT) ----------

    highs = [c["high"] for c in recent]
    lows = [c["low"] for c in recent]

    range_size = max(highs) - min(lows)

    # OTC флет допускаємо ширший
    if range_size > avg_body * 7:
        return None

    high_level = max(highs)
    low_level = min(lows)
    price = last["close"]

    zone = range_size * 0.25  # ⬅️ було 15%, тепер реалістично

    near_high = abs(price - high_level) <= zone
    near_low = abs(price - low_level) <= zone

    if not (near_high or near_low):
        return None

    # ---------- 2. OVERPOWER FILTER ----------

    # Забороняємо ТІЛЬКИ екстремальний імпульс
    if body(last) > rng(last) * 0.85:
        return None

    # ---------- 3. MICRO-EXHAUSTION ----------

    colors = []
    for c in candles[-3:]:
        if c["close"] > c["open"]:
            colors.append(1)
        elif c["close"] < c["open"]:
            colors.append(-1)

    # три однакові — пропуск
    if abs(sum(colors)) == 3:
        return None

    # ---------- 4. REJECTION QUALITY ----------

    up = upper_shadow(last)
    down = lower_shadow(last)
    b = body(last)

    soft_reject = False
    strong_reject = False

    if near_high:
        if up >= b * 0.7:
            soft_reject = True
        if up >= b * 1.3:
            strong_reject = True

    if near_low:
        if down >= b * 0.7:
            soft_reject = True
        if down >= b * 1.3:
            strong_reject = True

    if not soft_reject:
        return None

    # ---------- 5. CONFIRMATION (REALISTIC OTC) ----------

    prev = candles[-2]

    # забороняємо тільки ЧИСТИЙ пробій
    if near_high and prev["close"] > high_level:
        return None

    if near_low and prev["close"] < low_level:
        return None

    # ---------- 6. EXPIRATION LOGIC ----------

    if strong_reject:
        exp = 3
        sig_type = "OTC_STRONG_REJECTION"
    else:
        exp = 2
        sig_type = "OTC_SOFT_REJECTION"

    # ---------- 7. SIGNAL ----------

    if near_low:
        return {
            "direction": "CALL",
            "exp": exp,
            "type": sig_type
        }

    if near_high:
        return {
            "direction": "PUT",
            "exp": exp,
            "type": sig_type
        }

    return None

# ---------------- COMMANDS ----------------
@bot.message_handler(commands=["otc"])
def otc_mode(msg):
    print(f"DEBUG: /otc отримано від chat_id={msg.chat.id}")
    USER_MODE[msg.chat.id] = "OTC"
    try:
        bot.send_message(msg.chat.id, "⚠️ OTC MODE\n📸 Надішли СКРІН з Pocket Option")
        print("DEBUG: Повідомлення /otc відправлено успішно")
    except Exception as e:
        print(f"ERROR sending message: {e}")
        
@bot.message_handler(commands=["market"])
def market_mode(msg):
    print(f"Command /market from chat {msg.chat.id}")
    USER_MODE[msg.chat.id] = "MARKET"
    bot.send_message(msg.chat.id, "✅ MARKET MODE")

@bot.message_handler(commands=["signal", "scan"])
def scan_cmd(msg):
    print(f"Command /signal or /scan from chat {msg.chat.id}")
    if USER_MODE.get(msg.chat.id) == "OTC":
        bot.send_message(msg.chat.id, "❌ У режимі OTC використовуй СКРІН")
        return

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
            results.append({
                "display": a["display"],
                "trend": res["trend"],
                "strength": res["strength"],
                "payout": a["payout"]
            })

    if not results:
        bot.send_message(msg.chat.id, "❌ No strong signals right now")
        return

    results.sort(key=lambda x: x["strength"], reverse=True)

    out = []
    for r in results:
        out.append(
            f"📌 <b><code>{r['display']}</code></b>\n"
            f"🔔 {r['trend']} | {r['strength']}%\n"
            f"💰 Payout {int(r['payout']*100)}%\n"
            f"⏱ Expiry {EXPIRY_MIN} min\n"
            f"—"
        )

    bot.send_message(msg.chat.id, "\n".join(out))

# === OTC PHOTO ===
@bot.message_handler(content_types=["photo"])
def otc_screen(msg):
    print(f"Photo received from chat {msg.chat.id}")
    if USER_MODE.get(msg.chat.id) != "OTC":
        print(f"Chat {msg.chat.id} not in OTC mode, ignoring photo")
        return

    try:
        file_id = msg.photo[-1].file_id
        file_info = bot.get_file(file_id)
        image_bytes = bot.download_file(file_info.file_path)
    except Exception as e:
        print(f"Error downloading photo: {e}")
        bot.send_message(msg.chat.id, "❌ Помилка при завантаженні фото")
        return
        
    bot.send_message(msg.chat.id, "📥 Скрін отримано\n🔍 OTC аналіз...")

    candles = extract_candles_from_image(image_bytes)
    signal = otc_analyze(candles)

    if not signal:
        bot.send_message(msg.chat.id, "❌ OTC-сигналу немає")
        return

    bot.send_message(
        msg.chat.id,
        f"🔥 <b>OTC SIGNAL</b>\n"
        f"📊 {signal}\n"
        f"⏱ Експірація 1 хв\n"
        f"⚠️ Ризик: СЕРЕДНІЙ"
    )
    
# ---------------- WEBHOOK ----------------
@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_data(as_text=True)
    print(f"DEBUG: Отримано update json: {data}")

    update = telebot.types.Update.de_json(data)
    print(f"DEBUG: Створено об'єкт update: {update}")

    try:
        bot.process_new_updates([update])  # Замість threading.Thread(...)
        print("DEBUG: Виконано process_new_updates")
    except Exception as e:
        print(f"ERROR в process_new_updates: {e}")

    return "OK", 200

@app.route("/")
def root():
    return "Bot is running", 200

# ---------------- RUN ----------------
if __name__ == "__main__":
    import os

    print("Starting bot server...")
    print(f"Webhook URL should be set to: {WEBHOOK_URL}")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

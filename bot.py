# =====================
# bot.py — FINAL STABLE
# =====================

import json
import threading
from pathlib import Path
from datetime import datetime, timedelta, timezone
import os
import time
import io
import math

import requests
import pandas as pd
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from flask import Flask, request

import cv2
import numpy as np
from PIL import Image

# ---------------- CONFIG ----------------
TOKEN = os.getenv("TELEGRAM_TOKEN")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")

WEBHOOK_URL = "https://po-signal-bot-gwu0.onrender.com/webhook"

ASSETS_FILE = "assets.json"
CACHE_FILE = "cache.json"
CACHE_SECONDS = 120

EXPIRY_MIN = 5
MAX_ASSETS = 15

UTC = timezone.utc

bot = telebot.TeleBot(TOKEN, parse_mode="HTML", threaded=False)
app = Flask(__name__)

USER_MODE = {}  # chat_id -> MARKET | OTC

THRESHOLDS = {
    "MARKET": {"MIN_STRENGTH": 65, "USE_15M": True},
    "OTC": {"MIN_STRENGTH": 0, "USE_15M": False},
}

bot = telebot.TeleBot(TOKEN, parse_mode="HTML", threaded=False)
app = Flask(__name__)

USER_MODE = {}  # chat_id -> MARKET | OTC

# ---------------- HELPERS ----------------
def normalize_symbol(symbol: str) -> str:
    if symbol.startswith("FX:"):
        return symbol.replace("FX:", "").replace("_", "/")
    return symbol

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
    item = cache.get(key)
    if not item:
        return None
    ts = datetime.fromisoformat(item["ts"])
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    if datetime.now(UTC) - ts > timedelta(seconds=CACHE_SECONDS):
        cache.pop(key, None)
        return None
    return item["data"]

def cache_set(key, data):
    cache[key] = {"ts": datetime.now(UTC).isoformat(), "data": data}
    if len(cache) > 50:
        cache.pop(next(iter(cache)))
    save_cache(cache)
    
# ---------------- REALTIME MARKET DATA ----------------
def fetch_realtime(symbol):
    api_key = os.getenv("TWELVEDATA_API_KEY")
    if not api_key:
        print("ERROR: TWELVEDATA_API_KEY not set")
        return None

    # Twelve Data використовує формат символів з косою рискою (наприклад, "USD/JPY")
    symbol_td = normalize_symbol(symbol)

    cache_key = f"realtime:{symbol_td}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    url = "https://api.twelvedata.com/quote"
    params = {
        "symbol": symbol,
        "apikey": api_key
    }

    try:
        r = requests.get(url, params=params, timeout=5)
        data = r.json()

        if data.get("status") == "error":
            print(f"TwelveData error ({symbol_td}): {data.get('message')}")
            return None

        required = ["open", "high", "low", "close"]
        if not all(k in data for k in required):
            print(f"ERROR: incomplete quote data for {symbol_td}: {data}")
            return None

        result = {
            "Open": float(data["open"]),
            "High": float(data["high"]),
            "Low": float(data["low"]),
            "Close": float(data["close"])
        }

        cache_set(cache_key, result)
        return result

    except Exception as e:
        print(f"ERROR in fetch_realtime ({symbol_td}): {e}")
        return None

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
    assets = [
        # Валютні пари (FX)
        {"symbol": "FX:GBP_JPY", "display": "GBP/JPY", "category": "forex"},
        {"symbol": "FX:AUD_CAD", "display": "AUD/CAD", "category": "forex"},
        {"symbol": "FX:AUD_CHF", "display": "AUD/CHF", "category": "forex"},
        {"symbol": "FX:AUD_JPY", "display": "AUD/JPY", "category": "forex"},
        {"symbol": "FX:AUD_USD", "display": "AUD/USD", "category": "forex"},
        {"symbol": "FX:CAD_CHF", "display": "CAD/CHF", "category": "forex"},
        {"symbol": "FX:CAD_JPY", "display": "CAD/JPY", "category": "forex"},
        {"symbol": "FX:CHF_JPY", "display": "CHF/JPY", "category": "forex"},
        {"symbol": "FX:EUR_AUD", "display": "EUR/AUD", "category": "forex"},
        {"symbol": "FX:EUR_CAD", "display": "EUR/CAD", "category": "forex"},
        {"symbol": "FX:EUR_CHF", "display": "EUR/CHF", "category": "forex"},
        {"symbol": "FX:EUR_GBP", "display": "EUR/GBP", "category": "forex"},
        {"symbol": "FX:EUR_USD", "display": "EUR/USD", "category": "forex"},
        {"symbol": "FX:EUR_JPY", "display": "EUR/JPY", "category": "forex"},
        {"symbol": "FX:GBP_AUD", "display": "GBP/AUD", "category": "forex"},
        {"symbol": "FX:GBP_CHF", "display": "GBP/CHF", "category": "forex"},
        {"symbol": "FX:GBP_USD", "display": "GBP/USD", "category": "forex"},
        {"symbol": "FX:GBP_CAD", "display": "GBP/CAD", "category": "forex"},
        {"symbol": "FX:USD_CAD", "display": "USD/CAD", "category": "forex"},
        {"symbol": "FX:USD_CHF", "display": "USD/CHF", "category": "forex"},
        {"symbol": "FX:USD_JPY", "display": "USD/JPY", "category": "forex"},

        # Акції (Stocks)
        {"symbol": "AAPL", "display": "Apple", "category": "stocks"},
        {"symbol": "BA", "display": "Boeing Company", "category": "stocks"},
        {"symbol": "JPM", "display": "JPMorgan Chase & Co", "category": "stocks"},
        {"symbol": "MCD", "display": "McDonald's", "category": "stocks"},
        {"symbol": "MSFT", "display": "Microsoft", "category": "stocks"},
        {"symbol": "AXP", "display": "American Express", "category": "stocks"},
        {"symbol": "JNJ", "display": "Johnson & Johnson", "category": "stocks"},
        {"symbol": "PFE", "display": "Pfizer Inc", "category": "stocks"},
        {"symbol": "XOM", "display": "ExxonMobil", "category": "stocks"},
        {"symbol": "CSCO", "display": "Cisco", "category": "stocks"},
        {"symbol": "META", "display": "Facebook Inc (Meta)", "category": "stocks"},
        {"symbol": "INTC", "display": "Intel", "category": "stocks"},
        {"symbol": "NFLX", "display": "Netflix", "category": "stocks"},
        {"symbol": "BABA", "display": "Alibaba", "category": "stocks"},
        {"symbol": "TSLA", "display": "Tesla", "category": "stocks"},
        {"symbol": "C", "display": "Citigroup Inc", "category": "stocks"},

        # Криптовалюти (Crypto)
        {"symbol": "BINANCE:BTCUSDT", "display": "Bitcoin", "category": "crypto"},
        {"symbol": "BINANCE:DASHUSDT", "display": "Dash", "category": "crypto"},
        {"symbol": "BINANCE:ETHUSDT", "display": "Ethereum", "category": "crypto"},
        {"symbol": "BINANCE:BCHUSDT", "display": "Bitcoin Cash (BCH/USD)", "category": "crypto"},
        {"symbol": "BINANCE:BCHEUR", "display": "Bitcoin Cash (BCH/EUR)", "category": "crypto"},
        {"symbol": "BINANCE:BCHGBP", "display": "Bitcoin Cash (BCH/GBP)", "category": "crypto"},
        {"symbol": "BINANCE:BCHJPY", "display": "Bitcoin Cash (BCH/JPY)", "category": "crypto"},
        {"symbol": "BINANCE:BTCGBP", "display": "Bitcoin (BTC/GBP)", "category": "crypto"},
        {"symbol": "BINANCE:BTCJPY", "display": "Bitcoin (BTC/JPY)", "category": "crypto"},
        {"symbol": "BINANCE:LINKUSDT", "display": "Chainlink", "category": "crypto"},

        # Індекси (Indices)
        {"symbol": "INDEX:AUS200", "display": "AUS 200", "category": "indices"},
        {"symbol": "INDEX:US100", "display": "US100", "category": "indices"},
        {"symbol": "INDEX:E35EUR", "display": "E35EUR", "category": "indices"},
        {"symbol": "INDEX:100GBP", "display": "100GBP", "category": "indices"},
        {"symbol": "INDEX:F40EUR", "display": "F40/EUR", "category": "indices"},
        {"symbol": "INDEX:JPN225", "display": "JPN225", "category": "indices"},
        {"symbol": "INDEX:D30EUR", "display": "D30/EUR", "category": "indices"},
        {"symbol": "INDEX:E50EUR", "display": "E50/EUR", "category": "indices"},
        {"symbol": "INDEX:SP500", "display": "SP500", "category": "indices"},
        {"symbol": "INDEX:DJI30", "display": "DJI30", "category": "indices"},
        {"symbol": "INDEX:AEX25", "display": "AEX 25", "category": "indices"},
        {"symbol": "INDEX:CAC40", "display": "CAC 40", "category": "indices"},
        {"symbol": "INDEX:HONGKONG33", "display": "HONG KONG 33", "category": "indices"},
        {"symbol": "INDEX:SMI20", "display": "SMI 20", "category": "indices"},
    ]

    if not Path(ASSETS_FILE).exists():
        Path(ASSETS_FILE).write_text(json.dumps(assets, indent=2))

    return assets

# ---------------- DATA (MARKET) ----------------

INTERVAL_MAP = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "1h": "1h"
}

CANDLES_BACK = {
    "1min": 300,
    "5min": 300,
    "15min": 200,
    "1h": 120
}

def fetch(symbol: str, interval: str):
    api_key = os.getenv("TWELVEDATA_API_KEY")
    if not api_key:
        print("ERROR: TWELVEDATA_API_KEY not set")
        return None

    symbol_td = normalize_symbol(symbol)
    interval_td = INTERVAL_MAP.get(interval, "5min")

    cache_key = f"candles:{symbol_td}:{interval_td}"
    cached = cache_get(cache_key)
    if cached:
        return pd.read_json(cached)

    limit = CANDLES_BACK.get(interval_td, 300)

    params = {
        "symbol": symbol_td,
        "interval": interval_td,
        "outputsize": limit,
        "apikey": api_key
    }

    try:
        r = requests.get(
            "https://api.twelvedata.com/time_series",
            params=params,
            timeout=8
        )
        data = r.json()

        if data.get("status") == "error":
            print(f"TwelveData error ({symbol_td}): {data.get('message')}")
            return None

        values = data.get("values")
        if not values or len(values) < 50:
            print(f"DEBUG: Not enough candles for {symbol_td}")
            return None

        df = pd.DataFrame(values)
        df = df.astype({
            "open": float,
            "high": float,
            "low": float,
            "close": float
        })

        df = df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close"
        })

        df = df.iloc[::-1].reset_index(drop=True)

        cache_set(cache_key, df.to_json())
        return df

    except Exception as e:
        print(f"ERROR fetch({symbol_td}): {e}")
        return None
        
# ---------------- MARKET ANALYSIS ----------------

def detect_market_state(df):
    close = df["Close"]

    ema50 = ema_last(close, 50)
    ema200 = ema_last(close, 200)
    macd = macd_hist_last(close)
    atr = atr_last(df)

    if atr is None or atr == 0 or pd.isna(atr):
        return None

    # FLAT
    if (
        abs(ema50 - ema200) < atr * 0.25 and
        abs(macd) < atr * 0.20
    ):
        return "FLAT"

    return "TREND"

def analyze_flat(symbol, df):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    atr = atr_last(df)
    if atr is None or atr == 0:
        return None

    lookback = 60
    support = low.tail(lookback).min()
    resistance = high.tail(lookback).max()

    channel = resistance - support
    if channel < atr * 1.8:
        return None

    price = close.iloc[-1]
    zone = atr * 0.6

    if abs(price - support) <= zone:
        return {
            "trend": "КУПИТИ",
            "type": "FLAT_REBOUND",
            "strength": 70
        }

    if abs(price - resistance) <= zone:
        return {
            "trend": "ПРОДАТИ",
            "type": "FLAT_REBOUND",
            "strength": 70
        }

    return None

def analyze_trend(symbol, df, use_15m):
    close = df["Close"]

    ema50 = ema_last(close, 50)
    ema200 = ema_last(close, 200)
    rsi = rsi_last(close, 7)
    macd = macd_hist_last(close)
    atr = atr_last(df)

    if atr is None or atr == 0:
        return None

    trend = "КУПИТИ" if ema50 > ema200 else "ПРОДАТИ"
    score = 50

    if trend == "КУПИТИ" and 35 < rsi < 55:
        score += 20
    if trend == "ПРОДАТИ" and 45 < rsi < 65:
        score += 20

    if macd > 0 and trend == "КУПИТИ":
        score += 15
    if macd < 0 and trend == "ПРОДАТИ":
        score += 15

    if score < 70:
        return None

    if use_15m:
        df15 = fetch(symbol, "15m")
        if df15 is None or len(df15) < 200:
            return None

        ema50_15 = ema_last(df15["Close"], 50)
        ema200_15 = ema_last(df15["Close"], 200)

        if (ema50 > ema200) != (ema50_15 > ema200_15):
            return None

    return {
        "trend": trend,
        "type": "TREND_PULLBACK",
        "strength": min(score, 100)
    }

def analyze(symbol, use_15m):
    df = fetch(symbol, "5m")
    if df is None or len(df) < 200:
        return None

    state = detect_market_state(df)
    if state is None:
        return None

    if state == "FLAT":
        res = analyze_flat(symbol, df)
    else:
        res = analyze_trend(symbol, df, use_15m)

    if not res:
        return None

    print(f"MARKET {symbol} | {state} | {res['type']} | strength={res['strength']}")

    return {
        "symbol": symbol,
        "trend": res["trend"],
        "strength": res["strength"],
        "state": state
    }

# ================= OTC SCREEN ANALYSIS =================

import cv2
import numpy as np
from PIL import Image
import io
import math

# ---------- GLOBAL HELPERS (Допоміжні функції) ----------

def body(c):
    return abs(c["close"] - c["open"])


def rng(c):
    return max(1e-6, c["high"] - c["low"])


def upper_shadow(c):
    return max(0.0, c["high"] - max(c["open"], c["close"]))


def lower_shadow(c):
    return max(0.0, min(c["open"], c["close"]) - c["low"])
    
# 1. ВИТЯГ СВІЧОК (ВИПРАВЛЕНО РОЗПІЗНАВАННЯ)

def extract_candles_from_image(image_bytes, count=30):
    import cv2
    import numpy as np
    from PIL import Image
    import io

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = np.array(img)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Маски кольорів (Pocket Option)
    mask_green = cv2.inRange(hsv, (40, 50, 50), (90, 255, 255))
    mask_red1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
    mask_red2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
    
    mask_red_combined = cv2.bitwise_or(mask_red1, mask_red2)
    mask_combined = cv2.bitwise_or(mask_green, mask_red_combined)

    contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candles = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        if h < 8 or w < 3:
            continue
            
        y_top = y
        y_bottom = y + h

        high = float(y_top)
        low = float(y_bottom)

        body_top = y + int(h * 0.25)
        body_bottom = y + int(h * 0.75)

        open_price = float(body_bottom)
        close_price = float(body_top)

        if abs(open_price - close_price) < 1:
            continue
            
        if low - high < 4:
            continue
                
        candles.append({
            "x": x,
            "open": open_price,
            "close": close_price,
            "high": high,
            "low": low
        })

    candles = sorted(candles, key=lambda c: c["x"])

    return candles[-count:]

# OTC ANALYZE — ADAPTIVE (2m / 3m)

def otc_analyze(candles):
    MIN_CANDLES = 20
    MAX_RANGE_MULTIPLIER = 18
    BODY_RATIO_THRESHOLD = 0.95
    SHADOW_BODY_RATIO_WEAK = 0.4
    SHADOW_BODY_RATIO_SOFT = 0.7
    SHADOW_BODY_RATIO_STRONG = 1.3
    ZONE_MULTIPLIER = 0.4

    if not candles or len(candles) < MIN_CANDLES:
        return None, "Мало свічок"

    recent = candles[-MIN_CANDLES:]
    last = recent[-1]

    avg_body = sum(body(c) for c in recent) / MIN_CANDLES
    highs = [c["high"] for c in recent]
    lows = [c["low"] for c in recent]

    range_size = max(highs) - min(lows)
    if range_size > avg_body * MAX_RANGE_MULTIPLIER:
        return None, "Занадто широкий діапазон"

    high_level = max(highs)
    low_level = min(lows)
    
    price = last["close"]
    zone = range_size * ZONE_MULTIPLIER

    near_high = abs(price - high_level) <= zone
    near_low = abs(price - low_level) <= zone

    if not (near_high or near_low):
        return None, "Ціна не в зоні підтримки або опору"

    if body(last) > rng(last) * BODY_RATIO_THRESHOLD:
        return None, "Свічка занадто потужна"

    up = upper_shadow(last)
    down = lower_shadow(last)
    b = body(last)

    soft = False
    strong = False

    if near_high:
        if up >= b * SHADOW_BODY_RATIO_SOFT:
            soft = True
        if up >= b * SHADOW_BODY_RATIO_STRONG:
            strong = True
            
    if near_low:
        if down >= b * SHADOW_BODY_RATIO_SOFT:
            soft = True
        if down >= b * SHADOW_BODY_RATIO_STRONG:
            strong = True

    if not soft:
        return None, "Відбій занадто слабкий"

    prev = recent[-2]
    
    if near_high and prev["close"] > high_level:
        return None, "Пробій вгору"
        
    if near_low and prev["close"] < low_level:
        return None, "Пробій вниз"

    return {
        "direction": "CALL" if near_low else "PUT",
        "exp": 3 if strong_reject else 2,
        "type": "OTC_STRONG_REJECTION" if strong_reject else "OTC_SOFT_REJECTION"
    }, "OK"
    
# TREND FOLLOWING ANALYZE 

def trend_analyze(candles):
    MIN_CANDLES = 20
    MIN_RANGE_MULTIPLIER = 5
    MAX_BODY_MULTIPLIER = 1.5
    
    if len(candles) < MIN_CANDLES:
        return None

    recent = candles[-MIN_CANDLES:]
    last = recent[-1]

    avg_body = sum(body(c) for c in recent) / MIN_CANDLES
    trend = 1 if recent[0]["close"] < recent[-1]["close"] else -1
    
    range_size = max(c["high"] for c in recent) - min(c["low"] for c in recent)
    if range_size < avg_body * MIN_RANGE_MULTIPLIER:
        return None

    if trend == 1 and last["close"] >= last["open"]:
        return None
    if trend == -1 and last["close"] <= last["open"]:
        return None

    if body(last) > avg_body * MAX_BODY_MULTIPLIER:
        return None

    return {
        "direction": "CALL" if trend == 1 else "PUT",
        "exp": 2,
        "type": "TREND_PULLBACK"
    }

# BREAKOUT ANALYZE 

def breakout_analyze(candles):
    if len(candles) < 20:
        return None

    last = candles[-1]
    recent = candles[-20:]

    def body_ratio(c):
        return body(c) / rng(c)

    highs = [c["high"] for c in recent[:-1]]
    lows = [c["low"] for c in recent[:-1]]
    
    high_level = max(highs)
    low_level = min(lows)

    is_breakout_up = last["close"] > high_level and last["open"] <= high_level
    is_breakout_down = last["close"] < low_level and last["open"] >= low_level

    if not (is_breakout_up or is_breakout_down):
        return None

    if body_ratio(last) < 0.7:
        return None
        
    avg_body = sum(body(c) for c in recent[:-1]) / 19
    range_size = high_level - low_level
    
    if range_size > avg_body * 6:
         return None

    if is_breakout_up:
        return {
            "direction": "CALL",
            "exp": 2,
            "type": "BREAKOUT_CALL"
        }

    if is_breakout_down:
        return {
            "direction": "PUT",
            "exp": 2,
            "type": "BREAKOUT_PUT"
        }
        
    return None
    
# MARKET DISPATCHER

def analyze_market(candles):
    if not candles or len(candles) < 30:
        return None

    res = breakout_analyze(candles)
    if res:
        res["source"] = "breakout"
        return res

    res = trend_analyze(candles)
    if res:
        res["source"] = "trend"
        return res

    res, msg = otc_analyze(candles)
    if res:
        res["source"] = "otc"
        return res

    return None
    
# ---------------- COMMANDS ----------------
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
import threading

USER_MODE = {}  # chat_id -> "OTC" або "MARKET"

EXPIRY_MIN = 5
MAX_ASSETS = 20

THRESHOLDS = {
    "MARKET": {
        "USE_15M": True,
        "MIN_STRENGTH": 65,
    },
    "OTC": {
        "USE_15M": False,
        "MIN_STRENGTH": 0,
    }
}

@bot.message_handler(commands=["start", "help"])
def start_help(msg):
    bot.send_message(
        msg.chat.id,
        "Виберіть режим роботи бота:\n"
        "/otc — режим OTC (надсилайте скріншоти)\n"
        "/market — режим MARKET (аналіз пар)"
    )

@bot.message_handler(commands=["otc"])
def otc_mode(msg):
    USER_MODE[msg.chat.id] = "OTC"
    try:
        bot.send_message(msg.chat.id, "⚠️ Ви ввімкнули режим OTC\n📸 Надішліть скріншот з Pocket Option для аналізу.")
    except Exception as e:
        print(f"ERROR sending OTC message: {e}")
        
@bot.message_handler(commands=["market"])
def market_mode(msg):
    USER_MODE[msg.chat.id] = "MARKET"

    assets = get_assets()  # ← ВАЖЛИВО

    kb = InlineKeyboardMarkup(row_width=5)

    row = []
    for asset in assets:
        row.append(
            InlineKeyboardButton(
                text=asset["display"],
                callback_data=f"MARKET_PAIR:{asset['symbol']}"
            )
        )
        if len(row) == 5:
            kb.row(*row)
            row = []

    if row:
        kb.row(*row)

    try:
        bot.send_message(
            msg.chat.id,
            "📊 <b>Режим MARKET</b>\nОберіть валютну пару:",
            reply_markup=kb,
            parse_mode="HTML"
        )
    except Exception as e:
        print(f"ERROR sending MARKET keyboard: {e}")

@bot.callback_query_handler(func=lambda call: call.data.startswith("MARKET_PAIR:"))
def market_pair_selected(call):
    chat_id = call.message.chat.id
    if USER_MODE.get(chat_id) != "MARKET":
        bot.answer_callback_query(call.id, "❌ Ви не в режимі MARKET")
        return

    symbol = call.data.replace("MARKET_PAIR:", "")
    display = symbol.replace("FX:", "").replace("_", "/")

    bot.answer_callback_query(call.id)
    bot.send_message(chat_id, f"🔍 Аналізую <b>{display}</b>...", parse_mode="HTML")

    mode = USER_MODE.get(chat_id, "MARKET")
    use_15m = THRESHOLDS[mode].get("USE_15M", True)
    min_strength = THRESHOLDS[mode].get("MIN_STRENGTH", 65)

    try:
        res = analyze(symbol, use_15m)
    except Exception as e:
        print(f"ANALYZE ERROR for {symbol}: {e}")
        bot.send_message(chat_id, "❌ Помилка аналізу. Спробуйте пізніше.")
        return

    if not res or res["strength"] < min_strength:
        bot.send_message(chat_id, f"❌ По парі <code>{display}</code> сильних сигналів немає.", parse_mode="HTML")
        return

    bot.send_message(
        chat_id,
        f"🔥 <b>MARKET SIGNAL</b>\n"
        f"📌 <code>{display}</code>\n"
        f"🔔 {res['trend']} | {res['strength']}%\n"
        f"⏱ Expiry {EXPIRY_MIN} хв",
        parse_mode="HTML"
    )

    send_market_keyboard(chat_id)

@bot.message_handler(commands=["signal", "scan"])
def scan_cmd(msg):
    chat_id = msg.chat.id
    mode = USER_MODE.get(chat_id)

    if mode == "OTC":
        bot.send_message(chat_id, "❌ У режимі OTC використовуйте надсилання скріншотів для аналізу.")
        return
    elif mode != "MARKET":
        bot.send_message(chat_id, "⚠️ Виберіть режим роботи: /otc або /market")
        return

    bot.send_message(chat_id, "🔍 Сканую ринок... Це може зайняти кілька секунд.")

    threading.Thread(target=process_market_scan, args=(chat_id,)).start()

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

        bot.send_message(msg.chat.id, "📥 Скрін отримано\n🔍 Проводжу OTC аналіз...")

        candles = extract_candles_from_image(image_bytes)

        if not candles or len(candles) < 20:
            bot.send_message(msg.chat.id, "❌ Недостатньо даних на скріні для аналізу.")
            return

        signal, reason = otc_analyze(candles)

        if not signal:
            bot.send_message(msg.chat.id, f"❌ OTC сигнал не знайдено: {reason}")
            return

        direction_ua = "CALL (КУПІВЛЯ)" if signal["direction"] == "CALL" else "PUT (ПРОДАЖ)"
        expiry_min = signal.get("exp", 1)

        bot.send_message(
            msg.chat.id,
            f"🔥 <b>OTC SIGNAL</b>\n"
            f"📊 Напрямок: {direction_ua}\n"
            f"⏱ Експірація: {expiry_min} хв\n"
            f"⚠️ Ризик: СЕРЕДНІЙ",
            parse_mode="HTML"
        )
        
    except Exception as e:
        print(f"ERROR in OTC photo processing: {e}")
        bot.send_message(msg.chat.id, "❌ Помилка при обробці фото. Спробуйте ще раз.")

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

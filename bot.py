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
from collections import defaultdict
from datetime import date

STATS = defaultdict(lambda: {"win": 0, "loss": 0, "draw": 0})

LAST_SIGNALS = {}

# =====================
# SIGNAL COOLDOWN
# =====================
COOLDOWN_SECONDS = 10 * 60  # 10 хвилин
LAST_SIGNAL_TIME = {}

# =====================
# API CACHE (1 хв)
# =====================
API_CACHE = {}
CACHE_TTL = 120  # секунд

# ---------------- CONFIG ----------------
TOKEN = os.getenv("TELEGRAM_TOKEN")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")

WEBHOOK_URL = "https://po-signal-bot-gwu0.onrender.com/webhook"

ASSETS_FILE = "assets.json"
CACHE_FILE = "cache.json"
CACHE_SECONDS = CACHE_TTL

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

# ---------------- HELPERS ----------------
def normalize_symbol(symbol: str) -> str:
    if symbol.startswith("FX:"):
        return symbol.replace("FX:", "").replace("_", "/")
    return symbol

from datetime import datetime, timedelta

def next_m5_entry_time():
    now = datetime.utcnow()
    now_local = now + timedelta(hours=2)
    
    minute = (now_local.minute // 5 + 1) * 5

    if minute >= 60:
        entry_time = now_local.replace(
            hour=(now_local.hour + 1) % 24,
            minute=0,
            second=0,
            microsecond=0
        )
    else:
        entry_time = now_local.replace(
            minute=minute,
            second=0,
            microsecond=0
        )

    return entry_time.strftime("%H:%M")

# ---------------- CACHE ----------------
def load_cache():
    try:
        text = Path(CACHE_FILE).read_text()
        return json.loads(text)
    except Exception:
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
        print(f"Cache expired for key {key}")
        cache.pop(key, None)
        return None
    data = item["data"]
    
    if isinstance(data, list):
        try:
            df = pd.DataFrame(data)
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
            return df
        except Exception:
            return data
    return data

def cache_set(key, data):
    if isinstance(data, pd.DataFrame):
        data_to_save = data.to_dict(orient="records")
    else:
        data_to_save = data
    cache[key] = {"ts": datetime.now(UTC).isoformat(), "data": data_to_save}
    if len(cache) > 50:
        cache.pop(next(iter(cache)))
    save_cache(cache)

# ---------------- INDICATORS (MARKET) ----------------
def ema_last(series, period):
    return series.ewm(span=period, adjust=False).mean().iloc[-1]

def rsi_last(series, period=7):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return (100 - (100 / (1 + rs))).iloc[-1]

def macd_hist_last(series):
    fast = series.ewm(span=6, adjust=False).mean()
    slow = series.ewm(span=13, adjust=False).mean()
    macd = fast - slow
    signal = macd.ewm(span=5, adjust=False).mean()
    return (macd - signal).iloc[-1]

def atr_last(df, period=14):
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]

def adx_last(df, period=14):
    df = df.copy()

    df["TR"] = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)

    df["+DM"] = df["High"].diff()
    df["-DM"] = df["Low"].diff() * -1

    df["+DM"] = df.apply(lambda x: x["+DM"] if x["+DM"] > x["-DM"] and x["+DM"] > 0 else 0, axis=1)
    df["-DM"] = df.apply(lambda x: x["-DM"] if x["-DM"] > x["+DM"] and x["-DM"] > 0 else 0, axis=1)

    tr14 = df["TR"].rolling(period).sum()
    plus14 = df["+DM"].rolling(period).sum()
    minus14 = df["-DM"].rolling(period).sum()

    plusDI = 100 * (plus14 / tr14)
    minusDI = 100 * (minus14 / tr14)

    dx = (abs(plusDI - minusDI) / (plusDI + minusDI)) * 100
    adx = dx.rolling(period).mean()

    return adx.iloc[-1]
    
# ---------------- ASSETS ----------------
def get_assets():
    assets = [
        # Валютні пари (FX)
        {"symbol": "FX:EUR_USD", "display": "EUR/USD", "category": "forex"},
        {"symbol": "FX:GBP_USD", "display": "GBP/USD", "category": "forex"},
        {"symbol": "FX:USD_JPY", "display": "USD/JPY", "category": "forex"},
        {"symbol": "FX:USD_CHF", "display": "USD/CHF", "category": "forex"},
        {"symbol": "FX:AUD_USD", "display": "AUD/USD", "category": "forex"},
        {"symbol": "FX:EUR_CHF", "display": "EUR/CHF", "category": "forex"},
        {"symbol": "FX:AUD_CHF", "display": "AUD/CHF", "category": "forex"},
        {"symbol": "FX:USD_CAD", "display": "USD/CAD", "category": "forex"},
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
        return cached

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
        try:
            data = r.json()
        except Exception:
            print(f"TwelveData error ({symbol_td}): invalid JSON response")
            return None

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

    print(f"Detect market state: ema50={ema50}, ema200={ema200}, macd={macd}, atr={atr}")

    if atr is None or atr == 0 or pd.isna(atr):
        print("ATR invalid or zero")
        return None

    atr_avg = atr_last(df.tail(50))
    print(f"ATR average last 50: {atr_avg}")
    if atr < atr_avg * 0.7:
        print("Market is dead (low ATR)")
        return None

    if abs(ema50 - ema200) < atr * 0.3 and abs(macd) < atr * 0.25:
        print("Market is flat")
        return "FLAT"

    print("Market is trending")
    return "TREND"

def analyze_flat(symbol, df):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    atr = atr_last(df)
    print(f"Flat analyze: ATR={atr}")
    if atr is None or atr == 0:
        print("ATR invalid or zero in flat analysis")
        return None

    lookback = 80
    support = low.tail(lookback).min()
    resistance = high.tail(lookback).max()
    print(f"Flat analyze: support={support}, resistance={resistance}")

    channel = resistance - support
    print(f"Channel size: {channel}")
    if channel < atr * 2.2:
        print("Channel too small in flat analysis")
        return None

    price = close.iloc[-1]
    zone = atr * 0.5
    print(f"Price={price}, zone={zone}")
    
    if abs(price - support) <= zone:
        print("Flat signal: BUY zone")
        return {"trend": "КУПИТИ", "strength": 72}

    if abs(price - resistance) <= zone:
        print("Flat signal: SELL zone")
        return {"trend": "ПРОДАТИ", "strength": 72}

    print("No flat signal")
    return None


def analyze_trend(symbol, df, use_15m):
    close = df["Close"]
    
    ema50 = ema_last(close, 50)
    ema200 = ema_last(close, 200)
    rsi = rsi_last(close, 7)
    macd = macd_hist_last(close)
    atr = atr_last(df)
    adx = adx_last(df)
    print(f"ADX={adx}")

    if adx < 18:
        print("Trend too weak (ADX)")
        return None

    print(f"Trend analyze: ema50={ema50}, ema200={ema200}, rsi={rsi}, macd={macd}, atr={atr}")

    if atr is None or atr == 0:
        print("ATR invalid or zero in trend analysis")
        return None

    trend = "КУПИТИ" if ema50 > ema200 else "ПРОДАТИ"
    score = 65

    if trend == "КУПИТИ" and 38 <= rsi <= 50:
        score += 20
    if trend == "ПРОДАТИ" and 50 <= rsi <= 62:
        score += 20

    if macd > atr * 0.05 and trend == "КУПИТИ":
        score += 20
    if macd < -atr * 0.05 and trend == "ПРОДАТИ":
        score += 20

    print(f"Score for {symbol}: {score}")

    if score < 65:
        print("Score below threshold")
        return None

    if use_15m:
        df15 = fetch(symbol, "15m")
        print(f"Fetching 15m data for {symbol}, rows: {len(df15) if df15 is not None else 'None'}")
        if df15 is None or len(df15) < 200:
            print("Not enough 15m data")
            return None

        ema50_15 = ema_last(df15["Close"], 50)
        ema200_15 = ema_last(df15["Close"], 200)
        print(f"15m EMA50={ema50_15}, EMA200={ema200_15}")
        if (ema50_15 > ema200_15) != (trend == "КУПИТИ"):
            print("15m trend mismatch")
            return None
            
    return {"trend": trend, "strength": score}

def find_support_resistance(df, window=20):
    highs = df["High"].rolling(window=window, center=True).max()
    lows = df["Low"].rolling(window=window, center=True).min()

    resistance_levels = df["High"][(df["High"] == highs)].tolist()
    support_levels = df["Low"][(df["Low"] == lows)].tolist()

    resistance_levels = sorted(set(round(x, 5) for x in resistance_levels))
    support_levels = sorted(set(round(x, 5) for x in support_levels))

    return support_levels, resistance_levels

def is_pin_bar(candle):
    body = abs(candle["Close"] - candle["Open"])
    range_ = candle["High"] - candle["Low"]
    upper_shadow = candle["High"] - max(candle["Close"], candle["Open"])
    lower_shadow = min(candle["Close"], candle["Open"]) - candle["Low"]

    if body < range_ * 0.3 and (upper_shadow > body * 2 or lower_shadow > body * 2):
        return True
    return False

from datetime import datetime, timedelta

def analyze_1m_entry(df_1m, trend, support_levels=None, resistance_levels=None):

    if df_1m is None or len(df_1m) < 50:
        return None

    close = df_1m["Close"]
    open_ = df_1m["Open"]

    ema20 = close.ewm(span=20, adjust=False).mean()
    last = df_1m.iloc[-1]
    prev = df_1m.iloc[-2]

    body = abs(last["Close"] - last["Open"])
    avg_body = abs(close.diff()).rolling(20).mean().iloc[-1]

    if trend == "КУПИТИ":
        if (
            last["Close"] > ema20.iloc[-1] and
            last["Close"] > prev["Close"] and
            body > avg_body * 0.8
        ):
            return {"entry": "CALL", "confidence": 2}

    if trend == "ПРОДАТИ":
        if (
            last["Close"] < ema20.iloc[-1] and
            last["Close"] < prev["Close"] and
            body > avg_body * 0.8
        ):
            return {"entry": "PUT", "confidence": 2}

    return None

def analyze(symbol, use_15m):
    df5 = fetch(symbol, "5m")
    if df5 is None or len(df5) < 200:
        print(f"Not enough 5m data for {symbol}")
        return None

    state = detect_market_state(df5)
    print(f"Market state for {symbol}: {state}")
    if state is None:
        print(f"Market state is None for {symbol}")
        return None

    if state == "FLAT":
        res = analyze_flat(symbol, df5)
        print(f"Flat analysis result for {symbol}: {res}")
    else:
        res = analyze_trend(symbol, df5, use_15m)
        print(f"Trend analysis result for {symbol}: {res}")
    if not res:
        print(f"5m returned None for {symbol}")
        return None

    min_strength = THRESHOLDS["MARKET"]["MIN_STRENGTH"]

    if res["strength"] < min_strength:
        print(f"Strength too low on 5m for {symbol}")
        return None

    trend = res["trend"]
    print(f"Determined trend for {symbol}: {trend}")

    support_levels, resistance_levels = find_support_resistance(df5, window=20)

    df1 = fetch(symbol, "1m")
    print(f"Fetching 1m data for {symbol}, rows: {len(df1) if df1 is not None else 'None'}")
    if df1 is None or len(df1) < 30:
        print(f"Not enough 1m data for {symbol}")
        return None

    entry = analyze_1m_entry(df1, trend, support_levels, resistance_levels)
    print(f"1m entry analysis for {symbol}: {entry}")
    if not entry:
        print(f"No entry signal for {symbol}")
        return None       

    print(f"SIGNAL {symbol} | {trend} | 5m→1m")

    res["strength"] += 5
    return {
        "symbol": symbol,
        "signal": entry["entry"],
        "strength": res["strength"],
        "trend": res["trend"]
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

import time
import threading

MIN_STRENGTH = 70

def automatic_market_analysis(bot, chat_id, assets):
    index = 0
    assets_count = len(assets)
    while USER_MODE.get(chat_id) == "MARKET":
        for _ in range(1):
            asset = assets[index % assets_count]
            symbol = asset["symbol"]
            display_name = asset["display"]

            now = time.time()
            last_time = LAST_SIGNAL_TIME.get(symbol)
            if last_time and now - last_time < COOLDOWN_SECONDS:
                index += 1
                continue
            
            try:
                res = analyze(symbol, use_15m=True)  
                if res and res.get("strength", 0) >= MIN_STRENGTH:
                    entry_time = next_m5_entry_time()
                    trend = res["trend"].lower()

                    signal_key = f"{symbol}|{trend}|{entry_time}|{chat_id}"

                    if LAST_SIGNALS.get(chat_id) == signal_key:
                        index += 1
                        continue
                    LAST_SIGNALS[chat_id] = signal_key

                    trend_raw = res['trend'].upper()
                    
                    if "КУП" in trend_raw:
                        trend_display = "🟢 Купити"
                    elif "ПРОД" in trend_raw:
                        trend_display = "🔴 Продати"
                    else:
                        trend_display = trend_raw
                        
                    message = (
                        f"🔥 <b>MARKET SIGNAL</b>\n"
                        f"🪙 <code>{display_name}</code>\n"
                        f"{trend_display} | {res['strength']}%\n"
                        f"⏰ Вхід: <b>{entry_time}</b>\n"
                        f"⏳ Експірація: {EXPIRY_MIN} хв"
                    )

                    markup = InlineKeyboardMarkup()
                    markup.row(
                        InlineKeyboardButton("✅", callback_data=f"win|{symbol}|{entry_time}"),
                        InlineKeyboardButton("🟰", callback_data=f"draw|{symbol}|{entry_time}"),
                        InlineKeyboardButton("❌", callback_data=f"loss|{symbol}|{entry_time}")
                    )
                    
                    bot.send_message(
                        chat_id,
                        message,
                        parse_mode="HTML",
                        reply_markup=markup
                    )
                    
                    LAST_SIGNAL_TIME[symbol] = now
                    
                time.sleep(15)
                
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")

            index += 1
            
        time.sleep(12)
    
# ---------------- COMMANDS ----------------
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
import threading

USER_MODE = {}  # chat_id -> "OTC" або "MARKET"
STATS = {}

EXPIRY_MIN = 5
MAX_ASSETS = 20

@bot.message_handler(commands=["start", "help"])
def start_help(msg):
    bot.send_message(
        msg.chat.id,
        "Виберіть режим роботи бота:\n"
        "/otc — режим OTC (надсилайте скріншоти)\n"
        "/market — режим MARKET (аналіз пар)"
    )
    
@bot.message_handler(commands=["stop"])
def stop_mode(msg):
    USER_MODE[msg.chat.id] = None  # або можна del USER_MODE[msg.chat.id], якщо хочеш повністю прибрати
    bot.send_message(msg.chat.id, "⏹ Режим MARKET вимкнено. Аналіз зупинено.")

@bot.message_handler(commands=["testsignal"])
def testsignal(msg):
    symbol = "FX:EUR_USD"
    res = analyze(symbol, use_15m=True)
    bot.send_message(msg.chat.id, f"Test analyze for {symbol}: {res}")

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
    bot.send_message(msg.chat.id, "📊 Режим MARKET увімкнено. Аналізую пари.")

    assets = get_assets()
    
    # Запускаємо автоматичний аналіз у фоновому потоці
    analysis_thread = threading.Thread(target=automatic_market_analysis, args=(bot, msg.chat.id, assets))
    analysis_thread.daemon = True
    analysis_thread.start()

STATS = {}

@bot.message_handler(commands=["stats"])
def stats_today(message):
    today = date.today().isoformat()
    stats = STATS.get(today)

    if not stats:
        bot.send_message(message.chat.id, "📊 За сьогодні ще немає угод.")
        return

    wins = stats["win"]
    losses = stats["loss"]
    draws = stats["draw"]
    total = wins + losses + draws
    winrate = round((wins / total) * 100, 1) if total > 0 else 0

    bot.send_message(
        message.chat.id,
        f"📊 <b>Статистика за сьогодні</b>\n\n"
        f"✅ Win: {wins}\n"
        f"❌ Loss: {losses}\n"
        f"🟰 Draw: {draws}\n"
        f"📈 Winrate: {winrate}%",
        parse_mode="HTML"
    )

@bot.callback_query_handler(func=lambda call: call.data.startswith("MARKET_PAIR:"))
def market_pair_selected(call):
    print("CALLBACK MARKET:", call.data)
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
    min_strength = THRESHOLDS[mode].get("MIN_STRENGTH", 70)

    try:
        res = analyze(symbol, use_15m)
    except Exception as e:
        print(f"ANALYZE ERROR for {symbol}: {e}")
        bot.send_message(chat_id, "❌ Помилка аналізу. Спробуйте пізніше.")
        return

    if not res or res["strength"] < min_strength:
        bot.send_message(chat_id, f"❌ По парі <code>{display}</code> сильних сигналів немає.", parse_mode="HTML")
        send_market_keyboard(chat_id)
        return
    
    entry_time = next_m5_entry_time()
    
    markup = InlineKeyboardMarkup()
    markup.row(
        InlineKeyboardButton("✅", callback_data=f"win|{symbol}|{entry_time}"),
        InlineKeyboardButton("🟰", callback_data=f"draw|{symbol}|{entry_time}"),
        InlineKeyboardButton("❌", callback_data=f"loss|{symbol}|{entry_time}")
    )

    bot.send_message(
        chat_id,
        f"🔥 <b>MARKET SIGNAL</b>\n"
        f"📌 <code>{display}</code>\n"
        f"🔔 {res['trend']} | {res['strength']}%\n"
        f"🕒 Вхід в угоду: <b>{entry_time}</b>\n"
        f"⏱ Expiry {EXPIRY_MIN} хв",
        parse_mode="HTML",
        reply_markup=markup
    )

    send_market_keyboard(chat_id)
    
@bot.callback_query_handler(func=lambda call: call.data.split("|")[0] in ["win", "loss", "draw"])
def handle_result_callback(call):
    print("CALLBACK RESULT:", call.data)
    result, symbol, entry_time = call.data.split("|")
    today = date.today().isoformat()
    chat_id = call.message.chat.id

    if today not in STATS:
        STATS[today] = {"win": 0, "loss": 0, "draw": 0}

    STATS[today][result] += 1

    bot.answer_callback_query(call.id, f"Збережено: {result.upper()}")

    bot.edit_message_reply_markup(
        call.message.chat.id,
        call.message.message_id,
        reply_markup=None
    )
    
def send_market_keyboard(chat_id):
    assets = get_assets()  # Має повертати список словників зі схемою [{'symbol': 'FX:EUR_USD', 'display': 'EUR/USD'}, ...]

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

    bot.send_message(
        chat_id,
        "📊 <b>Режим MARKET</b>\nОберіть валютну пару:",
        reply_markup=kb,
        parse_mode="HTML"
    )
    
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
        bot.process_new_updates([update])
        print("DEBUG: Виконано process_new_updates")
    except Exception as e:
        print(f"ERROR в process_new_updates: {e}")

    return "OK", 200


@app.route("/")
def root():
    return "Bot is running", 200


# ---------------- RUN ----------------
if __name__ == "__main__":
    print("Starting bot server...")
    print(f"Webhook URL should be set to: {WEBHOOK_URL}")
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

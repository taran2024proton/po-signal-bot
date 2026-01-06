# =====================
# bot.py — FINAL STABLE
# =====================

import json
import threading
from pathlib import Path
from datetime import datetime, timedelta, timezone
import io
import os
import time

import requests
import pandas as pd
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from flask import Flask, request

# ---------------- CONFIG ----------------
TOKEN = os.getenv("TELEGRAM_TOKEN")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

WEBHOOK_URL = "https://po-signal-bot-gwu0.onrender.com/webhook"

ASSETS_FILE = "assets.json"
CACHE_FILE = "cache.json"
CACHE_SECONDS = 120

EXPIRY_MIN = 5
MAX_ASSETS = 15

MODE = "aggressive"
THRESHOLDS = {
    "conservative": {"MIN_STRENGTH": 70, "USE_15M": True},
    "aggressive": {"MIN_STRENGTH": 60, "USE_15M": False},
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
    item = cache.get(key)
    if not item:
        return None

    try:
        ts = datetime.fromisoformat(item["ts"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
    except Exception:
        return None

    if datetime.now(UTC) - ts > timedelta(seconds=CACHE_SECONDS):
        cache.pop(key, None)
        return None

    return item["data"]

def cache_set(key, data):
    cache[key] = {
        "ts": datetime.now(UTC).isoformat(),
        "data": data
    }

    if len(cache) > 50:
        oldest_key = min(
            cache.items(),
            key=lambda x: x[1]["ts"]
        )[0]
        cache.pop(oldest_key, None)

    save_cache(cache)
    
# ---------------- REALTIME MARKET DATA ----------------
def fetch_realtime(symbol):
    token = os.getenv("FINNHUB_API_KEY")
    if not token:
        print("ERROR: FINNHUB_API_KEY not set")
        return None

    try:
        url = "https://finnhub.io/api/v1/quote"
        params = {
            "symbol": symbol.replace("=X", ""),
            "token": token
        }
        r = requests.get(url, params=params, timeout=3)
        data = r.json()

        if not data or data.get("c") is None:
            return None

        return {
            "Close": data["c"],
            "High": data["h"],
            "Low": data["l"],
            "Open": data["o"]
        }
    except Exception as e:
        print(f"ERROR in fetch_realtime: {e}")
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
    Path(ASSETS_FILE).write_text(json.dumps(assets, indent=2))
    return assets

# ---------------- DATA (MARKET) ----------------

RESOLUTION_MAP = {
    "1m": "1",
    "5m": "5",
    "15m": "15",
    "1h": "60"
}

CANDLES_BACK = {
    "1": 300,
    "5": 300,
    "15": 200,
    "60": 120
}

ENDPOINTS = {
    "forex": "https://finnhub.io/api/v1/forex/candle",
    "crypto": "https://finnhub.io/api/v1/crypto/candle",
    "stock": "https://finnhub.io/api/v1/stock/candle",
    "index": "https://finnhub.io/api/v1/index/candle",
}


def detect_market_type(symbol: str) -> str:
    if symbol.startswith("FX:"):
        return "forex"
    if symbol.startswith("BINANCE:"):
        return "crypto"
    if symbol.startswith("INDEX:"):
        return "index"
    return "stock"


def fetch(symbol: str, interval: str):
    resolution = RESOLUTION_MAP.get(interval, "5")
    market_type = detect_market_type(symbol)
    endpoint = ENDPOINTS.get(market_type)

    if not endpoint:
        print(f"DEBUG: Unsupported market type for {symbol}")
        return None

    cache_key = f"{symbol}_{interval}"
    cached = cache_get(cache_key)
    if cached:
        return pd.read_json(cached)

    try:
        now = int(time.time())
        candles = CANDLES_BACK.get(resolution, 300)
        from_time = now - (candles * int(resolution) * 60)

        params = {
            "symbol": symbol,
            "resolution": resolution,
            "from": from_time,
            "to": now,
            "token": os.getenv("FINNHUB_API_KEY")
        }

        r = requests.get(endpoint, params=params, timeout=5)
        data = r.json()

        if not data or data.get("s") != "ok":
            print(f"DEBUG FETCH FAIL {symbol} [{market_type}] → {data}")
            return None

        df = pd.DataFrame({
            "Open": data["o"],
            "High": data["h"],
            "Low": data["l"],
            "Close": data["c"]
        })

        if df.empty or len(df) < 50:
            print(f"DEBUG: Not enough candles for {symbol}")
            return None

        cache_set(cache_key, df.to_json())
        return df

    except Exception as e:
        print(f"ERROR fetch({symbol}): {e}")
        return None

# ---------------- MARKET ANALYSIS ----------------

def analyze(symbol, use_15m):
    df5 = fetch(symbol, "5m")
    if df5 is None or len(df5) < 200:
        return None

    close = df5["Close"]
    high = df5["High"]
    low = df5["Low"]

    price = float(close.iloc[-1])

    ema50 = ema_last(close, 50)
    ema200 = ema_last(close, 200)
    rsi = rsi_last(close, 7)
    macd = macd_hist_last(close)
    atr = atr_last(df5)

    if atr is None or atr == 0 or pd.isna(atr):
        return None

    ema_distance = abs(ema50 - ema200)
    if ema_distance < atr * 0.3:
        return None

    trend = "КУПИТИ" if ema50 > ema200 else "ПРОДАТИ"

    lookback = 80
    support = float(low.tail(lookback).min())
    resistance = float(high.tail(lookback).max())

    if (resistance - support) < atr * 1.5:
        return None

    if trend == "КУПИТИ" and price > resistance + atr * 0.3:
        return None
    if trend == "ПРОДАТИ" and price < support - atr * 0.3:
        return None

    score = 40

    if trend == "КУПИТИ" and 35 < rsi < 50:
        score += 15
    if trend == "ПРОДАТИ" and 50 < rsi < 65:
        score += 15

    if trend == "КУПИТИ" and macd > 0:
        score += 15
    if trend == "ПРОДАТИ" and macd < 0:
        score += 15

    if trend == "КУПИТИ" and abs(price - support) < atr * 1.1:
        score += 20
    if trend == "ПРОДАТИ" and abs(price - resistance) < atr * 1.1:
        score += 20

    if ema_distance > 0.08:
        score += 10

    strength = min(score, 100)

    if strength < 65:
        return None

    if use_15m:
        df15 = fetch(symbol, "15m")
        if df15 is None or len(df15) < 200:
            return None

        ema50_15 = ema_last(df15["Close"], 50)
        ema200_15 = ema_last(df15["Close"], 200)
        trend15 = "КУПИТИ" if ema50_15 > ema200_15 else "ПРОДАТИ"

        if trend15 != trend:
            return None

    print(
        f"ANALYZE {symbol} | {trend} | strength={strength} | price={price:.5f} | RSI={rsi:.1f}"
    )
    
    return {
        "symbol": symbol,
        "trend": trend,
        "price": price,
        "strength": strength,
        "support": support,
        "resistance": resistance,
        "rsi": rsi,
        "atr": atr
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
    """
    Аналіз свічок OTC для виявлення сигналів CALL/PUT.

    Повертає кортеж: (сигнал dict або None, причина string).

    Вхідні дані:
    - candles: список словників зі свічками, кожна свічка має ключі:
        'open', 'close', 'high', 'low', 'x' (позиція по осі X, не обов'язково).
    """

    # Конфігураційні параметри (можна винести у глобальні налаштування)
    MIN_CANDLES = 20                   # Мінімальна кількість свічок для аналізу
    MAX_RANGE_MULTIPLIER = 25          # Максимальний мультиплікатор для діапазону (флет)
    BODY_RATIO_THRESHOLD = 0.95        # Максимальне співвідношення body до загального розміру свічки (занадто сильна свічка)
    SHADOW_BODY_RATIO_WEAK = 0.4       # Мінімальне співвідношення тіні до тіла для слабкого відбою
    SHADOW_BODY_RATIO_SOFT = 0.7       # Мінімальне співвідношення для "м'якого" відбою (soft reject)
    SHADOW_BODY_RATIO_STRONG = 1.3     # Мінімальне співвідношення для "сильного" відбою (strong reject)
    ZONE_MULTIPLIER = 0.4              # Розмір зони поблизу рівня підтримки/опору

    if len(candles) < MIN_CANDLES:
        return None, "Мало свічок для аналізу"

    last = candles[-1]
    recent = candles[-MIN_CANDLES:]

    for idx, c in enumerate(recent):
        if any(k not in c for k in ("open", "close", "high", "low")):
            return None, f"Некоректні дані свічки під індексом {idx}"
            
    def body(c):
        return abs(c["close"] - c["open"])

    def rng(c):
        return max(0.000001, c["high"] - c["low"])

    def upper_shadow(c):
        return c["high"] - max(c["open"], c["close"])

    def lower_shadow(c):
        return min(c["open"], c["close"]) - c["low"]

    avg_body = sum(body(c) for c in recent) / MIN_CANDLES
    highs = [c["high"] for c in recent]
    lows = [c["low"] for c in recent]

    range_size = max(highs) - min(lows)

    # Флет — занадто великий діапазон свічок (відсічка)
    if range_size > avg_body * MAX_RANGE_MULTIPLIER:
        return None, f"Діапазон надто широкий: {range_size:.5f} > {avg_body * MAX_RANGE_MULTIPLIER:.5f}"

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

    if near_high and up < b * SHADOW_BODY_RATIO_WEAK:
        return None, "Слабкий відбій від верхнього рівня"

    if near_low and down < b * SHADOW_BODY_RATIO_WEAK:
        return None, "Слабкий відбій від нижнього рівня"

    soft_reject = False
    strong_reject = False

    if near_high:
        if up >= b * SHADOW_BODY_RATIO_SOFT:
            soft_reject = True
        if up >= b * SHADOW_BODY_RATIO_STRONG:
            strong_reject = True

    if near_low:
        if down >= b * SHADOW_BODY_RATIO_SOFT:
            soft_reject = True
        if down >= b * SHADOW_BODY_RATIO_STRONG:
            strong_reject = True

    if not soft_reject:
        return None, "Відбій занадто слабкий"

    prev = candles[-2]
    if near_high and prev["close"] > high_level:
        return None, "Попередня свічка закрилась вище рівня опору"

    if near_low and prev["close"] < low_level:
        return None, "Попередня свічка закрилась нижче рівня підтримки"

    if strong_reject:
        exp = 3
        sig_type = "OTC_STRONG_REJECTION"
    else:
        exp = 2
        sig_type = "OTC_SOFT_REJECTION"

    if near_low:
        return {
            "direction": "CALL",
            "exp": exp,
            "type": sig_type
        }, "OK"

    if near_high:
        return {
            "direction": "PUT",
            "exp": exp,
            "type": sig_type
        }, "OK"

    return None, "Без сигналу"
    
# TREND FOLLOWING ANALYZE 

def trend_analyze(candles):
    """
    Аналіз тренду для пошуку сигналів на продовження руху.

    Параметри:
    - candles: список свічок у форматі dict з ключами open, close, high, low.

    Повертає:
    - dict з напрямком ("CALL" або "PUT") і експірацією,
      або None, якщо сигнал відсутній.
    """

    MIN_CANDLES = 20            # Мінімальна кількість свічок для аналізу
    MIN_RANGE_MULTIPLIER = 5    # Мінімальний мультиплікатор range_size / avg_body для тренду
    MAX_BODY_MULTIPLIER = 1.5   # Максимальний розмір тіла для корекції
    
    if len(candles) < MIN_CANDLES:
        return None

    last = candles[-1]
    recent = candles[-MIN_CANDLES:]

    def body(c):
        return abs(c["close"] - c["open"])

    avg_body = sum(body(c) for c in recent) / MIN_CANDLES
    
    # 1. Визначення напрямку тренду за зміною ціни за період
    trend_direction = 0
    if recent[0]["close"] < recent[-1]["close"]:
        trend_direction = 1 # UP
    elif recent[0]["close"] > recent[-1]["close"]:
        trend_direction = -1 # DOWN

    # 2. Фільтр сили тренду — діапазон має бути значний
    range_size = max([c["high"] for c in recent]) - min([c["low"] for c in recent])
    if range_size < avg_body * MIN_RANGE_MULTIPLIER:
        return None # Тренд слабкий або немає

    # 3. Фільтр корекції — остання свічка повинна бути проти тренду (корекція)
    if trend_direction == 1 and last["close"] >= last["open"]:
        return None  # Для тренду вгору очікуємо корекцію вниз (закриття < відкриття)
    if trend_direction == -1 and last["close"] <= last["open"]:
        return None  # Для тренду вниз очікуємо корекцію вгору (закриття > відкриття)

    # 4. Фільтр тіла останньої свічки — має бути невеликим (корекція, а не імпульс)
    if body(last) > avg_body * MAX_BODY_MULTIPLIER:
        return None

     # 5. Сигнал: вхід за напрямком тренду після корекції
        return {
            "direction": "CALL" if trend_direction == 1 else "PUT",
            "exp": 2  # Рекомендована експірація 2 свічки
        }

# BREAKOUT ANALYZE 

def breakout_analyze(candles):
    """
    Аналіз пробою рівня (Brekaout). Шукає імпульсний рух за межі діапазону.
    """
    if len(candles) < 20:
        return None

    last = candles[-1]
    recent = candles[-20:]

    def body_ratio(c):
        return body(c) / rng(c) if rng(c) > 0 else 0

    # 1. Визначення рівнів і діапазону (flat/range)
    highs = [c["high"] for c in recent[:-1]]  # Виключаємо останню свічку
    lows = [c["low"] for c in recent[:-1]]
    high_level = max(highs)
    low_level = min(lows)

    # 2. Перевірка пробою
    is_breakout_up = last["close"] > high_level and last["open"] <= high_level
    is_breakout_down = last["close"] < low_level and last["open"] >= low_level

    if not (is_breakout_up or is_breakout_down):
        return None

    # 3. Фільтр сили імпульсу
    if body_ratio(last) < 0.7:  # Тіло має займати >70% свічки
        return None
        
    avg_body = sum(body(c) for c in recent[:-1]) / 19
    range_size = high_level - low_level
    
    if range_size > avg_body * 6:
         return None # Найімовірніше, це вже був тренд, а не консолідація

    # 4. СИГНАЛ
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

    # Пріоритет 1: Пробій рівня (найсильніший імпульс)
    res = breakout_analyze(candles)
    if res:
        res["source"] = "breakout"
        return res

    # Пріоритет 2: Трендовий відкат
    res = trend_analyze(candles)
    if res:
        res["source"] = "trend"
        return res

    # Пріоритет 3: Флет та OTC розвороти
    res, msg = otc_analyze(candles)
    if res:
        res["source"] = "otc"
        return res

    return None
    
# ---------------- COMMANDS ----------------
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
import threading

# Змінні, які використовуєш
USER_MODE = {}  # chat_id -> "OTC" або "MARKET"
EXPIRY_MIN = 5  # наприклад
MAX_ASSETS = 20  # обмеження для сканування
THRESHOLDS = {
    "MARKET": {
        "USE_15M": True,
        "MIN_STRENGTH": 65,
    },
    "OTC": {
        # Тут можуть бути свої налаштування, якщо потрібно
    }
}

# Припускаю, get_assets() вже є і повертає список активів з 'symbol' і 'display'

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
    assets = get_assets()

    kb = InlineKeyboardMarkup(row_width=3)
    for asset in assets:
        kb.add(InlineKeyboardButton(text=asset["display"], callback_data=f"MARKET_PAIR:{asset['symbol']}"))

    try:
        bot.send_message(
            msg.chat.id,
            "📊 <b>Режим MARKET</b>\nОберіть валютну пару для аналізу:",
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

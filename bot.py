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

PAYOUT_MIN = 0.80
EXPIRY_MIN = 3
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
    ts = datetime.fromisoformat(item["ts"])
    if datetime.now(UTC) - ts > timedelta(seconds=CACHE_SECONDS):
        return None
    return item["data"]

def cache_set(key, data):
    cache[key] = {"ts": datetime.now(UTC).isoformat(), "data": data}
    if len(cache) > 50:  # ⬅️ ОБМЕЖЕННЯ CACHE
        cache.clear()
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
        period="1d",
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
    if df5 is None or len(df5) < 120:
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

    score = 50
    if trend == "КУПИТИ" and rsi < 60: score += 15
    if trend == "ПРОДАТИ" and rsi > 40: score += 15
    if trend == "КУПИТИ" and macd > 0: score += 15
    if trend == "ПРОДАТИ" and macd < 0: score += 15
    if trend == "КУПИТИ" and abs(price - support) < atr * 1.2: score += 15
    if trend == "ПРОДАТИ" and abs(price - resistance) < atr * 1.2: score += 15

    strength = min(score, 100)

    if use_15m:
        df15 = fetch(symbol, "15m")
        if df15 is None or len(df15) < 120:
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

# ---------- GLOBAL HELPERS (Допоміжні функції) ----------

def body(c):
    return abs(c["close"] - c["open"])

def rng(c):
    return max(0.000001, c["high"] - c["low"])

def upper_shadow(c):
    return c["high"] - max(c["open"], c["close"])

def lower_shadow(c):
    return min(c["open"], c["close"]) - c["low"]

# ------------------------------------------------------
# 1. ВИТЯГ СВІЧОК (ВИПРАВЛЕНО РОЗПІЗНАВАННЯ)
# ------------------------------------------------------
def extract_candles_from_image(image_bytes, count=30):
    import cv2
    import numpy as np
    from PIL import Image
    import io

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = np.array(img)
    h_img, w_img, _ = img.shape

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # --- ВИПРАВЛЕННЯ 1: Повний діапазон HSV для червоного та зеленого ---
    # Зелений
    mask_green = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([90, 255, 255]))
    
    # Червоний: два діапазони
    mask_red1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    
    # Об'єднання масок
    mask_red_combined = cv2.bitwise_or(mask_red1, mask_red2)
    mask_combined = cv2.bitwise_or(mask_green, mask_red_combined)

    contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_candles = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 5 and w > 2:  # Фільтр шуму
            # Використовуємо центр тіла для визначення кольору
            mid_pixel_rgb = img[y + h // 2, x + w // 2]
            is_green = mid_pixel_rgb[1] > mid_pixel_rgb[0]  # Перевірка G > R

            # Координати свічки (Y=0 зверху, Y зростає вниз)
            # ВИПРАВЛЕННЯ 2: high_coord та low_coord - це координати Y
            high_coord = y  # найвища точка на екрані (менше Y)
            low_coord = y + h # найнижча точка на екрані (більше Y)

            if is_green:
                # Для зеленої свічки: Open < Close. Open ціна знаходиться нижче на екрані (більше Y)
                open_coord = low_coord
                close_coord = high_coord
            else:
                # Для червоної свічки: Open > Close. Open ціна знаходиться вище на екрані (менше Y)
                open_coord = high_coord
                close_coord = low_coord
                
            raw_candles.append({
                "x": x,
                # Тут ми повертаємо координати Y, трактуючи їх як "ціни" на екрані
                "open": float(open_coord), 
                "close": float(close_coord),
                "high": float(high_coord),
                "low": float(low_coord)
            })

    # Сортуємо по осі X (зліва направо) та беремо останні count
    raw_candles = sorted(raw_candles, key=lambda x: x["x"])[-count:]
    return raw_candles

# ------------------------------------------------------
# OTC ANALYZE — ADAPTIVE (2m / 3m)
# ------------------------------------------------------

def otc_analyze(candles):
    """
    Повертає кортеж (signal_dict або None, message_reason)
    """
    if len(candles) < 20:
        return None, "Мало свічок для аналізу"

    last = candles[-1]
    recent = candles[-20:]

    def body(c):
        return abs(c["close"] - c["open"])

    def rng(c):
        return max(0.000001, c["high"] - c["low"])

    def upper_shadow(c):
        return c["high"] - max(c["open"], c["close"])

    def lower_shadow(c):
        return min(c["open"], c["close"]) - c["low"]

    avg_body = sum(body(c) for c in recent) / 20

    highs = [c["high"] for c in recent]
    lows = [c["low"] for c in recent]

    range_size = max(highs) - min(lows)

     # OTC флет допускаємо ширший
    if range_size > avg_body * 25:
        return None, "Діапазон надто широкий"

    high_level = max(highs)
    low_level = min(lows)
    price = last["close"]

    zone = range_size * 0.4

    near_high = abs(price - high_level) <= zone
    near_low = abs(price - low_level) <= zone

    if not (near_high or near_low):
        return None, "Ціна не в зоні підтримки/опору"

    if body(last) > rng(last) * 0.95:
        return None, "Свічка занадто потужна"

    up = upper_shadow(last)
    down = lower_shadow(last)
    b = body(last)

    if near_high and up < b * 0.4:
        return None, "Слабкий відбій від верхнього рівня"

    if near_low and down < b * 0.4:
        return None, "Слабкий відбій від нижнього рівня"

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
        return None, "Відбій слабкий"

    prev = candles[-2]

    if near_high and prev["close"] > high_level:
        return None, "Попередня свічка вище рівня опору"

    if near_low and prev["close"] < low_level:
        return None, "Попередня свічка нижче рівня підтримки"

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
    
# ------------------------------------------------------
# TREND FOLLOWING ANALYZE 
# ------------------------------------------------------

def trend_analyze(candles):
    """
    Аналіз тренду. Шукає сигнал на продовження руху.
    """
    if len(candles) < 20:
        return None

    last = candles[-1]
    recent = candles[-20:]

    avg_body = sum(body(c) for c in recent) / 20
    
    # 1. ВИЗНАЧЕННЯ ТРЕНДУ (НАПРЯМОК)
    trend_direction = 0
    if recent[0]["close"] < recent[-1]["close"]:
        trend_direction = 1 # UP
    elif recent[0]["close"] > recent[-1]["close"]:
        trend_direction = -1 # DOWN

    # Фільтр: тренд має бути достатньо сильним
    range_size = max([c["high"] for c in recent]) - min([c["low"] for c in recent])
    if range_size < avg_body * 5:
        return None # Недостатньо сильний тренд

    # 2. ФІЛЬТР ВІДКАТУ (КОРЕКЦІЇ)
    if trend_direction == 1:
        if last["close"] > last["open"]:
            return None

    if trend_direction == -1:
        if last["close"] < last["open"]:
            return None

    # 3. ФІЛЬТР ІМПУЛЬСУ НА ВХІД
    # Тіло останньої свічки не має бути занадто великим (це має бути саме корекція, а не розворот)
    if body(last) > avg_body * 1.5:
        return None

    # 4. СИГНАЛ (Вхід в напрямку тренду)
    if trend_direction == 1:
        return {
            "direction": "CALL",
            "exp": 2 # Експірація на 2 свічки
        }

    if trend_direction == -1:
        return {
            "direction": "PUT",
            "exp": 2 # Експірація на 2 свічки
        }

    return None

# ------------------------------------------------------
# BREAKOUT ANALYZE 
# ------------------------------------------------------

def breakout_analyze(candles):
    """
    Аналіз пробою рівня (Brekaout). Шукає імпульсний рух за межі діапазону.
    """
    if len(candles) < 20:
        return None

    last = candles[-1]
    recent = candles[-20:]

    # helpers
    def body_ratio(c):
        return body(c) / rng(c) if rng(c) > 0 else 0

    # 1. ВИЗНАЧЕННЯ РІВНІВ І ДІАПАЗОНУ (FLAT/RANGE)
    highs = [c["high"] for c in recent[:-1]] # Виключаємо останню свічку з розрахунку рівнів
    lows = [c["low"] for c in recent[:-1]]
    high_level = max(highs)
    low_level = min(lows)

    # 2. ПЕРЕВІРКА ПРОБОЮ
    # Остання свічка повинна була закритися за межами попереднього діапазону
    is_breakout_up = last["close"] > high_level and last["open"] <= high_level
    is_breakout_down = last["close"] < low_level and last["open"] >= low_level

    if not (is_breakout_up or is_breakout_down):
        return None # Пробоя не було

    # 3. ФІЛЬТР СИЛИ ІМПУЛЬСУ
    # Тіло останньої свічки має бути великим (імпульсним)
    if body_ratio(last) < 0.7: # Тіло займає > 70% всього діапазону свічки
        return None 
        
    # Також перевіряємо, щоб попередній діапазон не був надто трендовим,
    # інакше це може бути просто продовження тренду, а не чистою пробою консолідації.
    avg_body = sum(body(c) for c in recent[:-1]) / 19
    range_size = max(highs) - min(lows)
    if range_size > avg_body * 6:
         return None # Найімовірніше, це вже був тренд, а не консолідація

    # 4. СИГНАЛ
    if is_breakout_up:
        return {
            "direction": "CALL",
            "exp": 2, # Торгуємо на продовження пробою
            "type": "BREAKOUT_CALL"
        }

    if is_breakout_down:
        return {
            "direction": "PUT",
            "exp": 2,
            "type": "BREAKOUT_PUT"
        }
        
    return None
    
# ------------------------------------------------------
# MARKET DISPATCHER
# ------------------------------------------------------

def analyze_market(candles):
    if not candles or len(candles) < 30:
        return None

    # Пріоритет 1: Пробій рівня (найсильніший імпульс)
    res = breakout_analyze(candles)
    if res: return res

    # Пріоритет 2: Трендовий відкат
    res = trend_analyze(candles)
    if res: return res

    # Пріоритет 3: Флет та OTC розвороти
    res = otc_analyze(candles)
    if res: return res

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

    bot.send_message(msg.chat.id, "🔍 Сканую ринок...")
    
    checked = 0
    skipped_payout = 0

    assets = get_assets()
    use_15m = THRESHOLDS[MODE]["USE_15M"]
    min_strength = THRESHOLDS[MODE]["MIN_STRENGTH"]

    results = []

    for a in assets[:MAX_ASSETS]:
        checked += 1
        
        if a["payout"] < PAYOUT_MIN:
            skipped_payout += 1
            continue

        res = analyze(a["symbol"], use_15m)
        if res is None:
            no_data += 1
            continue
        if res and res["strength"] >= min_strength:
            results.append({
                "display": a["display"],
                "trend": res["trend"],
                "strength": res["strength"],
                "payout": a["payout"]
            })

    if not results:
        bot.send_message(
            msg.chat.id,
            f"ℹ️ Перевірено пар: {checked}\n"
            f"⏭ Пропущено через payout: {skipped_payout}\n"
            f"❌ Сильних сигналів поки немає"
            f"📉 Без даних (yfinance): {no_data}\n"
    )
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

    bot.send_message(msg.chat.id, "\n".join(results))


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

        bot.send_message(msg.chat.id, "📥 Скрін отримано\n🔍 OTC аналіз...")

        candles = extract_candles_from_image(image_bytes)
        signal, reason = otc_analyze(candles)  # Припущення, що повертає (signal, reason)

        if not signal:
            bot.send_message(msg.chat.id, f"❌ OTC сигнал не виявлено: {reason}")
            return

        direction_ua = "CALL (КУПІВЛЯ)" if signal["direction"] == "CALL" else "PUT (ПРОДАЖ)"

        bot.send_message(
            msg.chat.id,
            f"🔥 <b>OTC SIGNAL</b>\n"
            f"📊 Напрямок: {direction_ua}\n"
            f"⏱ Експірація 1 хв\n"
            f"⚠️ Ризик: СЕРЕДНІЙ"
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

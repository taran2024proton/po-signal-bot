# main.py — Тестер/Симулятор сигналів (жорсткі фільтри, expiry info=3min)
import json, time
from pathlib import Path
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import pandas as pd
import telebot

# ========== Налаштування ==========
TOKEN = "8517986396:AAENPrASLsQlLu21BxG-jKIYZEaEL-RKxYs"   # <- встав свій токен
ASSETS_FILE = "assets.json"
CACHE_FILE = "cache.json"
CACHE_SECONDS = 90           # кеш, щоб зменшити запити
MAX_ASSETS_PER_SCAN = 6      # скільки активів сканувати за виклик
PAYOUT_MIN = 0.85            # враховуємо лише активи з payout>=85% (в assets.json)
EXPIRY_MIN = 3               # інформативна експірація
MODE = "conservative"        # conservative (жорстко) або aggressive
THRESHOLDS = {
    "conservative": {"MIN_STRENGTH": 80, "USE_15M": True},
    "aggressive":   {"MIN_STRENGTH": 70, "USE_15M": False}
}
# ==================================

bot = telebot.TeleBot(TOKEN)

# ---------- cache ----------
def load_cache():
    p = Path(CACHE_FILE)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except:
        return {}

def save_cache(cache):
    try:
        Path(CACHE_FILE).write_text(json.dumps(cache), encoding='utf-8')
    except:
        pass

cache = load_cache()

def cache_get(key):
    obj = cache.get(key)
    if not obj:
        return None
    ts = datetime.fromisoformat(obj.get("_ts"))
    if datetime.utcnow() - ts > timedelta(seconds=CACHE_SECONDS):
        return None
    return obj.get("data")

def cache_set(key, data):
    cache[key] = {"_ts": datetime.utcnow().isoformat(), "data": data}
    save_cache(cache)

# ---------- indicators ----------
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

# ---------- assets loader ----------
def ensure_assets():
    p = Path(ASSETS_FILE)
    if not p.exists():
        example = [
            {"symbol":"EURUSD=X","display":"EUR/USD","payout":0.90},
            {"symbol":"GBPUSD=X","display":"GBP/USD","payout":0.88},
            {"symbol":"USDJPY=X","display":"USD/JPY","payout":0.86},
            {"symbol":"BTC-USD","display":"BTC/USD","payout":0.92},
            {"symbol":"ETH-USD","display":"ETH/USD","payout":0.91},
            {"symbol":"AUDUSD=X","display":"AUD/USD","payout":0.87}
        ]
        p.write_text(json.dumps(example, ensure_ascii=False, indent=2))
    return json.loads(p.read_text(encoding='utf-8'))

# ---------- data fetch with cache ----------
def fetch_ohlcv(symbol, interval):
    key = f"{symbol}_{interval}"
    c = cache_get(key)
    if c is not None:
        try:
            return pd.read_json(c).set_index("Datetime")
        except:
            pass
    try:
        df = yf.download(symbol, period="3d", interval=interval, progress=False)
        if df is None or df.empty:
            return None
        df2 = df.reset_index()
        cache_set(key, df2.to_json(date_format='iso'))
        df = df.reset_index().set_index("Datetime")
        return df
    except:
        return None

# ---------- analysis (strict filters) ----------
def analyze_one(symbol_data):
    df5 = symbol_data.copy()
    df5 = df5.tail(300).reset_index(drop=True)

    if df5.empty:
        return None

    # ========== SAFE ATR ==========
    atr_series = atr(df5, 14)
    if (
        atr_series is None or 
        not isinstance(atr_series, pd.Series) or 
        atr_series.dropna().empty
    ):
        return None
    last_atr = float(atr_series.iloc[-1])

    if last_atr == 0 or pd.isna(last_atr):
        return None

    # ========== SAFE EMA ==========
    ema50 = ema(df5, 50)
    ema200 = ema(df5, 200)

    if ema50 is None or ema200 is None:
        return None

    try:
        last_ema50 = float(ema50.iloc[-1].item())
        last_ema200 = float(ema200.iloc[-1].item())
    except:
        return None

    # ========== SAFE RSI ==========
    rsi5 = rsi(df5, 5)
    if rsi5 is None or rsi5.dropna().empty:
        return None

    try:
        last_rsi = float(rsi5.iloc[-1].item())
    except:
        return None

    # ========== SAFE MACD ==========
    macd5 = macd(df5)
    if macd5 is None or macd5.dropna().empty:
        return None

    try:
        last_macd = float(macd5.iloc[-1].item())
    except:
        return None

    # ========== SAFE PRICE ==========
    try:
        last_price = float(df5["close"].iloc[-1])
    except:
        return None

    # ========== SIGNAL LOGIC ==========
    buy_score = 0
    sell_score = 0

    # EMA cross
    if last_ema50 > last_ema200:
        buy_score += 1
    if last_ema50 < last_ema200:
        sell_score += 1

    # RSI
    if last_rsi < 30:
        buy_score += 1
    if last_rsi > 70:
        sell_score += 1

    # MACD
    if last_macd > 0:
        buy_score += 1
    if last_macd < 0:
        sell_score += 1

    # Volatility filter (ATR)
    if last_atr < last_price * 0.001:
        return None  # too low volatility

    # ======= FINAL SIGNAL =======
    if buy_score >= 2 and buy_score > sell_score:
        return {
            "signal": "BUY",
            "power": int((buy_score / 3) * 100),
            "price": last_price
        }

    if sell_score >= 2 and sell_score > buy_score:
        return {
            "signal": "SELL",
            "power": int((sell_score / 3) * 100),
            "price": last_price
        }

    return None


    # indicators
    last_rsi = float(rsi(close5,14).iloc[-1])
    last_macd = float(macd_hist(close5).iloc[-1])

    # support/resistance simple
    window = 60
    support = float(df5['Low'][-window:].min())
    resistance = float(df5['High'][-window:].max())
    last_price = float(close5.iloc[-1])
    near_support = abs(last_price - support) <= max(last_atr*1.2, last_price*0.0015)
    near_resist = abs(last_price - resistance) <= max(last_atr*1.2, last_price*0.0015)

    # strength scoring (5 checks)
    score = 0
    reasons = []
    # 1 trend
    score += 20; reasons.append("Trend")
    # 2 RSI (conservative)
    if trend5 == "BUY" and last_rsi < 55:
        score += 20; reasons.append(f"RSI{int(last_rsi)}")
    if trend5 == "SELL" and last_rsi > 45:
        score += 20; reasons.append(f"RSI{int(last_rsi)}")
    # 3 MACD
    if trend5 == "BUY" and last_macd > 0:
        score += 20; reasons.append("MACD+")
    if trend5 == "SELL" and last_macd < 0:
        score += 20; reasons.append("MACD-")
    # 4 S/R proximity
    if trend5 == "BUY" and near_support:
        score += 20; reasons.append("NearS")
    if trend5 == "SELL" and near_resist:
        score += 20; reasons.append("NearR")
    # 5 volatility
    vol_ok = last_atr >= (last_price * 0.00018)
    if vol_ok:
        score += 20; reasons.append("VolOK")

    strength = int(min(100, score))

    # 15m confirmation (conservative mode)
    if use_15m_confirm:
        df15 = fetch_ohlcv(symbol, '15m')
        if df15 is None or len(df15) < 40:
            return None
        close15 = df15['Close']
        ema50_15 = ema(close15, 50)
        ema200_15 = ema(close15, 200)
        if np.isnan(ema200_15.iloc[-1]):
            return None
        trend15 = "BUY" if ema50_15.iloc[-1] > ema200_15.iloc[-1] else "SELL"
        if trend15 != trend5:
            return None

    # final gating by ATR minimal (avoid vanishing volatility)
    if last_atr < (last_price * 0.00012):
        return None

    if strength >= THRESHOLDS[MODE]["MIN_STRENGTH"]:
        return {
            "symbol": symbol,
            "trend": trend5,
            "price": round(last_price,6),
            "strength": strength,
            "reasons": reasons,
            "support": round(support,6),
            "resistance": round(resistance,6)
        }
    return None

# ---------- commands ----------
@bot.message_handler(commands=["mode"])
def cmd_mode(msg):
    global MODE
    chat = msg.chat.id
    text = msg.text.strip().lower()
    if "/mode" in text:
        parts = text.split()
        if len(parts) == 2 and parts[1] in THRESHOLDS:
            MODE = parts[1]
            bot.send_message(chat, f"Режим: {MODE}. Порог: {THRESHOLDS[MODE]}")
            return
    bot.send_message(chat, f"Поточний режим: {MODE}. Щоб змінити: /mode aggressive або /mode conservative")

@bot.message_handler(commands=["signal","scan"])
def cmd_signal(msg):
    chat = msg.chat.id
    bot.send_message(chat, f"🔎 Сканую (mode={MODE}) — перегляну до {MAX_ASSETS_PER_SCAN} активів з payout≥{int(PAYOUT_MIN*100)}% ...")
    assets = ensure_assets()
    candidates = [a for a in assets if a.get("payout",0) >= PAYOUT_MIN]
    if not candidates:
        bot.send_message(chat, "⚠️ В assets.json немає активів з payout>=threshold.")
        return
    candidates = candidates[:MAX_ASSETS_PER_SCAN]
    res_list = []
    for a in candidates:
        sym = a["symbol"]
        cache_key = f"ana_{sym}_{MODE}"
        cached = cache_get(cache_key)
        if cached:
            res_list.append(cached); continue
        res = analyze_one(sym, use_15m_confirm=THRESHOLDS[MODE]["USE_15M"])
        if res:
            res["display"] = a.get("display", sym)
            res["payout"] = a.get("payout",0)
            res_list.append(res)
            cache_set(cache_key, res)
        else:
            cache_set(cache_key, None)
        time.sleep(0.8)
    if not res_list:
        bot.send_message(chat, "❌ Немає сильних сигналів зараз. Спробуй пізніше або /mode aggressive")
        return
    res_list = sorted(res_list, key=lambda x: (x["strength"], x.get("payout",0)), reverse=True)
    lines = []
    for r in res_list[:5]:
        lines.append(
            f"📌 {r['display']} ({r['symbol']})\n"
            f"🔔 {r['trend']}  |  Strength: {r['strength']}%\n"
            f"💰 Payout: {int(r.get('payout',0)*100)}%  |  Price: {r['price']}\n"
            f"📈 Reasons: {', '.join(r['reasons'])}\n"
            f"🛑 S: {r['support']}  ▶ R: {r['resistance']}\n"
            f"⏱ Expiry (info): {EXPIRY_MIN} min\n"
            "—"
        )
    bot.send_message(chat, "\n\n".join(lines))

@bot.message_handler(commands=["help","start"])
def cmd_help(msg):
    chat = msg.chat.id
    bot.send_message(chat,
        "Тестер сигналів (інформаційно):\n"
        "/signal — шукати сигнали\n"
        "/mode <aggressive|conservative> — переключити режим\n"
        "Режим conservative потребує 15m підтвердження (рідкіше, але якісніше)."
    )

# ---------- run ----------
if __name__ == "__main__":
    print("Tester bot starting (strict filters).")
    bot.infinity_polling(timeout=60, long_polling_timeout=60)

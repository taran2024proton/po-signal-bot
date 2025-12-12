# main.py — Fixed version for Render webhook only (NO polling)
import json, time
from pathlib import Path
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import pandas as pd
import telebot
from flask import Flask, request

# ========== CONFIG ==========
TOKEN = "8517986396:AAENPrASLsQlLu21BxG-jKIYZEaEL-RKxYs"
WEBHOOK_URL = "https://po-signal-bot-gwu0.onrender.com/webhook"
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
# ==================================

bot = telebot.TeleBot(TOKEN)
app = Flask(__name__)

# ---------- cache ----------
def load_cache():
    p = Path(CACHE_FILE)
    if not p.exists(): return {}
    try: return json.loads(p.read_text(encoding='utf-8'))
    except: return {}

def save_cache(cache):
    try: Path(CACHE_FILE).write_text(json.dumps(cache), encoding='utf-8')
    except: pass

cache = load_cache()

def cache_get(key):
    obj = cache.get(key)
    if not obj: return None
    ts = datetime.fromisoformat(obj.get("_ts"))
    if datetime.now(datetime.UTC) - ts > timedelta(seconds=CACHE_SECONDS):
        return None
    return obj.get("data")

def cache_set(key, data):
    cache[key] = {"_ts": datetime.now(datetime.UTC).isoformat(), "data": data}
    save_cache(cache)

# ---------- indicators ----------
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    d = series.diff(); up = d.clip(lower=0); dn = -1 * d.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_dn = dn.rolling(period).mean()
    rs = ma_up / ma_dn
    return 100 - (100 / (1 + rs))

def macd_hist(series, fast=12, slow=26, signal=9):
    ef = series.ewm(span=fast, adjust=False).mean()
    es = series.ewm(span=slow, adjust=False).mean()
    line = ef - es
    sig = line.ewm(span=signal, adjust=False).mean()
    return line - sig

def atr(df, period=14):
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift()).abs()
    lc = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ---------- assets ----------
def ensure_assets():
    p = Path(ASSETS_FILE)
    if not p.exists():
        ex = [
            {"symbol":"EURUSD=X","display":"EUR/USD","payout":0.90},
            {"symbol":"GBPUSD=X","display":"GBP/USD","payout":0.88},
            {"symbol":"USDJPY=X","display":"USD/JPY","payout":0.86},
            {"symbol":"BTC-USD","display":"BTC/USD","payout":0.92},
            {"symbol":"ETH-USD","display":"ETH/USD","payout":0.91},
            {"symbol":"AUDUSD=X","display":"AUD/USD","payout":0.87}
        ]
        p.write_text(json.dumps(ex, ensure_ascii=False, indent=2))
    return json.loads(p.read_text(encoding='utf-8'))

# ---------- fetch with cache ----------
def fetch_ohlcv(symbol, interval):
    key = f"{symbol}_{interval}"
    c = cache_get(key)
    if c is not None:
        try: return pd.read_json(c).set_index("Datetime")
        except: pass
    try:
        df = yf.download(symbol, period="3d", interval=interval, progress=False, auto_adjust=True)
        if df is None or df.empty: return None
        js = df.reset_index().to_json(date_format='iso')
        cache_set(key, js)
        return df.reset_index().set_index("Datetime")
    except:
        return None

# ---------- analysis ----------
def analyze_one(symbol, use_15m):
    df5 = fetch_ohlcv(symbol, "5m")
    if df5 is None or df5.empty or len(df5) < 60: return None
    df5 = df5.tail(300).copy()

    atr_series = atr(df5)
    if atr_series.dropna().empty: return None
    last_atr = atr_series.iloc[-1]
    if pd.isna(last_atr) or last_atr == 0: return None

    ema50 = ema(df5["Close"], 50)
    ema200 = ema(df5["Close"], 200)
    if ema50.dropna().empty or ema200.dropna().empty: return None

    trend5 = "BUY" if ema50.iloc[-1] > ema200.iloc[-1] else "SELL"

    rsi5 = rsi(df5["Close"], 5)
    if rsi5.dropna().empty: return None
    last_rsi = rsi5.iloc[-1]

    macd5 = macd_hist(df5["Close"])
    if macd5.dropna().empty: return None
    last_macd = macd5.iloc[-1]

    last_price = df5["Close"].iloc[-1]

    support = df5["Low"].tail(60).min()
    resist = df5["High"].tail(60).max()

    near_s = abs(last_price - support) <= max(last_atr * 1.2, last_price * 0.0015)
    near_r = abs(last_price - resist) <= max(last_atr * 1.2, last_price * 0.0015)

    score = 20

    if trend5 == "BUY" and last_rsi < 55: score += 20
    if trend5 == "SELL" and last_rsi > 45: score += 20

    if trend5 == "BUY" and last_macd > 0: score += 20
    if trend5 == "SELL" and last_macd < 0: score += 20

    if trend5 == "BUY" and near_s: score += 20
    if trend5 == "SELL" and near_r: score += 20

    if not (last_atr >= last_price * 0.00018): return None

    strength = min(100, score)

    if use_15m:
        df15 = fetch_ohlcv(symbol, "15m")
        if df15 is None or df15.empty or len(df15) < 60: return None
        e50_15 = ema(df15["Close"], 50)
        e200_15 = ema(df15["Close"], 200)
        if e50_15.dropna().empty or e200_15.dropna().empty: return None
        trend15 = "BUY" if e50_15.iloc[-1] > e200_15.iloc[-1] else "SELL"
        if trend15 != trend5: return None

    return {
        "symbol": symbol,
        "trend": trend5,
        "price": round(last_price, 6),
        "strength": int(strength),
        "support": round(support, 6),
        "resistance": round(resist, 6)
    }

# ---------- Telegram commands ----------
@bot.message_handler(commands=["mode"])
def cmd_mode(msg):
    global MODE
    chat = msg.chat.id
    t = msg.text.lower().split()
    if len(t) == 2 and t[1] in THRESHOLDS:
        MODE = t[1]
        bot.send_message(chat, f"MODE set: {MODE}")
    else:
        bot.send_message(chat, f"Current mode: {MODE}")

@bot.message_handler(commands=["signal","scan"])
def cmd_signal(msg):
    chat = msg.chat.id
    print(f"DEBUG: Command signal received for chat {chat}, mode {MODE}.")
    bot.send_message(chat, f"🔎 Scanning ({MODE})...")
    assets = ensure_assets()
    cand = [a for a in assets if a.get("payout",0) >= PAYOUT_MIN][:MAX_ASSETS_PER_SCAN]

    print(f"DEBUG: Found {len(cand)} assets to scan.")
    
    if not cand:
        bot.send_message(chat, "No assets with enough payout.")
        return
    results = []
    for a in cand:
        sym = a['symbol']
        ck = f"ana_{sym}_{MODE}"
        pc = cache_get(ck)
        if pc: results.append(pc); continue

        print(f"DEBUG: Analyzing {sym}...")
        
        r = analyze_one(sym, THRESHOLDS[MODE]["USE_15M"])
        if r:
            print(f"DEBUG: Signal found for {sym} (Strength: {r['strength']})")
            r['display'] = a.get('display', sym)
            r['payout'] = a.get('payout',0)
            results.append(r)
            cache_set(ck, r)
        else:
            print(f"DEBUG: No strong signal for {sym}, skipping.")
            cache_set(ck, None)
        time.sleep(0.8)
    if not results:
        bot.send_message(chat, "❌ No strong signals now.")
        return
    results = sorted(results, key=lambda x: (x['strength'], x.get('payout',0)), reverse=True)

    out = []
    for r in results[:5]:
        out.append(
            f"📌 {r['display']} ({r['symbol']})\n"
            f"🔔 {r['trend']} | Strength {r['strength']}%\n"
            f"💰 Payout {int(r['payout']*100)}% | Price {r['price']}\n"
            f"🛑 S {r['support']} ▶ R {r['resistance']}\n"
            f"⏱ Expiry {EXPIRY_MIN} min\n—"
        )
    bot.send_message(chat, "\n\n".join(out))

@bot.message_handler(commands=["start","help"])
def cmd_help(msg):
    bot.send_message(msg.chat.id,
        "Signal tester (webhook).\n"
        "/signal – scan signals\n"
        "/mode <aggressive|conservative> — switch mode"
    )

# ---------- Webhook ----------
@app.route(f"/webhook", methods=["POST"])
def telegram_webhook():
    json_update = request.get_json(force=True)
    bot.process_new_updates([telebot.types.Update.de_json(json_update)])
    return "", 200

@app.route("/")
def home(): return "OK", 200

# ---------- RUN (Render) ----------
if __name__ == "__main__":
    bot.delete_webhook()
    bot.set_webhook(url=WEBHOOK_URL)
    app.run(host="0.0.0.0", port=10000)

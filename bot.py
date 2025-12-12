# bot.py — Final stable version for Render (long-polling, nice signals UA)
import os
import time
import threading
import logging
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler

# ---------------- CONFIG (from Render environment) ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
PORT = int(os.getenv("PORT", "10000"))
SIGNAL_INTERVAL = int(os.getenv("SIGNAL_INTERVAL", "60"))  # seconds between market scans
MIN_STRENGTH = int(os.getenv("MIN_STRENGTH", "75"))        # minimal score to send signal
# ------------------------------------------------------------------

if not TELEGRAM_TOKEN:
    raise SystemExit("ERROR: TELEGRAM_TOKEN not set in environment variables")

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
SEND_MESSAGE_URL = TELEGRAM_API + "/sendMessage"
GET_UPDATES_URL = TELEGRAM_API + "/getUpdates"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("po-signal-bot")

SUBSCRIBERS = set()
LAST_UPDATE_ID = None

# ---------- Minimal HTTP server so Render sees an open port ----------
class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"Bot is running\n")
    def log_message(self, format, *args):
        return

def run_http_server():
    server = HTTPServer(("0.0.0.0", PORT), SimpleHandler)
    logger.info(f"HTTP server listening on port {PORT}")
    server.serve_forever()

# ---------------- Telegram helpers ----------------
def send_message(chat_id, text):
    try:
        r = requests.post(SEND_MESSAGE_URL, json={
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": True
        }, timeout=10)
        if not r.ok:
            logger.warning("send_message failed: %s %s", r.status_code, r.text)
        return r.ok
    except Exception:
        logger.exception("send_message exception")
        return False

# robust getUpdates loop (avoids webhook, avoids 409 crash)
def process_updates_loop():
    global LAST_UPDATE_ID
    params = {"timeout": 20, "limit": 50}
    while True:
        try:
            if LAST_UPDATE_ID is not None:
                params["offset"] = LAST_UPDATE_ID + 1

            r = requests.get(GET_UPDATES_URL, params=params, timeout=30)

            if not r.ok:
                if r.status_code == 409:
                    logger.error("409 conflict — another getUpdates is running elsewhere. Stop other instances.")
                    time.sleep(10)
                else:
                    time.sleep(5)
                continue

            data = r.json()
            for upd in data.get("result", []):
                LAST_UPDATE_ID = upd["update_id"]
                msg = upd.get("message") or upd.get("edited_message")
                if not msg:
                    continue

                chat_id = msg["chat"]["id"]
                text = (msg.get("text") or "").strip()

                logger.info("Received message from %s: %s", chat_id, text)

                if text == "/start":
                    SUBSCRIBERS.add(chat_id)
                    send_message(chat_id, "Привіт! Тепер ти підписаний(а) на сигнали. Напиши /stop щоб відписатися.")
                elif text == "/stop":
                    SUBSCRIBERS.discard(chat_id)
                    send_message(chat_id, "Ти відписаний(а).")
                elif text == "/subs":
                    send_message(chat_id, f"Підписників: {len(SUBSCRIBERS)}")
                elif text == "/help":
                    send_message(chat_id, "Команди: /start, /stop, /subs.")

        except Exception:
            logger.exception("process_updates crashed, retry in 5s")
            time.sleep(5)

# ---------------- Indicators ----------------
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd_hist(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd - macd_signal

# ---------------- Market analysis ----------------
ASSETS = [
    ("EURUSD=X", "EUR/USD"),
    ("GBPUSD=X", "GBP/USD"),
    ("USDJPY=X", "USD/JPY"),
    ("AUDUSD=X", "AUD/USD"),
    ("BTC-USD", "BTC/USD")
]

def analyze_symbol(symbol):
    try:
        # -------- 5m data --------
        df = yf.download(symbol, period="3d", interval="5m", progress=False)
        if df is None or df.empty or len(df) < 80:
            return None

        close = df["Close"].dropna()
        if len(close) < 50:
            return None

        ema50 = ema(close, 50)
        ema200 = ema(close, 200)
        rsi5 = rsi(close)
        macd5 = macd_hist(close)

        # convert to numeric scalars
        ema50_last = float(ema50.iloc[-1])
        ema200_last = float(ema200.iloc[-1])
        last_rsi = float(rsi5.iloc[-1])
        last_macd = float(macd5.iloc[-1])
        last_price = float(close.iloc[-1])

        if pd.isna(ema200_last) or pd.isna(ema50_last):
            return None

        # trend
        trend = "BUY" if ema50_last > ema200_last else "SELL"

        # ATR approx
        atr = (df["High"] - df["Low"]).rolling(14).mean()
        if pd.isna(atr.iloc[-1]):
            return None
        last_atr = float(atr.iloc[-1])

        # support / resistance
        recent = df[-120:]
        support = float(recent["Low"].min())
        resistance = float(recent["High"].max())

        near_support = abs(last_price - support) <= max(last_atr * 1.2, last_price * 0.0015)
        near_resistance = abs(last_price - resistance) <= max(last_atr * 1.2, last_price * 0.0015)

        # scoring
        score = 0
        reasons = []

        score += 20; reasons.append("Trend")

        if trend == "BUY" and last_rsi < 55:
            score += 25; reasons.append(f"RSI{int(last_rsi)}")
        if trend == "SELL" and last_rsi > 45:
            score += 25; reasons.append(f"RSI{int(last_rsi)}")

        if trend == "BUY" and last_macd > 0:
            score += 20; reasons.append("MACD+")
        if trend == "SELL" and last_macd < 0:
            score += 20; reasons.append("MACD-")

        if (trend == "BUY" and near_support) or (trend == "SELL" and near_resistance):
            score += 15; reasons.append("NearSR")

        strength = min(100, int(score))

        # -------- 15m confirmation --------
        df15 = yf.download(symbol, period="7d", interval="15m", progress=False)
        if df15 is None or df15.empty or len(df15) < 50:
            return None

        close15 = df15["Close"].dropna()
        ema50_15 = float(close15.ewm(span=50).mean().iloc[-1])
        ema200_15 = float(close15.ewm(span=200).mean().iloc[-1])

        trend15 = "BUY" if ema50_15 > ema200_15 else "SELL"
        if trend15 != trend:
            return None

        return {
            "symbol": symbol,
            "price": round(last_price, 6),
            "trend": trend,
            "strength": strength,
            "reasons": reasons,
            "support": round(support, 6),
            "resistance": round(resistance, 6)
        }

    except Exception:
        logger.exception("analyze_symbol error")
        return None

# ---------------- Message builder ----------------
def build_message(item, display):
    verb = "Купити" if item["trend"] == "BUY" else "Продати"
    return (
        f"📌 Пара: {display}\n"
        f"🔔 Сигнал: {verb}\n\n"
        f"🔥 Потужність: {item['strength']}%\n"
        f"⏱️ Експірація: 3 хв\n\n"
        f"💵 Ціна: {item['price']}\n"
        f"📈 Підтвердження: {', '.join(item['reasons'])}\n"
        f"🛑 S: {item['support']}  ▶️ R: {item['resistance']}\n"
    )

# ---------------- Signal worker ----------------
def signal_worker():
    logger.info("Signal worker started")
    while True:
        try:
            if not SUBSCRIBERS:
                time.sleep(SIGNAL_INTERVAL)
                continue

            results = []
            for sym, disp in ASSETS:
                res = analyze_symbol(sym)
                if res and res["strength"] >= MIN_STRENGTH:
                    results.append((res, disp))

            if results:
                results.sort(key=lambda x: x[0]["strength"], reverse=True)
                for res, disp in results[:3]:
                    text = build_message(res, disp)
                    for chat_id in list(SUBSCRIBERS):
                        send_message(chat_id, text)
                        time.sleep(0.3)
            else:
                logger.info("No strong signals this cycle")

        except Exception:
            logger.exception("signal_worker crashed")

        time.sleep(SIGNAL_INTERVAL)

# ---------------- Main ----------------
if __name__ == "__main__":
    threading.Thread(target=run_http_server, daemon=True).start()
    threading.Thread(target=process_updates_loop, daemon=True).start()
    signal_worker()

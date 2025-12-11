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
MIN_STRENGTH = int(os.getenv("MIN_STRENGTH", "75"))       # minimal score to send signal
# -----------------------------------------------------------------

if not TELEGRAM_TOKEN:
    raise SystemExit("ERROR: TELEGRAM_TOKEN not set in environment variables")

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
SEND_MESSAGE_URL = TELEGRAM_API + "/sendMessage"
GET_UPDATES_URL = TELEGRAM_API + "/getUpdates"
GET_ME_URL = TELEGRAM_API + "/getMe"
GET_WEBHOOK_INFO = TELEGRAM_API + "/getWebhookInfo"

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

# robust getUpdates handler (long-polling)
def process_updates_loop():
    global LAST_UPDATE_ID
    params = {"timeout": 20, "limit": 50}
    backoff = 1
    while True:
        try:
            if LAST_UPDATE_ID is not None:
                params["offset"] = LAST_UPDATE_ID + 1
            r = requests.get(GET_UPDATES_URL, params=params, timeout=30)
            if not r.ok:
                txt = r.text if r is not None else ""
                logger.warning("getUpdates failed: %s %s", r.status_code, txt)
                # handle 409 Conflict specifically
                if r.status_code == 409:
                    logger.error("409 conflict — another getUpdates is running elsewhere. Stop other instances and restart this service.")
                    time.sleep(10)  # wait and retry
                else:
                    time.sleep(min(backoff, 30))
                    backoff = min(backoff * 2, 30)
                continue
            backoff = 1
            data = r.json()
            for upd in data.get("result", []):
                LAST_UPDATE_ID = upd["update_id"]
                msg = upd.get("message") or upd.get("edited_message")
                if not msg:
                    continue
                chat = msg.get("chat", {})
                chat_id = chat.get("id")
                text = (msg.get("text") or "").strip()
                if not text or chat_id is None:
                    continue
                logger.info("Received message from %s: %s", chat_id, text)
                # commands
                if text.startswith("/start"):
                    SUBSCRIBERS.add(chat_id)
                    send_message(chat_id, "Привіт! Тепер ти підписаний(а) на сигнали. Напиши /stop щоб відписатися.")
                elif text.startswith("/stop"):
                    if chat_id in SUBSCRIBERS:
                        SUBSCRIBERS.remove(chat_id)
                    send_message(chat_id, "Ти відписаний(а).")
                elif text.startswith("/subs"):
                    send_message(chat_id, f"Підписників: {len(SUBSCRIBERS)}")
                elif text.startswith("/help"):
                    send_message(chat_id, "Команди: /start підписатися, /stop відписатися, /subs кількість підписників.")
        except requests.exceptions.ReadTimeout:
            # normal for long polling
            continue
        except Exception:
            logger.exception("process_updates crashed, retrying in 5s")
            time.sleep(5)

# ---------------- Indicators (no external C libs) ----------------
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
        df5 = yf.download(symbol, period="3d", interval="5m", progress=False)
        if df5 is None or df5.empty or len(df5) < 60:
            return None
        close5 = df5["Close"].dropna()
        ema50 = ema(close5, 50)
        ema200 = ema(close5, 200)
        rsi5 = rsi(close5)
        macd5 = macd_hist(close5)

        # last numeric values (use iloc to avoid ambiguity)
        if len(close5) < 1:
            return None
        last_close = float(close5.iloc[-1])

        if pd.isna(ema200.iloc[-1].item()):
            return None

        trend = "BUY" if ema50.iloc[-1] > ema200.iloc[-1] else "SELL"
        last_rsi = float(rsi5.iloc[-1])
        last_macd = float(macd5.iloc[-1])

        # ATR approx
        high_low = df5["High"] - df5["Low"]
        atr = high_low.rolling(14).mean()
        if pd.isna(atr.iloc[-1]) or atr.iloc[-1] == 0:
            return None
        last_atr = float(atr.iloc[-1])

        # support/resistance
        window = min(len(df5), 120)
        recent = df5[-window:]
        support = float(recent["Low"].min())
        resistance = float(recent["High"].max())
        near_support = abs(last_close - support) <= max(last_atr * 1.2, last_close * 0.0015)
        near_resistance = abs(last_close - resistance) <= max(last_atr * 1.2, last_close * 0.0015)

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

        # 15m confirmation
        df15 = yf.download(symbol, period="7d", interval="15m", progress=False)
        if df15 is None or df15.empty or len(df15) < 30:
            return None
        close15 = df15["Close"].dropna()
        ema50_15 = ema(close15, 50)
        ema200_15 = ema(close15, 200)
        if pd.isna(ema200_15.iloc[-1]):
            return None
        trend15 = "BUY" if ema50_15.iloc[-1] > ema200_15.iloc[-1] else "SELL"
        if trend15 != trend:
            return None

        return {
            "symbol": symbol,
            "price": round(last_close, 6),
            "trend": trend,
            "strength": strength,
            "reasons": reasons,
            "support": round(support, 6),
            "resistance": round(resistance, 6)
        }
    except Exception:
        logger.exception("analyze_symbol error")
        return None

def build_message(item, display):
    verb = "Купити" if item["trend"] == "BUY" else "Продати"
    text = (
        f"📌 Пара: {display}\n"
        f"🔔 Сигнал: {verb}\n\n"
        f"🔥 Потужність: {item['strength']}%\n"
        f"⏱️ Експірація: 3 хв\n\n"
        f"💵 Ціна: {item['price']}\n"
        f"📈 Підтвердження: {', '.join(item['reasons'])}\n"
        f"🛑 S: {item['support']}  ▶️ R: {item['resistance']}\n"
    )
    return text

# ---------------- Signal worker ----------------
def signal_worker():
    logger.info("Signal worker started")
    while True:
        try:
            if not SUBSCRIBERS:
                logger.debug("No subscribers, skipping scan")
                time.sleep(SIGNAL_INTERVAL)
                continue

            results = []
            for sym, disp in ASSETS:
                res = analyze_symbol(sym)
                if res and res["strength"] >= MIN_STRENGTH:
                    results.append((res, disp))

            if results:
                results.sort(key=lambda x: x[0]["strength"], reverse=True)
                to_send = results[:3]
                for (res, disp) in to_send:
                    text = build_message(res, disp)
                    for chat in list(SUBSCRIBERS):
                        send_message(chat, text)
                        time.sleep(0.3)
            else:
                logger.info("No strong signals this cycle")
        except Exception:
            logger.exception("signal_worker crashed")
        time.sleep(SIGNAL_INTERVAL)

# ---------------- Main start ----------------
if __name__ == "__main__":
    # 1) start HTTP server thread so Render sees open port
    t_http = threading.Thread(target=run_http_server, daemon=True)
    t_http.start()

    # 2) start updates polling thread
    t_upd = threading.Thread(target=process_updates_loop, daemon=True)
    t_upd.start()

    # 3) run signal loop in main thread
    signal_worker()

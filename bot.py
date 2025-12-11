# bot.py
import os
import time
import threading
import logging
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler

# ----- Налаштування (через Render ENV) -----
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")  # постав у Render -> Environment
PORT = int(os.getenv("PORT", "10000"))        # Render задає PORT автоматично
POLL_INTERVAL = 1      # як часто перевіряти Telegram updates (сек)
SIGNAL_INTERVAL = 60   # як часто аналізувати ринок (сек)
# --------------------------------------------

if not TELEGRAM_TOKEN:
    raise SystemExit("ERROR: TELEGRAM_TOKEN not set in environment variables")

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
SEND_MESSAGE_URL = TELEGRAM_API + "/sendMessage"
GET_UPDATES_URL = TELEGRAM_API + "/getUpdates"
SUBSCRIBERS = set()  # chat_id підписників
LAST_UPDATE_ID = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("po-signal-bot")


# ---------- Minimal HTTP server so Render sees an open port ----------
class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"Bot is running\n")
    def log_message(self, format, *args):
        return  # silence


def run_http_server():
    server = HTTPServer(("0.0.0.0", PORT), SimpleHandler)
    logger.info(f"HTTP server listening on port {PORT}")
    server.serve_forever()


# ---------- Telegram helpers ----------
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
    except Exception as e:
        logger.exception("send_message exception")
        return False


def process_updates():
    """
    Long-polling getUpdates; adds chat_id to SUBSCRIBERS on /start
    """
    global LAST_UPDATE_ID
    params = {"timeout": 20, "limit": 50}
    while True:
        try:
            if LAST_UPDATE_ID:
                params["offset"] = LAST_UPDATE_ID + 1
            r = requests.get(GET_UPDATES_URL, params=params, timeout=30)
            if r.ok:
                data = r.json()
                for upd in data.get("result", []):
                    LAST_UPDATE_ID = upd["update_id"]
                    # message handling
                    msg = upd.get("message") or upd.get("edited_message")
                    if not msg:
                        continue
                    chat = msg.get("chat", {})
                    chat_id = chat.get("id")
                    text = msg.get("text", "").strip()
                    if not text or not chat_id:
                        continue
                    logger.info("Received message from %s: %s", chat_id, text)
                    # simple commands
                    if text.startswith("/start"):
                        SUBSCRIBERS.add(chat_id)
                        send_message(chat_id, "Привіт! Ти підписаний на сигнали. /stop — відписатись.")
                    elif text.startswith("/stop"):
                        if chat_id in SUBSCRIBERS:
                            SUBSCRIBERS.remove(chat_id)
                        send_message(chat_id, "Ти відписаний.")
                    elif text.startswith("/subs"):
                        send_message(chat_id, f"Підписники: {len(SUBSCRIBERS)}")
                    # you can expand more commands here
            else:
                logger.warning("getUpdates failed: %s %s", r.status_code, r.text)
                time.sleep(2)
        except requests.exceptions.ReadTimeout:
            # normal for long polling
            continue
        except Exception:
            logger.exception("process_updates crashed, retrying in 5s")
            time.sleep(5)


# ---------- Indicators (simple, no talib) ----------
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

# ---------- Market analysis ----------
ASSETS = [
    ("EURUSD=X", "EUR/USD"),
    ("GBPUSD=X", "GBP/USD"),
    ("USDJPY=X", "USD/JPY"),
    ("AUDUSD=X", "AUD/USD"),
    ("BTC-USD", "BTC/USD")
]

def analyze_symbol(symbol):
    """
    Analyze symbol on 5m (base) and confirm on 15m.
    Returns dict or None.
    """
    try:
        df5 = yf.download(symbol, period="3d", interval="5m", progress=False)
        if df5 is None or df5.empty or len(df5) < 60:
            return None
        close5 = df5["Close"].dropna()
        # indicators 5m
        ema50 = ema(close5, 50)
        ema200 = ema(close5, 200)
        rsi5 = rsi(close5)
        macd5 = macd_hist(close5)

        last_close = float(close5.iat[-1])
        # ensure numeric
        if pd.isna(ema200.iat[-1]):
            return None

        trend = "BUY" if ema50.iat[-1] > ema200.iat[-1] else "SELL"
        last_rsi = float(rsi5.iat[-1])
        last_macd = float(macd5.iat[-1])

        # simple volatility check via ATR approximation
        high_low = df5["High"] - df5["Low"]
        atr = high_low.rolling(14).mean()
        last_atr = float(atr.iat[-1]) if not pd.isna(atr.iat[-1]) else 0.0
        if last_atr == 0:
            return None

        # near support/resistance
        window = 120
        recent = df5[-window:]
        support = float(recent["Low"].min())
        resistance = float(recent["High"].max())
        near_support = abs(last_close - support) <= max(last_atr * 1.2, last_close * 0.0015)
        near_resistance = abs(last_close - resistance) <= max(last_atr * 1.2, last_close * 0.0015)

        # scoring
        score = 0
        reasons = []
        # trend baseline
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
        if pd.isna(ema200_15.iat[-1]):
            return None
        trend15 = "BUY" if ema50_15.iat[-1] > ema200_15.iat[-1] else "SELL"
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
        f"📌 {display} ({item['symbol']})\n"
        f"🔔 {verb}  |  Сила: {item['strength']}%\n"
        f"💵 Ціна: {item['price']}\n"
        f"📈 Підтвердження: {', '.join(item['reasons'])}\n"
        f"🛑 S: {item['support']}  ▶ R: {item['resistance']}\n"
        f"⏱️ Expiry (info): 3 хв"
    )
    return text


def signal_worker():
    """
    Main loop to scan assets and send messages to subscribers.
    """
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
                if res and res["strength"] >= 75:  # базовий поріг; можна зміняти
                    results.append((res, disp))

            if results:
                # sort by strength desc
                results.sort(key=lambda x: x[0]["strength"], reverse=True)
                # send top 3
                to_send = results[:3]
                for (res, disp) in to_send:
                    text = build_message(res, disp)
                    for chat in list(SUBSCRIBERS):
                        send_message(chat, text)
                        time.sleep(0.3)  # avoid flood
            else:
                logger.info("No strong signals this cycle")
        except Exception:
            logger.exception("signal_worker crashed")

        time.sleep(SIGNAL_INTERVAL)


# ---------- Start threads ----------
if __name__ == "__main__":
    # start HTTP server thread (so Render sees open port)
    t_http = threading.Thread(target=run_http_server, daemon=True)
    t_http.start()

    # start telegram updates polling thread
    t_upd = threading.Thread(target=process_updates, daemon=True)
    t_upd.start()

    # start signal worker (in main thread)
    signal_worker()

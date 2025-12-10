import time
import requests
import numpy as np
from telegram import Bot
import talib
import os

TOKEN = os.getenv("8517986396:AAENPrASLsQlLu21BxG-jKIYZEaEL-RKxYs")
CHAT_ID = os.getenv("477570593")

bot = Bot(token=TOKEN)

# ========= TRADINGVIEW DATA =========
def get_tv_ohlc(symbol="EURUSD", interval="5"):
    url = f"https://api.tradingview.com/markets/forex/quotes/{symbol}?interval={interval}"
    r = requests.get(url).json()

    candles = r["candles"]

    o = np.array([c[1] for c in candles], dtype=float)
    h = np.array([c[2] for c in candles], dtype=float)
    l = np.array([c[3] for c in candles], dtype=float)
    c = np.array([c[4] for c in candles], dtype=float)

    return o, h, l, c


# ========= SIGNAL LOGIC =========
def generate_signal(prices):
    o, h, l, c = prices

    rsi = talib.RSI(c, timeperiod=14)
    ma50 = talib.MA(c, timeperiod=50)
    ma200 = talib.MA(c, timeperiod=200)

    last_rsi = rsi[-1]
    last_ma50 = ma50[-1]
    last_ma200 = ma200[-1]
    last_close = c[-1]
    prev_close = c[-2]

    up_trend = last_close > last_ma50 and last_close > last_ma200
    down_trend = last_close < last_ma50 and last_close < last_ma200

    # BUY (КУПИТИ)
    if last_rsi < 30 and up_trend and last_close > prev_close:
        return "КУПИТИ"

    # SELL (ПРОДАТИ)
    if last_rsi > 70 and down_trend and last_close < prev_close:
        return "ПРОДАТИ"

    return None


# ========= MAIN LOOP =========
def run():
    bot.send_message(chat_id=CHAT_ID, text="🚀 Бот запущено. Аналізую ринок...")

    while True:
        try:
            # M5
            m5 = get_tv_ohlc(interval="5")
            signal_m5 = generate_signal(m5)

            # M15
            m15 = get_tv_ohlc(interval="15")
            signal_m15 = generate_signal(m15)

            if signal_m5:
                bot.send_message(
                    chat_id=CHAT_ID,
                    text=f"📊 М5 Сигнал: {signal_m5}"
                )

            if signal_m15:
                bot.send_message(
                    chat_id=CHAT_ID,
                    text=f"⏱ М15 Сигнал: {signal_m15}"
                )

        except Exception as e:
            bot.send_message(chat_id=CHAT_ID, text=f"⚠️ Помилка: {e}")

        time.sleep(40)  # інтервал між сигналами


if __name__ == "__main__":
    run()

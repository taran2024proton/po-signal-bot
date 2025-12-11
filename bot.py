import os
import asyncio
import logging
import yfinance as yf
import pandas as pd
from telegram import Bot
from flask import Flask

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

logging.basicConfig(level=logging.INFO)

# --- INDICATORS --- #

def rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def moving_average(prices, period=20):
    return prices.rolling(period).mean()

# --- SIGNAL LOGIC --- #

async def check_signal():
    pairs = {
        "EURUSD=X": "EUR/USD",
        "GBPUSD=X": "GBP/USD",
        "USDJPY=X": "USD/JPY",
        "AUDUSD=X": "AUD/USD",
    }

    bot = Bot(token=TELEGRAM_TOKEN)

    for ticker, name in pairs.items():
        data = yf.download(ticker, interval="5m", period="2d")

        if len(data) < 50:
            continue

        close = data["Close"]

        rsi_val = float(rsi(close).iloc[-1])
        ma20 = float(moving_average(close).iloc[-1])
        price = float(close.iloc[-1])

        direction = None

        if rsi_val < 30 and price > ma20:
            direction = "BUY (купити)"
        elif rsi_val > 70 and price < ma20:
            direction = "SELL (продати)"

        if direction:
            text = (
                f"📌 {name}\n"
                f"🔔 Сигнал: {direction}\n"
                f"💹 RSI: {round(rsi_val,2)}\n"
                f"📈 MA20: {round(ma20,5)}\n"
                f"💰 Ціна: {price}\n"
                f"🕒 Таймфрейм: 5 хв"
            )
            await bot.send_message(chat_id=CHAT_ID, text=text)

async def signal_loop():
    while True:
        await check_signal()
        await asyncio.sleep(60)

# --- SIMPLE FLASK SERVER FOR RENDER --- #

app = Flask(__name__)

@app.route("/")
def home():
    return "Bot is running!"

def start_server():
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

# --- MAIN --- #

async def main():
    # запускаємо бота окремим асинхронним процесом
    asyncio.create_task(signal_loop())

    # запускаємо Flask у паралельному потоці
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, start_server)

if __name__ == "__main__":
    asyncio.run(main())

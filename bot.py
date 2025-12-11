import os
import asyncio
import logging
import yfinance as yf
import pandas as pd
from telegram import Bot

# -------------------------------------------
# LOAD TOKENS FROM RENDER ENVIRONMENT
# -------------------------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

logging.basicConfig(level=logging.INFO)

# -------------------------------------------
# SIMPLE RSI (without TA-LIB)
# -------------------------------------------
def rsi(prices, period=14):
    delta = prices.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# -------------------------------------------
# SIMPLE MOVING AVERAGE
# -------------------------------------------
def moving_average(prices, period=20):
    return prices.rolling(period).mean()

# -------------------------------------------
# SIGNAL LOGIC
# -------------------------------------------
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

        # Need enough data
        if len(data) < 50:
            continue

        close = data["Close"].copy()

        # Calculate indicators
        rsi_series = rsi(close)
        ma20_series = moving_average(close)

        rsi_val = float(rsi_series.iloc[-1])
        ma20 = float(ma20_series.iloc[-1])
        price = float(close.iloc[-1])

        direction = None

        # ---------- SIGNAL CONDITIONS ----------
        if rsi_val < 30 and price > ma20:
            direction = "BUY (купити)"
        elif rsi_val > 70 and price < ma20:
            direction = "SELL (продати)"

        # ---------- SEND SIGNAL ----------
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

# -------------------------------------------
# MAIN LOOP
# -------------------------------------------
async def main():
    while True:
        await check_signal()
        await asyncio.sleep(60)

# -------------------------------------------
# RUN BOT
# -------------------------------------------
if __name__ == "__main__":
    asyncio.run(main())

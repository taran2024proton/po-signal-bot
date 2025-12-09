import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import requests
import random

TELEGRAM_TOKEN = "8517986396:AAENPrASLsQlLu21BxG-jKIYZEaEL-RKxYs"

logging.basicConfig(level=logging.INFO)

# --- Команда /start ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Бот працює! Надішли /signal щоб отримати сигнал 📈")

# --- Генерація тестового сигналу ---
async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "EUR/JPY"]

    pair = random.choice(pairs)
    direction = random.choice(["BUY", "SELL"])
    payout = random.randint(85, 95)
    strength = random.randint(70, 99)

    text = f"""
📌 Пара: {pair}
🔔 Сигнал: {direction}
💰 Виплата: {payout}%
🔥 Потужність: {strength}%
⏱ Експірація: 3 хв
"""

    await update.message.reply_text(text)

# --- Головний запуск ---
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signal", signal))

    app.run_polling()

if __name__ == "__main__":
    main()

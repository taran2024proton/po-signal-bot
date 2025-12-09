import os
from telegram.ext import Updater, CommandHandler

TELEGRAM_TOKEN = "PUT_YOUR_TOKEN_HERE"

def start(update, context):
    update.message.reply_text("Привіт! Напиши /scan щоб отримати сигнали.")

def scan(update, context):
    update.message.reply_text("Поки що тестовий сигнал ❤️ BUY EUR/USD 5 хв.")

def main():
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("scan", scan))
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()

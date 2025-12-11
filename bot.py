import logging
import yfinance as yf
import pandas as pd
from telegram.ext import Updater, CommandHandler

# =============== LOGGING ===============
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# =============== ANALYTICS ===============
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def moving_average(series, window=20):
    return series.rolling(window).mean()

# =============== BOT LOGIC ===============
def start(update, context):
    update.message.reply_text(
        "Привіт! Введи тикер, наприклад:\n\n/btc\n/tsla\n/aapl"
    )

def analyze_ticker(update, context):
    ticker = update.message.text.replace("/", "").upper()
    update.message.reply_text(f"🔍 Аналізую {ticker}...")

    data = yf.download(ticker, period="3mo", interval="1d")

    if data.empty:
        update.message.reply_text("❌ Не вдалося знайти такі дані.")
        return

    close = data["Close"]

    rsi_val = rsi(close).iloc[-1]
    ma20 = moving_average(close).iloc[-1]
    price = close.iloc[-1]

    update.message.reply_text(
        f"📊 *{ticker}*\n"
        f"Ціна: {price:.2f}\n"
        f"RSI: {rsi_val:.2f}\n"
        f"MA20: {ma20:.2f}",
        parse_mode="Markdown"
    )

# =============== MAIN ===============
def main():
    TOKEN = "ТУТ_ТВІЙ_TOKEN"
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("btc", analyze_ticker))
    dp.add_handler(CommandHandler("tsla", analyze_ticker))
    dp.add_handler(CommandHandler("aapl", analyze_ticker))

    # endless bot polling (works on Render)
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()

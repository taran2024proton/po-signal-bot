import telebot
import yfinance as yf
import numpy as np
import time

TOKEN = 8517986396:AAENPrASLsQlLu21BxG-jKIYZEaEL-RKxYs

bot = telebot.TeleBot(TOKEN)

# --- ІНДИКАТОРИ ---

def rsi(close, period=14):
    delta = np.diff(close)
    up = delta.clip(min=0)
    down = -1 * delta.clip(max=0)
    ma_up = np.convolve(up, np.ones(period)/period, mode='valid')
    ma_down = np.convolve(down, np.ones(period)/period, mode='valid')
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def ma(close, period=20):
    return np.convolve(close, np.ones(period)/period, mode='valid')

# --- АНАЛІЗ ПАРИ ---

def analyze_pair(symbol):
    data = yf.download(symbol, period="1d", interval="5m")
    if len(data) < 30:
        return None

    close = data["Close"].values

    rsi_val = rsi(close)[-1]
    ma20 = ma(close, 20)[-1]
    last_price = close[-1]

    # Рівні підтримки/опору
    support = np.min(close[-20:])
    resistance = np.max(close[-20:])

    signal = None

    if rsi_val < 30 and last_price <= support:
        signal = "BUY"
    if rsi_val > 70 and last_price >= resistance:
        signal = "SELL"

    return {
        "symbol": symbol,
        "rsi": round(rsi_val, 2),
        "ma20": round(ma20, 5),
        "support": round(support, 5),
        "resistance": round(resistance, 5),
        "signal": signal
    }

# Список пар (такі ж як на Pocket Option)
pairs = [
    "EURUSD=X", "GBPJPY=X", "AUDUSD=X", "USDJPY=X",
    "EURJPY=X", "NZDUSD=X", "GBPUSD=X", "USDCAD=X",
    "EURGBP=X", "AUDJPY=X"
]

# --- КОМАНДА /signal ---

@bot.message_handler(commands=["signal"])
def send_signal(message):
    bot.reply_to(message, "🔍 Аналізую ринок... зачекай 3–5 секунд...")

    best = None

    for p in pairs:
        res = analyze_pair(p)
        if res and res["signal"]:
            best = res
            break

    if not best:
        bot.send_message(message.chat.id, "❌ Немає сильного сигналу зараз. Спробуй ще раз.")
        return

    text = f"""
📌 **Пара:** {best['symbol']}
📊 **Сигнал:** {best['signal']}
📈 **RSI:** {best['rsi']}
📉 **MA20:** {best['ma20']}
🛑 **Підтримка:** {best['support']}
🟩 **Опір:** {best['resistance']}
⏳ **Експірація:** 5 хвилин
    """

    bot.send_message(message.chat.id, text, parse_mode="Markdown")

bot.polling()

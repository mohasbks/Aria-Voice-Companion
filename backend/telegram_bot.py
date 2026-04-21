"""
telegram_bot.py – Aria on Telegram

Provides a Telegram interface to the Aria AI companion.
Features:
- /start: Welcome message
- /lang: Toggle between English and Arabic
- Text messages: Sends text to Llama, generates voice via Orpheus, sends audio back
- Handles API key loading from `.env`
"""

import os
import io
import asyncio
import logging
from pathlib import Path

# Fix for python 3.13 apscheduler timezone issue
import pytz
import apscheduler.util
apscheduler.util.get_localzone = lambda: pytz.utc
apscheduler.util.astimezone = lambda tz: pytz.utc

# Load .env
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from telegram import Update, BotCommand
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# Aria core logic
from llm import get_llm_response
from tts import synthesize_speech_stream
from memory import init_db, get_recent_messages, get_emotion_arc, get_session_mood, save_message
from text_processor import normalize_text

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_KEY_ARABIC = os.getenv("GROQ_API_KEY_ARABIC")

if not TELEGRAM_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN not set in .env")
    exit(1)

# Ensure DB is ready
init_db()


async def set_commands(application):
    """Set bot commands for the menu"""
    commands = [
        BotCommand("start", "Start conversation"),
        BotCommand("lang", "Toggle Language (AR/EN)"),
        BotCommand("clear", "Clear conversation history")
    ]
    await application.bot.set_my_commands(commands)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    session_id = str(update.message.chat_id)
    # Default language is English
    context.user_data["lang"] = "en"
    
    welcome_text = (
        "Hi! I'm Aria, your voice companion. 🎙️\n\n"
        "Send me a message, and I'll reply with a voice note.\n"
        "Use /lang to switch between English and Arabic (Saudi)."
    )
    await update.message.reply_text(welcome_text)


async def toggle_lang(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /lang command"""
    current_lang = context.user_data.get("lang", "en")
    new_lang = "ar" if current_lang == "en" else "en"
    context.user_data["lang"] = new_lang
    
    if new_lang == "ar":
        await update.message.reply_text("✅ تم تغيير اللغة إلى: العربية (السعودية).")
    else:
        await update.message.reply_text("✅ Language changed to: English.")


async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clear memory by changing the session ID (internally tracking chat_id_timestamp)"""
    import time
    context.user_data["session_suffix"] = str(int(time.time()))
    await update.message.reply_text("🧹 Conversation cleared." if context.user_data.get("lang") == "en" else "🧹 تم مسح المحادثة.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Process incoming text message from Telegram"""
    user_text = update.message.text
    chat_id = update.message.chat_id
    
    # Telegram chats use a combination of chat_id and an optional suffix for fresh sessions
    suffix = context.user_data.get("session_suffix", "")
    session_id = f"tg_{chat_id}_{suffix}"
    
    lang = context.user_data.get("lang", "en")
    engine_key = GROQ_API_KEY_ARABIC if lang == "ar" else GROQ_API_KEY
    
    if not engine_key:
        await update.message.reply_text(
            f"❌ API key not configured for language: {lang}"
        )
        return

    # Show "typing..." indicator
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    # 1. Load context
    history      = get_recent_messages(n=6, session_id=session_id)
    emotion_arc  = get_emotion_arc(n=4,  session_id=session_id)
    session_mood = get_session_mood(n=4,  session_id=session_id)

    # 2. Get LLM Response
    result = await get_llm_response(
        user_message=user_text,
        history=history,
        emotion_arc=emotion_arc,
        session_mood=session_mood,
        lang=lang,
        engine_key=engine_key,
    )
    
    ai_text = result["text"]
    emotion = result["emotion"]

    # 3. Save to memory
    save_message("user", user_text, emotion="neutral", session_id=session_id)
    save_message("assistant", ai_text, emotion=emotion, session_id=session_id)

    # 4. Generate Voice
    # Show "recording audio..." indicator
    await context.bot.send_chat_action(chat_id=chat_id, action="record_voice")
    
    # Collect all audio chunks
    audio_buffer = bytearray()
    async for chunk in synthesize_speech_stream(text=ai_text, emotion=emotion, lang=lang, engine_key=engine_key):
        audio_buffer.extend(chunk)
        
    if not audio_buffer:
        # TTS failed (maybe rate limit)
        await update.message.reply_text(
            f"*(Aria - {emotion})*\n\n{ai_text}\n\n_(Audio generation failed)_", 
            parse_mode="Markdown"
        )
        return
        
    # 5. Send Voice Message
    # Convert bytearray to BytesIO for python-telegram-bot
    audio_io = io.BytesIO(audio_buffer)
    audio_io.name = "aria_response.ogg" # Telegram prefers ogg/opus, but will convert mp3/wav
    
    caption = f"🫧 {emotion}"
    await update.message.reply_voice(voice=audio_io, caption=caption)


def main():
    """Start the bot"""
    # Disable job_queue to prevent APScheduler timezone conflicts
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).job_queue(None).build()

    # Commands
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("lang", toggle_lang))
    application.add_handler(CommandHandler("clear", clear_history))

    # Text Messages
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_text))

    logger.info("🟢 Telegram Bot Starting...")
    
    # Optional: setup commands menu on startup
    loop = asyncio.get_event_loop()
    loop.run_until_complete(set_commands(application))

    # Run polling
    application.run_polling()


if __name__ == "__main__":
    main()

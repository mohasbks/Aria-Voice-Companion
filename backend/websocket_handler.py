"""
websocket_handler.py – Core real-time pipeline with Sesame CSM concepts.

Additions over v1:
  - Emotion arc tracking: passes last-N emotions to LLM for contextual continuity
  - Session mood: dominant emotion fed as ambient context
  - Fixed WebSocket disconnect error (graceful handling of closed connections)
  - Processing stage events for richer UI feedback

Protocol (server → client):
  {"type": "ready",       "session_id": "..."}
  {"type": "processing",  "stage": "stt|llm|tts"}
  {"type": "transcript",  "text": "..."}
  {"type": "response",    "text": "...", "emotion": "...", "arc": [...]}
  {"type": "audio_start", "emotion": "..."}
  <binary MP3 frames>
  {"type": "audio_end"}
  {"type": "error",       "message": "..."}

Protocol (client → server):
  {"type": "init",   "session_id": "...", "mime_type": "..."}
  {"type": "text",   "text": "...",       "session_id": "..."}
  {"type": "ping"}
  <binary audio blob>
"""

import json
import logging
from fastapi import WebSocket, WebSocketDisconnect

from stt import transcribe_audio
from llm import get_llm_response
from tts import synthesize_speech_stream
from memory import (
    save_message,
    get_recent_messages,
    get_emotion_arc,
    get_session_mood,
    init_db,
)

logger = logging.getLogger(__name__)


async def _send(ws: WebSocket, payload: dict) -> bool:
    """
    Safely send a JSON text frame.
    Returns False (and logs) if the connection is already closed.
    """
    try:
        await ws.send_text(json.dumps(payload))
        return True
    except Exception as exc:
        logger.debug("WS send failed (connection closed?): %s", exc)
        return False


async def _stream_audio(ws: WebSocket, text: str, emotion: str, lang: str, engine_key: str) -> None:
    """Stream TTS audio as binary WebSocket frames with start/end bookmarks."""
    await _send(ws, {"type": "audio_start", "emotion": emotion})
    try:
        async for chunk in synthesize_speech_stream(text, emotion, lang, engine_key):
            try:
                await ws.send_bytes(chunk)
            except Exception:
                break   # Connection closed during stream — exit cleanly
    except Exception as exc:
        logger.error("Audio stream error: %s", exc)
    await _send(ws, {"type": "audio_end"})


async def _run_pipeline(
    ws: WebSocket, 
    user_text: str, 
    session_id: str, 
    lang: str = "en",
    groq_api_key: str = "",
    groq_api_key_arabic: str = "",
) -> None:
    """
    Full STT→LLM→TTS pipeline for an already-transcribed user_text.

    CSM improvements:
      - Gets emotion arc (last 6 assistant emotions) → feeds to LLM
      - Gets session mood (dominant emotion) → feeds to LLM
      - Saves messages with emotion labels for future arc computation
    """
    # 1. Echo transcript
    if not await _send(ws, {"type": "transcript", "text": user_text}):
        return

    # 2. Load conversation context from SQLite
    history      = get_recent_messages(n=10, session_id=session_id)
    emotion_arc  = get_emotion_arc(n=6,  session_id=session_id)
    session_mood = get_session_mood(n=6,  session_id=session_id)

    # Determine which key to pass based on lang
    engine_key = groq_api_key_arabic if lang == "ar" else groq_api_key

    # 3. LLM inference (with emotion arc context)
    await _send(ws, {"type": "processing", "stage": "llm"})
    result   = await get_llm_response(
        user_message=user_text,
        history=history,
        emotion_arc=emotion_arc,
        session_mood=session_mood,
        lang=lang,
        engine_key=engine_key,
    )
    ai_text  = result["text"]
    emotion  = result["emotion"]

    # 4. Persist both turns
    save_message("user",      user_text, emotion="neutral", session_id=session_id)
    save_message("assistant", ai_text,   emotion=emotion,   session_id=session_id)

    # 5. Send text response (UI shows this while audio loads)
    new_arc = get_emotion_arc(n=5, session_id=session_id)
    await _send(ws, {
        "type":    "response",
        "text":    ai_text,
        "emotion": emotion,
        "arc":     new_arc,   # send arc to frontend for visualization
    })

    # 6. Stream TTS audio
    await _send(ws, {"type": "processing", "stage": "tts"})
    await _stream_audio(ws, ai_text, emotion, lang, engine_key)


async def handle_websocket(ws: WebSocket) -> None:
    """
    Main WebSocket handler. One instance per connected client.
    Handles both text (Web Speech API) and binary (Groq Whisper) frames.
    """
    await ws.accept()
    session_id = "default"
    mime_type  = "audio/webm"
    # User session overrides (default to server .env if empty)
    user_lang = "en"
    user_groq_key = ""
    user_groq_key_ar = ""

    logger.info("WebSocket connected")

    try:
        while True:
            try:
                message = await ws.receive()
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected cleanly (session=%s)", session_id)
                break
            except Exception as exc:
                # Catches starlette's internal disconnect state errors
                err_str = str(exc).lower()
                if "disconnect" in err_str or "closed" in err_str:
                    logger.info("WebSocket closed (session=%s)", session_id)
                    break
                logger.error("WebSocket receive error: %s", exc)
                break

            # ── Text frame ────────────────────────────────────────────
            if message.get("type") == "websocket.receive" and message.get("text"):
                text_data = message["text"]
            elif "text" in message and message["text"]:
                text_data = message["text"]
            else:
                text_data = None

            if text_data:
                try:
                    data = json.loads(text_data)
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type", "")

                if msg_type == "init":
                    session_id = data.get("session_id", "default")
                    mime_type  = data.get("mime_type", "audio/webm")
                    user_lang  = data.get("lang", "en")
                    user_groq_key = data.get("groq_key", "")
                    user_groq_key_ar = data.get("groq_key_ar", "")
                    logger.info("Session: %s | mime: %s | lang: %s", session_id, mime_type, user_lang)
                    await _send(ws, {"type": "ready", "session_id": session_id})

                elif msg_type == "settings":
                    # Update session settings on the fly
                    if "lang" in data: user_lang = data["lang"]
                    if "groq_key" in data: user_groq_key = data["groq_key"]
                    if "groq_key_ar" in data: user_groq_key_ar = data["groq_key_ar"]
                    logger.info("Settings updated: lang=%s", user_lang)

                elif msg_type == "text":
                    user_text  = data.get("text", "").strip()
                    session_id = data.get("session_id", session_id)
                    if user_text:
                        await _run_pipeline(ws, user_text, session_id, user_lang, user_groq_key, user_groq_key_ar)

                elif msg_type == "ping":
                    await _send(ws, {"type": "pong"})

                continue

            # ── Binary frame (audio blob from MediaRecorder) ──────────
            audio_bytes = message.get("bytes")
            if audio_bytes:
                logger.info("Received audio: %d bytes", len(audio_bytes))
                await _send(ws, {"type": "processing", "stage": "stt"})
                user_text = await transcribe_audio(audio_bytes, mime_type)

                if not user_text:
                    await _send(ws, {
                        "type":    "error",
                        "message": "Could not transcribe audio. Please try again.",
                    })
                    continue

                await _run_pipeline(ws, user_text, session_id, user_lang, user_groq_key, user_groq_key_ar)

    except Exception as exc:
        logger.error("WebSocket handler error: %s", exc)
        await _send(ws, {"type": "error", "message": str(exc)})

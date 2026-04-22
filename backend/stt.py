"""
stt.py – Speech-to-Text via Groq Whisper.
"""

import os
import logging
import httpx
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
# Use turbo model for lowest possible latency
STT_MODEL = "whisper-large-v3-turbo"

async def transcribe_audio(audio_bytes: bytes, mime_type: str = "audio/webm") -> str:
    """
    Transcribe audio blob using Groq API.
    Returns transcript text, or empty string on failure.
    """
    if not GROQ_API_KEY:
        logger.warning("No GROQ_API_KEY provided for STT")
        return ""

    if not audio_bytes:
        return ""

    # Provide a simple extension and filename based on mime type
    ext = mime_type.split("/")[-1].split(";")[0]
    if ext not in ["webm", "ogg", "mp4", "wav", "mp3", "m4a"]:
        ext = "webm"
    
    filename = f"audio.{ext}"

    files = {
        "file": (filename, audio_bytes, mime_type)
    }
    data = {
        "model": STT_MODEL,
        "temperature": "0.0"
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(GROQ_STT_URL, headers=headers, data=data, files=files)
        
        if resp.status_code == 200:
            text = resp.json().get("text", "").strip()
            logger.info("STT Transcript: %s", text)
            return text
            
        logger.error("STT Error %d: %s", resp.status_code, resp.text)
        return ""
    except Exception as exc:
        logger.error("STT Exception: %s", exc)
        return ""

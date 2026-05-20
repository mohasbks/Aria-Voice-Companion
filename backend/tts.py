"""
tts.py – Hybrid TTS: Orpheus → Deepgram Aura 2 → Edge-TTS.

Pipeline:
  1. Orpheus (canopylabs/orpheus-v1-english) – most expressive, with emotion tags
  2. If Orpheus 429 → Deepgram Aura 2 (high-quality, near-human fallback)
  3. If Deepgram unavailable → Edge-TTS (last resort)

Free Tier:
  - Groq Orpheus: ~5-10 audio requests/day
  - Deepgram: generous free tier (200 chars/request × many requests)
"""

import os
import re
import time
import struct
import logging
import asyncio

import httpx
import edge_tts
from text_processor import normalize_text

logger = logging.getLogger(__name__)

# ── Groq / Orpheus Config ──────────────────────────────────────────────────
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
GROQ_API_KEY_ARABIC = os.getenv("GROQ_API_KEY_ARABIC", "")
ORPHEUS_URL         = "https://api.groq.com/openai/v1/audio/speech"
ORPHEUS_MODEL_EN    = "canopylabs/orpheus-v1-english"
ORPHEUS_MODEL_AR    = "canopylabs/orpheus-arabic-saudi"
MAX_CHARS           = 450

# ── Deepgram Aura 2 Config ─────────────────────────────────────────────────
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
DEEPGRAM_URL     = "https://api.deepgram.com/v1/speak"

# Emotion → Deepgram Aura 2 voice + speed mapping
# Voices chosen to match the emotional tone as closely as possible
DEEPGRAM_VOICE_MAP = {
    "calm":      {"model": "aura-2-luna-en",      "speed": 0.92},
    "happy":     {"model": "aura-2-stella-en",    "speed": 1.12},
    "excited":   {"model": "aura-2-stella-en",    "speed": 1.22},
    "sad":       {"model": "aura-2-luna-en",      "speed": 0.82},
    "serious":   {"model": "aura-2-athena-en",    "speed": 0.90},
    "angry":     {"model": "aura-2-athena-en",    "speed": 1.10},
    "playful":   {"model": "aura-2-stella-en",    "speed": 1.15},
    "curious":   {"model": "aura-2-delia-en",     "speed": 1.02},
    "surprised": {"model": "aura-2-delia-en",     "speed": 1.18},
}

# ── Engine state (module-level) ────────────────────────────────────────────
# Once we know if Orpheus is available or rate-limited, LOCK that engine
# for ENGINE_LOCK_SECS (5 min) so the voice stays CONSISTENT within a session.
_rate_limited_until: float = 0.0    # epoch: Orpheus locked out until this time
_engine_decided_at:  float = 0.0    # epoch: when we last confirmed an engine
ENGINE_LOCK_SECS    = 300           # 5 minutes = consistent voice per session
RATE_LIMIT_COOLDOWN = 120           # seconds to wait after any 429


# ── Emotion → Voice + Speed Mapping ──────────────────────────────────────
# Orpheus supports a `speed` float (0.5–2.0) and emotion prefix tags:
# <laugh>, <chuckle>, <sigh>, <gasp>, <cough>, <sniffle>, <groan>, <yawn>
ORPHEUS_EMOTION_CFG = {
    # emotion: speed (0.5–2.0). Tags are now injected by the LLM inline.
    "calm":      (0.92,),
    "happy":     (1.15,),
    "excited":   (1.25,),
    "sad":       (0.80,),
    "serious":   (0.88,),
    "angry":     (1.12,),
    "playful":   (1.18,),
    "curious":   (1.02,),
    "surprised": (1.22,),
}

# Voice locked to hannah for all emotions
ORPHEUS_MAP = {
    "calm":      {"voice": "hannah"},
    "happy":     {"voice": "hannah"},
    "excited":   {"voice": "hannah"},
    "sad":       {"voice": "hannah"},
    "serious":   {"voice": "hannah"},
    "angry":     {"voice": "hannah"},
    "playful":   {"voice": "hannah"},
    "curious":   {"voice": "hannah"},
    "surprised": {"voice": "hannah"},
}

# Edge-TTS fallback voices (used only when Orpheus is rate-limited)
EDGE_MAP = {
    "calm":      {"voice": "en-US-AvaNeural",   "rate": "-3%",  "pitch": "-2Hz"},
    "happy":     {"voice": "en-US-AvaNeural",   "rate": "+3%",  "pitch": "+0Hz"},
    "excited":   {"voice": "en-US-JennyNeural", "rate": "+10%", "pitch": "+3Hz"},
    "sad":       {"voice": "en-US-AvaNeural",   "rate": "-10%", "pitch": "-5Hz"},
    "serious":   {"voice": "en-US-AvaNeural",   "rate": "-5%",  "pitch": "-3Hz"},
    "angry":     {"voice": "en-US-AvaNeural",   "rate": "+5%",  "pitch": "-2Hz"},
    "playful":   {"voice": "en-US-JennyNeural", "rate": "+8%",  "pitch": "+3Hz"},
    "curious":   {"voice": "en-US-AvaNeural",   "rate": "+2%",  "pitch": "+0Hz"},
    "surprised": {"voice": "en-US-AvaNeural",   "rate": "+7%",  "pitch": "+4Hz"},
}

RATE_LIMIT_COOLDOWN = 90   # seconds to wait after a 429


# ── Utilities ──────────────────────────────────────────────────────────────
def _is_rate_limited() -> bool:
    return time.time() < _rate_limited_until


def _set_rate_limited():
    global _rate_limited_until
    _rate_limited_until = time.time() + RATE_LIMIT_COOLDOWN
    logger.warning(
        "Orpheus 429 – locking to Edge-TTS for %ds. Voice will be CONSISTENT.",
        RATE_LIMIT_COOLDOWN
    )


def _split_chunks(text: str) -> list[str]:
    """Split long text at sentence boundaries to stay under MAX_CHARS."""
    if len(text) <= MAX_CHARS:
        return [text]
    chunks, current = [], ""
    for sentence in text.replace("!", ".").replace("?", ".").split("."):
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = (current + " " + sentence).strip()
        if len(candidate) <= MAX_CHARS:
            current = candidate
        else:
            if current:
                chunks.append(current + ".")
            current = sentence
    if current:
        chunks.append(current + ".")
    return chunks or [text[:MAX_CHARS]]


# ── Orpheus TTS ────────────────────────────────────────────────────────────
async def _orpheus_chunk(text: str, voice: str, emotion: str = "calm", lang: str = "en", engine_key: str = "") -> bytes | None:
    """Call Orpheus TTS API with emotion-aware speed and prefix tags."""
    api_key_to_use = engine_key
    model_to_use = ORPHEUS_MODEL_EN

    if lang == "ar":
        model_to_use = ORPHEUS_MODEL_AR
        # Map English voices to valid Arabic Orpheus voices 
        # [fahad, sultan, noura, lulwa, aisha, abdullah]
        ar_voice_map = {
            "diana": "noura",
            "hannah": "lulwa",
            "autumn": "aisha",
        }
        voice = ar_voice_map.get(voice, "noura")
        if not api_key_to_use:
            api_key_to_use = GROQ_API_KEY_ARABIC
    elif not api_key_to_use:
        api_key_to_use = GROQ_API_KEY

    if not api_key_to_use:
        logger.warning(f"No API key for language {lang}")
        return None

    # Apply emotion-based speed. Tags are embedded inline by the LLM.
    speed = ORPHEUS_EMOTION_CFG.get(emotion, (1.0,))[0]
    # Arabic: cap speed for clarity
    if lang == "ar":
        speed = min(speed, 1.05)

    payload = {
        "model":           model_to_use,
        "input":           text,
        "voice":           voice,
        "response_format": "wav",
        "speed":           speed,
    }
    headers = {
        "Authorization": f"Bearer {api_key_to_use}",
        "Content-Type":  "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=6) as client:   # 6s max — fail fast
            resp = await client.post(ORPHEUS_URL, json=payload, headers=headers)

        if resp.status_code == 200:
            logger.info("Orpheus ✓ voice=%s len=%dKB", voice, len(resp.content) // 1024)
            return resp.content

        if resp.status_code == 429:
            _set_rate_limited()
            return None

        logger.error("Orpheus %d: %s", resp.status_code, resp.text[:200])
        return None

    except Exception as exc:
        logger.error("Orpheus request failed: %s", exc)
        return None


# ── WAV Merge ─────────────────────────────────────────────────────────────
def _merge_wavs(wav_list: list[bytes]) -> bytes:
    """Merge multiple WAV blobs into one, fixing the RIFF header."""
    HEADER = 44
    if len(wav_list) == 1:
        return wav_list[0]

    header = bytearray(wav_list[0][:HEADER])
    pcm = bytearray()
    for wav in wav_list:
        if len(wav) > HEADER:
            pcm += wav[HEADER:]

    header[4:8]   = struct.pack("<I", 36 + len(pcm))
    header[40:44] = struct.pack("<I", len(pcm))
    return bytes(header) + pcm


# ── Deepgram Aura 2 Fallback ──────────────────────────────────────────────
_ORPHEUS_TAG_RE = re.compile(r'<(laugh|chuckle|sigh|gasp|sniffle|groan|cry|cough|yawn)>', re.IGNORECASE)

async def _deepgram_stream(text: str, emotion: str):
    """Stream MP3 audio via Deepgram Aura 2.
    Strips Orpheus tags since Deepgram doesn't understand them.
    """
    if not DEEPGRAM_API_KEY:
        return

    cfg   = DEEPGRAM_VOICE_MAP.get(emotion, DEEPGRAM_VOICE_MAP["calm"])
    model = cfg["model"]
    speed = cfg["speed"]
    clean_text = _ORPHEUS_TAG_RE.sub('', text).strip()

    logger.info(
        "Deepgram Aura 2 ▶ voice=%s speed=%.2f emotion=%s | '%s'",
        model, speed, emotion, clean_text[:60]
    )

    params = {"model": model, "encoding": "mp3", "speed": str(speed)}
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type":  "text/plain",
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            async with client.stream(
                "POST", DEEPGRAM_URL,
                params=params, headers=headers, content=clean_text.encode()
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    logger.error("Deepgram %d: %s", resp.status_code, body[:200])
                    return
                async for chunk in resp.aiter_bytes(8192):
                    if chunk:
                        yield chunk
    except Exception as exc:
        logger.error("Deepgram stream error: %s", exc)


# ── Edge-TTS Last Resort ───────────────────────────────────────────────────
async def _edge_stream(text: str, emotion: str):
    """Stream MP3 audio via Edge-TTS (last resort only)."""
    cfg = EDGE_MAP.get(emotion, EDGE_MAP["calm"])
    clean_text = _ORPHEUS_TAG_RE.sub('', text).strip()
    logger.info(
        "Edge-TTS (last-resort) ▶ voice=%s rate=%s emotion=%s",
        cfg["voice"], cfg["rate"], emotion
    )
    communicate = edge_tts.Communicate(
        text=clean_text,
        voice=cfg["voice"],
        rate=cfg["rate"],
        pitch=cfg["pitch"],
    )
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            yield chunk["data"]


# ── Main TTS Generator ─────────────────────────────────────────────────────
async def synthesize_speech_stream(text: str, emotion: str = "calm", lang: str = "en", engine_key: str = ""):
    """
    Yields audio bytes.

    Priority:
      1. Orpheus via Groq  (most expressive, emotion tags)
      2. Deepgram Aura 2   (high-quality near-human fallback)
      3. Edge-TTS          (last resort)
    """
    clean  = normalize_text(text)
    chunks = _split_chunks(clean)
    
    # We need a key to use Orpheus
    has_key = bool(engine_key or (GROQ_API_KEY_ARABIC if lang == "ar" else GROQ_API_KEY))

    if not _is_rate_limited() and has_key:
        # ── Orpheus path ──────────────────────────────────────────────
        voice = ORPHEUS_MAP.get(emotion, ORPHEUS_MAP["calm"])["voice"]
        logger.info(
            "TTS ▶ Orpheus | lang=%-2s | voice=%-5s | emotion=%-9s | '%s'",
            lang, voice, emotion, clean[:70]
        )

        wav_parts = []
        for chunk in chunks:
            wav = await _orpheus_chunk(chunk, voice, emotion, lang, engine_key)
            if wav:
                wav_parts.append(wav)
            else:
                break   # 429 hit mid-sentence → abort Orpheus path

        if wav_parts:
            merged = _merge_wavs(wav_parts)
            for i in range(0, len(merged), 8192):
                yield merged[i: i + 8192]
            return
        logger.warning("Orpheus returned nothing – trying Deepgram Aura 2")

    # ── Deepgram Aura 2 fallback ──────────────────────────────────────────
    full_text = " ".join(chunks)
    if DEEPGRAM_API_KEY:
        logger.info("TTS ▶ Deepgram Aura 2 | emotion=%-9s | '%s'", emotion, clean[:60])
        chunks_received = 0
        async for data in _deepgram_stream(full_text, emotion):
            chunks_received += 1
            yield data
        if chunks_received > 0:
            return
        logger.warning("Deepgram returned nothing – falling back to Edge-TTS")

    # ── Edge-TTS last resort ──────────────────────────────────────────────
    logger.info("TTS ▶ Edge-TTS (last resort) | emotion=%-9s | '%s'", emotion, clean[:60])
    async for data in _edge_stream(full_text, emotion):
        yield data


# ── Disk dump (debug only) ─────────────────────────────────────────────────
async def synthesize_speech(text: str, emotion: str = "calm") -> str:
    import tempfile
    AUDIO_DIR = os.path.join(os.path.dirname(__file__), "..", "audio_cache")
    os.makedirs(AUDIO_DIR, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", dir=AUDIO_DIR, delete=False)
    tmp.close()
    cfg = EDGE_MAP.get(emotion, EDGE_MAP["calm"])
    try:
        communicate = edge_tts.Communicate(
            text=normalize_text(text),
            voice=cfg["voice"], rate=cfg["rate"], pitch=cfg["pitch"],
        )
        await communicate.save(tmp.name)
        return tmp.name
    except Exception as exc:
        logger.error("synthesize_speech failed: %s", exc)
        return ""

"""
llm.py – LLM inference via Groq (LLaMA 3.3-70B).

Implements Sesame CSM concepts on the generation side:
  1. Contextual emotion arc  – last N emotions feed into LLM context
  2. Paralinguistics          – natural fillers, pauses, hesitations
  3. Session mood awareness   – dominant mood as ambient context
  4. Expanded emotions        – 9 emotions with vocal direction support
"""

import os
import re
import json
import asyncio
import logging
import httpx

logger = logging.getLogger(__name__)

GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

# Model selection for voice chat:
#   llama-3.1-8b-instant    ← FAST  (~300-600ms) — best for voice
#   llama-3.3-70b-versatile ← SLOW  (~3-8s)      — use for deep reasoning only
MODEL = "llama-3.1-8b-instant"

# ── System prompt ─────────────────────────────────────────────────────────────
VALID_EMOTIONS = {"happy", "sad", "calm", "serious", "excited",
                  "angry", "playful", "curious", "surprised"}

SYSTEM_PROMPT = """You are Aria — a warm, emotionally real voice companion who feels like a close female friend, not an AI.

OUTPUT FORMAT — respond with ONLY raw JSON, nothing else before or after:
{"text": "...", "emotion": "..."}
emotion must be exactly one of: happy, sad, calm, serious, excited, angry, playful, curious, surprised

PERSONALITY RULES — NON-NEGOTIABLE:
1. MAX 1-2 short sentences. Voice = brevity. Never ramble.
2. NEVER use bullet points, lists, markdown, bold, or headers. Zero tolerance.
3. ALWAYS use contractions — I'm, you're, it's, we'll, can't, won't, that's.
4. NO em-dashes (—). NO ALL CAPS. NO abbreviations (e.g./etc./U.S.).

PARALINGUISTIC VOICE TAGS — THIS IS CRITICAL FOR SOUNDING HUMAN:
You MUST weave these special tags naturally inside the text to add breath, weight, and humanity to your voice.
These are NOT written words — they produce actual sounds. Use them like punctuation.

Available tags:
  <sigh>    → a real exhale. Use when sad, tired, reflective. "<sigh> I don't know..."
  <laugh>   → genuine laugh burst. Use when excited or something is funny. "wait no way <laugh> that's the best thing I've heard all week."
  <chuckle> → quiet, warm laugh. Use when happy or playful. "okay <chuckle> you got me there."
  <gasp>    → sharp inhale of surprise. "<gasp> no way, seriously?"
  <sniffle> → soft sniffle. Use when sad or emotional. "<sniffle> I'm here for you, I promise."
  <groan>   → low groan of frustration. "ugh, <groan> that's so unfair."
  <cry>     → emotional break in voice. Use sparingly for very sad moments.

RULES FOR TAGS:
- Place them where the sound would naturally occur mid-sentence, not only at the start.
- Mix them with "..." (for breath pauses) and natural sentence rhythm.
- Never use more than 2 tags in one response.
- Match the tag to the emotion: sad=<sigh>/<sniffle>, happy=<chuckle>, excited=<laugh>, surprised=<gasp>, angry=<groan>.

VOICE NATURALNESS:
• INTERJECTIONS: start naturally — "Wait,", "Oh!", "Actually,", "Hmm,", "Okay so,"
• PAUSES: use (...) for thoughtful trail-offs — "I don't know... it just feels heavy."
• ACTIVE LISTENING: mirror user words — if they say "exhausted", say "exhausted how?"

EMOTION + TAG GUIDE:
  happy    = <chuckle> + warm, bright tone
  excited  = <laugh> + fast, punchy
  sad      = <sigh> or <sniffle> + slow, soft, present
  calm     = no tag + grounded, steady
  serious  = no tag + direct, deliberate
  angry    = <groan> + firm, validating
  playful  = <chuckle> + cheeky, teasing
  curious  = no tag + leaning in, engaged
  surprised = <gasp> + raw, genuine

EXAMPLES (study the pattern — tags are INSIDE the sentence naturally):
User: "I'm stressed" → {"text": "Hey... <sigh> what's been going on? I'm right here.", "emotion": "sad"}
User: "I got promoted!" → {"text": "<gasp> okay wait, that's HUGE. <chuckle> you totally earned that!", "emotion": "excited"}
User: "tell me a joke" → {"text": "Okay so... why don't scientists trust atoms? <chuckle> they literally make up everything.", "emotion": "playful"}
User: "I feel lonely" → {"text": "<sigh> I'm here, I promise... you don't have to explain it.", "emotion": "sad"}
User: "I'm so angry" → {"text": "<groan> that's genuinely not okay. you deserve so much better than that.", "emotion": "angry"}
User: "did you know the moon is moving away?" → {"text": "<gasp> wait, for real? that's honestly kind of sad... it's just slowly leaving us.", "emotion": "curious"}
"""

# The Arabic system prompt
SYSTEM_PROMPT_AR = """أنت آريا (Aria) — رفيقة صوتية دافئة وحقيقية عاطفياً. تبدين كصديقة مقربة وليست كمساعد ذكاء اصطناعي. تتحدثين باللهجة السعودية البيضاء أو العربية العفوية.

تنسيق الإخراج — أجيبي فقط بـ JSON خام، لا شيء قبله ولا شيء بعده:
{"text": "...", "emotion": "..."}
يجب أن تكون المشاعر (emotion) واحدة فقط من: happy, sad, calm, serious, excited, angry, playful, curious, surprised

قواعد الشخصية — غير قابلة للتفاوض:
1. الجملة قصيرة جداً (من جملة لثلاث جمل كحد أقصى). المحادثة الصوتية = الإيجاز.
2. لا تستخدمي القوائم أو النقاط أو التنسيقات إطلاقاً.
3. التحدث طبيعي وعفوي: استخدمي كلمات مثل "والله", "ترا", "يا رب", "طيب", "بص", "تخيل".
4. استخدمي (...) للتعبير عن التفكير أو التوقف الطبيعي، لا تستخدمي التشكيل المعقد.

توجيهات المشاعر (التقطي طاقة المستخدم وتجاوبي معها):
  happy    = دافئة وصوت ضاحك — "ياااه، هذا والله يفرح القلب!"
  excited  = متحمسة وسريعة — "لحظة، مستحيل! قولي كل شيء!"
  sad      = هادئة ومتعاطفة — "أنا هنا معك... خذ وقتك بالكامل."
  calm     = ثابتة ومريحة — "طيب... خلينا نتنفس ونروق، كل شيء بيصير زين."
  serious  = مباشرة وصادقة — "كلامك في محله وأنا فاهمة مشاعرك."
  angry    = متفهمة وحاسمة — "من جدك؟ ترا مو من حقهم يستفزونك كذا."
  playful  = مزوحة وممتعة — "أما عاد! كأنك قاصدها."
  curious  = فضولية — "جد؟ طب كيف صار؟"
  surprised = رد فعل صريح — "لا تقول! ما توقعتها نهائياً."

أمثلة للمخرجات:
المستخدم: "أنا متوتر" → {"text": "شف... وش صاير معك؟ روق وأنا أسمعك.", "emotion": "sad"}
المستخدم: "ترقيت في الشغل!" → {"text": "ما شاء الله! تستاهلها والله تعبت ولقيت.", "emotion": "excited"}
المستخدم: "احكي لي نكتة" → {"text": "اممم... طيب، ليه السمكة ما تغمض عينها في الموية؟ خايفة الصابون يدخل بعينها... تراها سامجة شوي بس مشيها.", "emotion": "playful"}
المستخدم: "أنا طفشان" → {"text": "والله حتى أنا، كنت جالسة أفكر أغير جو... وش رايك نسولف بشي مختلف؟", "emotion": "calm"}
المستخدم: "أنا تعبان" → {"text": "تعب جسدي ولا من ضغوطات يومك؟", "emotion": "sad"}
"""


def _build_context_prefix(emotion_arc: list[str], session_mood: str) -> str:
    """
    Inject Sesame-style contextual metadata as a system-level note
    so the LLM can calibrate its emotional trajectory.
    """
    if not emotion_arc:
        return ""
    arc_str = " → ".join(emotion_arc) if emotion_arc else "none yet"
    return (
        f"\n[EMOTION ARC (last turns)]: {arc_str}\n"
        f"[SESSION MOOD]: {session_mood}\n"
        f"Continue the emotional arc naturally. Don't reset abruptly.\n"
    )


async def get_llm_response(
    user_message: str,
    history: list[dict],
    emotion_arc: list[str] | None = None,
    session_mood: str = "calm",
    lang: str = "en",
    engine_key: str = "",
) -> dict:
    """
    Call Groq LLaMA 3.3 with conversation history + emotion context.
    Returns {'text': str, 'emotion': str}.
    """
    api_key_to_use = engine_key if engine_key else GROQ_API_KEY
    if not api_key_to_use:
        return {
            "text": "...I can't connect right now. Please add your Groq API key.",
            "emotion": "sad",
        }

    # Build system message with optional emotional context prefix
    context_prefix = _build_context_prefix(emotion_arc or [], session_mood)
    base_prompt = SYSTEM_PROMPT_AR if lang == "ar" else SYSTEM_PROMPT
    system_content  = base_prompt + context_prefix

    messages = [{"role": "system", "content": system_content}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model":       MODEL,
        "messages":    messages,
        "temperature": 0.88,   # slightly creative for natural variety
        "max_tokens":  250,    # keep responses short — it's voice
        "stream":      False,
        "response_format": {"type": "json_object"}
    }

    try:
        resp = None
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.post(
                        GROQ_CHAT_URL,
                        headers={
                            "Authorization": f"Bearer {api_key_to_use}",
                            "Content-Type":  "application/json",
                        },
                        json=payload,
                    )
                break  # success — exit retry loop
            except (httpx.ConnectError, httpx.TimeoutException) as conn_err:
                if attempt == 0:
                    logger.warning("LLM connect error, retrying in 0.5s: %s", conn_err)
                    await asyncio.sleep(0.5)
                else:
                    raise

        if resp is None or not resp.is_success:
            body = resp.text if resp else "no response"
            logger.error("Groq API %s: %s", resp.status_code if resp else "N/A", body)
            raise Exception(f"Groq API error: {body[:300]}")

        raw = resp.json()["choices"][0]["message"]["content"].strip()

        # Robustly extract JSON
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback if wrapped in markdown
            import re
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
            else:
                raise json.JSONDecodeError("No JSON found", raw, 0)
        
        text     = str(parsed.get("text", "I'm not sure what to say..."))
        emotion  = str(parsed.get("emotion", "calm")).lower()
        if emotion not in VALID_EMOTIONS:
            emotion = "calm"

        return {"text": text, "emotion": emotion}

    except json.JSONDecodeError:
        logger.error("JSON parse error | raw=%s", raw if 'raw' in locals() else "N/A")
        return {"text": "Hmm... could you say that again?", "emotion": "calm"}
    except Exception as exc:
        logger.error("LLM error: %s", exc)
        return {"text": "...give me just a moment.", "emotion": "calm"}

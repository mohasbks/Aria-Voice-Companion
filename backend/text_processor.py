"""
text_processor.py – Plain-text normalizer for edge-tts 7.x.

Key insight: edge-tts sounds most natural when text is clean, 
conversational, and flows like actual spoken English.

Rules:
  - NO XML/SSML (edge-tts 7.x reads tags as literal text)
  - No ALL CAPS (edge-tts may spell them or read oddly)
  - Em-dash → ", " (comma pause instead of em-dash)
  - Ellipsis → "..." (max 3 dots, edge-tts pauses naturally)
  - No abbreviations
  - "us" disambiguation
"""

import re


# ── Strip any XML/SSML that leaked in ────────────────────────────────────────
def _strip_xml(text: str) -> str:
    return re.sub(r'<[^>]+>', '', text)


# ── Convert ALL CAPS words to Title Case ──────────────────────────────────────
# edge-tts doesn't use caps for emphasis — it may read oddly.
# "that's REAL" → "that's Real" (edge-tts adds stress naturally from punctuation)
def _normalize_caps(text: str) -> str:
    # Only convert words that are 2+ letters and fully uppercase
    return re.sub(
        r'\b([A-Z]{2,})\b',
        lambda m: m.group(1).capitalize(),
        text
    )


# ── Em-dash → natural comma pause ────────────────────────────────────────────
# edge-tts doesn't pause on em-dashes well — use commas instead
def _normalize_emdash(text: str) -> str:
    return re.sub(r'\s*—\s*', ', ', text)


# ── Abbreviation expansion ────────────────────────────────────────────────────
_ABBREV = [
    (r'\bDr\.\s',    'Doctor '),
    (r'\bMr\.\s',    'Mister '),
    (r'\bMrs\.\s',   'Missus '),
    (r'\bMs\.\s',    'Miss '),
    (r'\bvs\.\s',    'versus '),
    (r'\betc\.',     'and so on'),
    (r'\be\.g\.\s',  'for example, '),
    (r'\bi\.e\.\s',  'that is, '),
    (r'\bU\.S\.A\b', 'the USA'),
    (r'\bU\.S\.\b',  'the US'),
    (r'\bU\.K\.\b',  'the UK'),
    (r'\bA\.I\.\b',  'AI'),
]


# ── "us" pronoun disambiguation ───────────────────────────────────────────────
_US_FIXES = [
    (r'\bfor both of us\b',   'for the two of us'),
    (r'\bbetween us\b',       'between the two of us'),
    (r'\bjoin us\b',          'join in'),
    (r'\bcontact us\b',       'reach out'),
    (r'\bwith us\s*\.',       'with us here.'),
    (r'\btrust us\b',         'trust me'),
    (r'\bus\s*\.',            'us,'),
    (r'\bus\s*$',             'us here'),
]


# ── Homograph hint cleanup ────────────────────────────────────────────────────
# LLM writes "lead (leed)" → we extract phonetic form → "leed"
def _clean_homograph_hints(text: str) -> str:
    return re.sub(r'\b\w+\s*\(([^)]{1,15})\)', r'\1', text)


# ── Punctuation normalization ─────────────────────────────────────────────────
def _normalize_punctuation(text: str) -> str:
    # Keep `...` as exactly 3 dots — Orpheus uses it as a natural "thinking" pause.
    # Only collapse 4+ dots to exactly 3 — never remove ellipsis entirely.
    text = re.sub(r'\.{4,}', '...', text)         # .... → ... (max 3 dots)
    text = re.sub(r'…', '...', text)              # unicode ellipsis → ASCII ...
    text = re.sub(r'\s{2,}', ' ', text)           # collapse spaces
    text = re.sub(r'([.!?])\s+([.!?])', r'\1\2', text)  # fix double punctuation
    return text.strip()


# ── Main pipeline ─────────────────────────────────────────────────────────────
def normalize_text(text: str) -> str:
    """
    Full normalization → clean plain text for edge-tts.
    Order matters: strip XML first, then transform text.
    """
    text = _strip_xml(text)
    text = _clean_homograph_hints(text)
    text = _normalize_caps(text)
    text = _normalize_emdash(text)

    for pattern, replacement in _ABBREV:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    for pattern, replacement in _US_FIXES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    text = _normalize_punctuation(text)
    return text

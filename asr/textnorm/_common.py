"""
Common utilities shared by recipe textnorm scripts.

Provides:
  - remove_paralinguistic_tags() — strips [breath], [cough], [laugh],
    [noise], etc. tags from Arabic transcripts.
  - normalize_char_level() — shared CHARACTER-LEVEL Arabic normalization,
    the intersection of every dialect recipe script (SAU/ARE/EGY/IRQ/SYR/MSA),
    the central gigaspeech3.py normalization, and decode.py's script cleanup.
"""

import regex as re

# Patterns for paralinguistic tags: [breath], [cough], [laugh], [noise], etc.
_TAG_RE = re.compile(
    r"\[(?:breath|cough|laugh|noise|clap|sneeze|lip|tongue|click|"
    r"music|sing|ring|beep|applause|silence|noise|other)\]",
    re.IGNORECASE,
)

# Also match # filler markers like # أأأ
_HASH_FILLER_RE = re.compile(r"#\s*\S*")


def remove_paralinguistic_tags(text: str) -> str:
    """
    Remove paralinguistic annotation tags from Arabic transcripts.

    Handles:
      - [breath], [cough], [laugh], [noise], etc.
      - # filler markers
      - Empty brackets like [], ()
    """
    text = _TAG_RE.sub("", text)
    text = _HASH_FILLER_RE.sub("", text)
    return text.strip()


# ══════════════════════════════════════════════════════════════════════
# Shared character-level normalization
# ══════════════════════════════════════════════════════════════════════

# Tashkeel / diacritics — union of the ranges used by the dialect scripts
# ([ً-ْ] in SAU, ً-ٟ in EGY, ؗ-ًؚ-ْ in SYR,
# plus superscript alef ٰ and Quranic marks ۖ-ۭ from the
# central gigaspeech3.py normalization).
_RE_TASHKEEL = re.compile(r"[ؗ-ًؚ-ٰٟۖ-ۭ]")

# Tatweel / kashida
_RE_TATWEEL = re.compile(r"ـ")

# Zero-width / bidi-control characters
_RE_ZERO_WIDTH = re.compile(r"[​-‏‪-‮⁦-⁩﻿]")

# Alef variants → bare alef (all dialects agree on this)
_RE_ALEF_VARIANTS = re.compile(r"[أإآٱ]")

# Elongated alef hesitation (ااا+) — ARE/SYR/decode.py use 3+ alefs
_RE_ELONGATED_ALEF = re.compile(r"ا{3,}")

# Punctuation & symbols (Unicode categories P and S) — removed, matching the
# decode-side script cleanup so training targets tokenize the same way.
_RE_PUNCT_SYMBOL = re.compile(r"[\p{P}\p{S}]")

# Whitespace
_RE_MULTI_SPACE = re.compile(r"\s+")

# Eastern Arabic digits → Western (٠-٩ → 0-9)
_EASTERN_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# Persian / non-standard letters → Arabic equivalents
_PERSIAN_LETTERS = str.maketrans({"پ": "ب", "ڤ": "ف", "چ": "ج", "گ": "ك"})


def normalize_char_level(text: str) -> str:
    """
    Shared character-level Arabic normalization, common to all dialects.

    Used as the DEFAULT normalization for TRAINING targets (see
    datamodule.ArabicAsrDataModule._apply_recipe_textnorm).

    Operations (all dialect scripts agree on these):
      1. Remove paralinguistic tags ([breath], # أأأ, ...)
      2. Remove tashkeel / diacritics
      3. Remove tatweel, zero-width / bidi-control characters
      4. Alef variants → ا  (أ إ آ ٱ)
      5. Persian letters → Arabic  (پ→ب, ڤ→ف, چ→ج, گ→ك)
      6. Eastern Arabic digits → Western (٠-٩ → 0-9)
      7. Remove elongated alef hesitation (ااا+)
      8. Remove punctuation / symbols
      9. Collapse whitespace

    Deliberately EXCLUDED — these change word surface forms or are
    dialect-specific, and stay in the dialect-level textnorm (applied at
    decode time always, at training time only with --dialect-textnorm):
      - ة→ه, ى→ي, ؤ→و, ئ→ي, standalone ء removal
        (dialects disagree; SYR explicitly preserves them)
      - word-level dialect mappings / filler lists / number-word
        standardization (e.g. هذه↔هذي, بعد↔كمان)
      - Latin character removal (kept for code-switched content)
    """
    if not text or not text.strip():
        return ""

    text = remove_paralinguistic_tags(text)
    text = _RE_TASHKEEL.sub("", text)
    text = _RE_TATWEEL.sub("", text)
    text = _RE_ZERO_WIDTH.sub("", text)
    text = _RE_ALEF_VARIANTS.sub("ا", text)
    text = text.translate(_PERSIAN_LETTERS)
    text = text.translate(_EASTERN_DIGITS)
    text = _RE_ELONGATED_ALEF.sub("", text)
    text = _RE_PUNCT_SYMBOL.sub("", text)
    return _RE_MULTI_SPACE.sub(" ", text).strip()

"""
Saudi Arabic dialect text normalization.

Based on error analysis of SAU ASR decoding (25.15% → 21.33% WER with
proper textnorm).  Handles:

1. Character-level cleanup: diacritics, alef/hamza/ta-marbuta normalization
2. Dialect ↔ MSA word pairs (هذه↔هذي, فيه↔في, etc.)
3. Number form standardization (اربعه↔اربع, etc.)
4. Filler / hesitation removal (ااا, امم, اه, etc.)
5. Single-letter fragment cleanup (subword artifacts from BPE tokenizer)
6. Paralinguistic tag removal ([breath], [laugh], etc.)
"""

import regex as re

from textnorm._common import remove_paralinguistic_tags


# ══════════════════════════════════════════════════════════════════════
# Stage 1: Word-level dialect ↔ MSA mapping
# ══════════════════════════════════════════════════════════════════════

# Canonical form → {variants}
_DIALECT_CANONICAL = {
    # Demonstratives — canonical: SAUDI SPOKEN form
    "هذي": {"هذه", "هاذا", "هاي"},
    "هذا": {"هاذا", "هذاك", "هذول", "هذولا", "هذولي"},
    # Prepositions + attached pronouns — canonical: WITHOUT pronoun
    "في": {"فيه", "فيها", "فيهم", "فينا"},
    "من": {"منه", "منها", "منهم", "منا"},
    "علي": {"عليه", "عليها", "عليهم", "علينا", "وعلا", "وعلي"},
    "مع": {"معه", "معها", "معهم", "معنا"},
    "عند": {"عنده", "عندها", "عندهم"},
    "له": {"لها", "لهم"},
    # Complementizer — canonical: WITH ha
    "انه": {"ان", "انها", "انهم", "انو"},
    "لانه": {"لان", "لانها"},
    # "thing" — canonical: WITHOUT hamza
    "شي": {"شيء", "اشي", "اشيء"},
    "الشي": {"الشيء"},
    # "like this" — canonical: كذا (MSA/Gulf standard)
    "كذا": {"كده", "كدا", "كذي", "كدهي"},
    # "went / will go"
    "راح": {"رح", "رايح", "رحت"},
    # "always"
    "دايما": {"دائما", "دايم", "دايماً"},
    # "we"
    "احنا": {"حنا", "نحن", "نحنا"},
    # "yes / what" — disambiguated by context, normalize to common form
    "اي": {"ايه", "ايوا", "ايوة", "إي", "إيه", "إيوا", "إيوة"},
    # "what" (dialect interrogative)
    "ايش": {"وش", "ويش", "ايشو", "وشو", "شو"},
    # "see / look" — canonical: bare imperative
    "شوف": {"شوفوا", "شوفي", "شف", "طالع"},
    # "a little"
    "شوي": {"شويه", "شوية"},
    # "inside"
    "جوا": {"جوه", "جو", "داخل"},
    # "one (f.)"
    "واحده": {"وحده", "واحدة", "وحدة"},
    # "honestly"
    "صراحه": {"بصراحه", "بصراحة", "صراحة"},
    # "and then / after"
    "بعدين": {"وبعدين", "بعدان"},
    # "they"
    "هم": {"هما", "اهم", "همه"},
    # "by God" → keep الله, drop و prefix
    "الله": {"والله", "بالله", "تالله", "لله"},
    # "which" (relative)
    "اللي": {"الي", "الذي", "التي", "الذين", "اللاتي", "اللواتي"},
    # "now"
    "الحين": {"الحين", "دلحين", "هلحين", "الان", "الآن", "تو", "توه"},
    # "here"
    "هنا": {"هني", "هان", "هنا"},
    # "there"
    "هناك": {"هناك", "هناكي", "هناكا"},
    # "not" (dialect negation)
    "مو": {"مش", "ما", "موب", "مهو"},
    # "because"
    "عشان": {"عليشان", "علشان", "لان", "لانه", "علي", "عل"},
    # "what" (MSA)
    "ماذا": {"ماذا", "ماهو", "ما", "وش", "ايش", "شو"},
    # "where"
    "وين": {"فين", "وين", "فينه", "وينه"},
    # "come"
    "تعال": {"تعال", "تعالي", "تعالوا", "تعالو"},
    # "go"
    "روح": {"روح", "روحي", "روحو", "راح"},
    # "doctor"
    "الدكتور": {"دكتور", "الدكتورة", "دكتورة"},
    # "Mr."
    "الاستاذ": {"استاذ", "الاستاذة", "استاذة", "استاذه"},
    # name abbreviations
    # "عبدالعزيز": {"عبد", "ابو", "بو", "عبدالعزيز"},
    # ^ DISABLED (2026-08-17, per external evaluation report "Saudi Arabic ASR
    #   Evaluation"): mapping the bare fragment "عبد" → "عبدالعزيز" makes
    #   normalize() expand a BPE fragment into a high-frequency full name.
    #   BPE splits عبدالرحمن as ▁عبد + ال + ر + ح + م + ن, so a hyp like
    #   "عبد الرحمن القحطاني" was displayed as "عبدالعزيز رحمن قحطاني" —
    #   the evaluator's example of proper-name hallucination.  Do not map
    #   fragments/abbreviations to full names.
    "عبدالله": {"عبدالله", "عبود", "عبادي"},
    # "brother"
    "اخو": {"اخ", "اخو", "اخوي", "خوي"},
    # "father"
    "ابو": {"اب", "ابو", "ابوي", "بوبا"},
    # "good / OK"
    "طيب": {"طيب", "طيبب", "طيبون", "زين", "كويس", "تمام"},
    # "also" — canonical: بعد (dominant Saudi form).
    # Direction fixed 2026-08-17, per external evaluation report: the old
    # entry made بعد a *variant* of كمان, so normalize() rewrote every بعد
    # into كمان — the evaluator saw "بعد شوي → كمان شوي" / "بعد نص ساعة →
    # كمان نص ساعة" and concluded the model was contaminated with
    # Levantine/Egyptian vocabulary.  Native SAU data shows بعد (138) ≫
    # كمان (8) — see the SAU normalizer docstring; the mapping
    # should be كمان → بعد, not the reverse.
    "بعد": {"بعد", "كمان", "برضه", "برضو"},
    # verb ending normalization
    "بدات": {"بدت", "بدئت", "بدات"},
    "وصلنا": {"ووصلنا", "وصلنا"},
    "كانت": {"فكانت", "كانت"},
    "جاء": {"جا", "جاء"},
    # Extra dialect variants of common prepositions
    "فيه": {"فايه"},  # "in it" dialect spelling variant
}

# Build reverse mapping: any variant → canonical.
# Order matters: later canonical entries should NOT overwrite earlier
# variant mappings.  e.g. "فيه" is a variant of "في" (first), but also a
# canonical for "فايه" (later).  We keep the first mapping.
_VARIANT_TO_CANONICAL: dict = {}
for canonical, variants in _DIALECT_CANONICAL.items():
    if canonical not in _VARIANT_TO_CANONICAL:
        _VARIANT_TO_CANONICAL[canonical] = canonical
    for v in variants:
        if v not in _VARIANT_TO_CANONICAL:
            _VARIANT_TO_CANONICAL[v] = canonical


# ══════════════════════════════════════════════════════════════════════
# Stage 2: Number form standardization
# ══════════════════════════════════════════════════════════════════════

# Number words — all forms map to a canonical form
_NUMBER_MAP = {
    # 1-10
    "واحد": "واحد", "وحده": "واحده", "وحدة": "واحده",
    "اثنين": "اثنين", "اثنان": "اثنين", "ثنين": "اثنين",
    "ثلاثه": "ثلاث", "ثلاثة": "ثلاث", "تلاته": "ثلاث", "تلات": "ثلاث",
    "اربعه": "اربع", "اربعة": "اربع", "اربعا": "اربع",
    "خمسه": "خمس", "خمسة": "خمس",
    "سته": "ست", "ستة": "ست", "سته": "ست",
    "سبعه": "سبع", "سبعة": "سبع",
    "ثمانيه": "ثمان", "ثمانية": "ثمان",
    "تسعه": "تسع", "تسعة": "تسع",
    "عشره": "عشر", "عشرة": "عشر",
    # Tens
    "عشرين": "عشرين", "عشرون": "عشرين",
    "ثلاثين": "ثلاثين", "ثلاثون": "ثلاثين", "تلاتين": "ثلاثين",
    "اربعين": "اربعين", "اربعون": "اربعين",
    "خمسين": "خمسين", "خمسون": "خمسين",
    "ستين": "ستين", "ستون": "ستين",
    "سبعين": "سبعين", "سبعون": "سبعين",
    "ثمانين": "ثمانين", "ثمانون": "ثمانين",
    "تسعين": "تسعين", "تسعون": "تسعين",
    # Hundreds
    "ميه": "ميه", "مائة": "ميه", "مئه": "ميه", "مائه": "ميه",
    "ميتين": "ميتين", "مئتان": "ميتين", "مائتين": "ميتين",
    "ثلاثميه": "ثلاث ميه", "ثلاثمائة": "ثلاث ميه",
    "اربعميه": "اربع ميه", "اربعمائة": "اربع ميه",
    "خمسميه": "خمس ميه", "خمسمائة": "خمس ميه",
    "ستميه": "ست ميه", "ستمائة": "ست ميه",
    "سبعميه": "سبع ميه", "سبعمائة": "سبع ميه",
    "ثمانميه": "ثمان ميه", "ثمانمائة": "ثمان ميه",
    "تسعميه": "تسع ميه", "تسعمائة": "تسع ميه",
    # Thousands
    "الف": "الف", "الاف": "الف",
    "الفين": "الفين", "الفان": "الفين",
    "ثلاثه الاف": "ثلاث الاف", "ثلاثة الاف": "ثلاث الاف",
    # Large
    "مليون": "مليون", "مليونات": "مليون",
    "مليار": "مليار", "مليارات": "مليار",
}

# Compound numbers: وثمانين → و + ثمانين → و + canonical
# We handle these by first splitting on و then mapping each part

# ══════════════════════════════════════════════════════════════════════
# Stage 3: Filler / hesitation / disfluency removal
# ══════════════════════════════════════════════════════════════════════

_FILLER_WORDS = frozenset({
    # Stretched hesitations
    "ااا", "اا", "آآآ", "اه", "ااه", "اااه",
    "امم", "اممم", "إممم", "أمم", "ممم", "هممم",
    "هيه", "ههه", "هههه", "هاه", "هه", "هوه",
    "ششش",
    # Vocal fillers / discourse markers (pure filler, no content)
    "يعني", "بس", "خلاص", "المهم", "اها", "اوكي",
    "ماشي", "ايوا", "ها", "حاجة", "بص", "حسنا",
    "كويس", "زين", "عد",
    # Vocative particles (pure function, remove for WER)
    "يا", "وي", "وييي", "ألو", "لك",
    # Loose prefixes (BPE fragmentation artifacts)
    "فـ", "بـ", "لـ", "ال", "لل", "بال", "كال", "وال",
    "فلل", "وبال", "فبال",
})

# ══════════════════════════════════════════════════════════════════════
# Regex patterns (compiled once)
# ══════════════════════════════════════════════════════════════════════

_RE_DIACRITICS = re.compile(r"[ً-ْ]")
_RE_PUNCT_SYMBOL = re.compile(r"[\p{P}\p{S}]")
_RE_TATWEEL = re.compile(r"ـ")
_RE_ALEF_VARIANTS = re.compile(r"[إأآٱ]")
_RE_ALEF_MAKSURA = re.compile(r"ى")
_RE_TEH_MARBUTA = re.compile(r"ة")
_RE_WAW_HAMZA = re.compile(r"ؤ")
_RE_YEH_HAMZA = re.compile(r"ئ")
_RE_INDEPENDENT_HAMZA = re.compile(r"ء")
_RE_PERSIAN_PEH = re.compile(r"پ")
_RE_PERSIAN_VEH = re.compile(r"ڤ")
_RE_START_AND = re.compile(r"^و")  # leading و (and) for number decomposition
_RE_MULTI_SPACE = re.compile(r"\s+")


def _normalize_compound_number(word: str) -> str:
    """
    Normalize compound Saudi number words.

    - Drops ال (definite article) from numbers: العشرين → عشرين
    - Preserves و (and): وعشرين → وعشرين
    - Normalizes the number root: اربعه → اربع, مائه → ميه

    BUGFIX 2026-08-17, per external evaluation report: the old code stripped
    ال/و from ANY word with that prefix and never restored it when the
    remainder was not a number, silently rewriting ordinary nouns during WER
    normalization and in the HF Space display, e.g.:
        التقرير → تقرير,  الوزارة → وزاره,  السيارة → سياره,
        المبلغ الإجمالي → مبلغ اجمالي,  الله → له
    — exactly the "systematic ال omission" substitutions reported by the
    evaluator (27/55 substitutions).  The prefix is now only removed when
    the remainder is actually a number word.
    """
    if word.startswith("وال"):
        waw_prefix = "و"   # conjunction و — we preserve this
        rest = word[3:]
    elif word.startswith("ال") and len(word) > 3:
        waw_prefix = ""
        rest = word[2:]
    elif word.startswith("و") and len(word) > 1:
        waw_prefix = "و"
        rest = word[1:]
    else:
        return word

    if rest not in _NUMBER_MAP:
        # Not a number word — the ال/و prefix belongs to the word itself
        # (e.g. التقرير, السيارة), so keep the original form.
        return word

    return waw_prefix + _NUMBER_MAP[rest]


# ══════════════════════════════════════════════════════════════════════
# Main normalize function
# ══════════════════════════════════════════════════════════════════════

def normalize(text: str) -> str:
    """
    Full Saudi Arabic text normalization pipeline.

    Stages:
      1. Remove paralinguistic tags ([breath], [laugh], # أأأ, etc.)
      2. Character-level cleanup (diacritics, alef/hamza, tatweel)
      3. Word-level dialect → canonical mapping (هذه→هذي etc.)
      4. Number form standardization
      5. Filler / hesitation removal
      6. Single-char fragment removal
      7. Whitespace collapse
    """
    if text is None:
        return ""
    if not text.strip():
        return ""

    # ── Stage 1: Remove paralinguistic tags ──
    text = remove_paralinguistic_tags(text)

    # ── Stage 2: Character-level cleanup ──
    # Remove diacritics
    text = _RE_DIACRITICS.sub("", text)

    # Normalize alef variants: إ أ آ ٱ → ا
    text = _RE_ALEF_VARIANTS.sub("ا", text)

    # Alef maksura → yeh
    text = _RE_ALEF_MAKSURA.sub("ي", text)

    # Teh marbuta → heh
    text = _RE_TEH_MARBUTA.sub("ه", text)

    # Waw with hamza → waw
    text = _RE_WAW_HAMZA.sub("و", text)

    # Yeh with hamza → yeh (conservative — some dialects differ)
    # text = _RE_YEH_HAMZA.sub("ي", text)
    # Actually keep ئ as is for Saudi

    # Independent hamza → remove
    text = _RE_INDEPENDENT_HAMZA.sub("", text)

    # Persian letters → Arabic
    text = _RE_PERSIAN_PEH.sub("ب", text)
    text = _RE_PERSIAN_VEH.sub("ف", text)

    # Remove tatweel
    text = _RE_TATWEEL.sub("", text)

    # Remove punctuation and symbols
    text = _RE_PUNCT_SYMBOL.sub("", text)

    # ── Stage 3-5: Word-level normalization ──
    words = text.split()
    normalized = []
    for w in words:
        if not w:
            continue

        # Stage 5: Skip pure filler words
        if w in _FILLER_WORDS:
            continue

        # Stage 5b: Skip single-char fragments
        if len(w) <= 1:
            continue

        # Stage 5c: Remove stretched characters (ااا+, ههه+, ممم+)
        w_stripped = re.sub(r"(.)\1{2,}", "", w)
        if not w_stripped:
            continue
        w = w_stripped

        # Stage 3: Dialect → canonical mapping
        if w in _VARIANT_TO_CANONICAL:
            w = _VARIANT_TO_CANONICAL[w]

        # Stage 4: Number normalization
        w = _normalize_compound_number(w)

        # Stage 4b: Full number mapping
        if w in _NUMBER_MAP:
            w = _NUMBER_MAP[w]

        if w:
            normalized.append(w)

    text = " ".join(normalized)

    # ── Stage 7: Collapse whitespace ──
    text = _RE_MULTI_SPACE.sub(" ", text).strip()

    return text


def arabic_text_normalize(text):
    """
    Word-level filler and stretch removal.  Called separately by the decode
    pipeline for WER normalization.

    Accepts str or list of words; returns list of words.
    """
    if isinstance(text, str):
        words = text.split()
    else:
        words = list(text)

    result = []
    for w in words:
        # Remove stretched characters
        w_stripped = re.sub(r"(.)\1{2,}", "", w)
        if not w_stripped:
            continue
        # Remove filler words
        if w_stripped in _FILLER_WORDS:
            continue
        # Skip single chars
        if len(w_stripped) <= 1:
            continue
        result.append(w_stripped)

    return result


if __name__ == "__main__":
    examples = [
        # Paralinguistic tags
        "اللهم صلِّ وسلمْ [breath] على سيدِنا محمد، أهلاً وسهلاً",
        "أأأ يعني هذي مشكلة لياقـيّة [laugh]",
        # Dialect ↔ MSA pairs
        "هذه المدينه رقم واحد في جذب السياح",
        "هذي المدينه رقم واحد في جذب السياح",
        "السياح الدوليين موزعين في كافه مزارات المنطقه",
        "السياح الدوليين موزعين فيه كافه مزارات المنطقه",
        "بنسبه نمو وصلت الي ميه وسته وستين في المئه",
        "وفقا لاحصاءات وزاره السياحه الصادره في عام عشرين ثلاثه وعشرين",
        # Numbers
        "اربع ميه وواحد وتسعين في الميه",
        "اربعه مليون وخمسميه الف سايح",
        # Fillers
        "ااا اممم يعني اه شوف طيب خلاص",
        # Single char fragments (BPE artifacts)
        "ال ب ا م ي السلام عليكم",
        # Real examples from error file
        "للباحثين عن الوجهات المميزه بخريطه متنوعه من التجارب السياحيه",
        "تتجه البوصله دائما نحو عروس الشمال حائل",
        "وعراقه تاريخها وتضم الكثير من مواقع التراث الوطني",
        "حرف تعكس صبر طويل وفن متوارث",
        "وانا انبسط لاني اخذ راحتي في الطبخ",
    ]

    print("原始 → 标准化后\n")
    for ex in examples:
        normed = normalize(ex)
        print(f"  IN:  {ex}")
        print(f"  OUT: {normed}")
        print()

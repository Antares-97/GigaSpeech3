"""Dialect-token (language-ID) conditioning helpers for the ARAB model.

Design:
  - The tokenizer vocabulary is left untouched; N extra token IDs are reserved AFTER
    sp.get_piece_size(): [offset, offset+N) for DIALECT_TOKENS.
  - Training (train_dialect.py) prepends the dialect token ID of each cut
    to its target sequence:  y = [<D>, tok1, tok2, ...].
  - Decoding (decode_dialect.py) strips a leading dialect token from hyps
    before text conversion (the token was forced into every training target,
    so the model emits it at position 0 with high reliability).

Token set: DZA EGY IRQ SYR SAU MSA + ARE.  ARE has no training data in the
ARAB set yet — its ID is reserved so that adding ARE data later does NOT
change vocab size (joiner dim stays fixed, checkpoints stay compatible).
Until ARE data exists the token is never a training target; decoding ARE
audio emits one of the six trained tokens, which is stripped as usual.
"""
from typing import List, Optional, Tuple

DIALECT_TOKENS = ["DZA", "EGY", "IRQ", "SYR", "SAU", "MSA", "ARE"]
DIALECT_INDEX = {d: i for i, d in enumerate(DIALECT_TOKENS)}

_ALIASES = {
    "DZA": "DZA", "DZ": "DZA", "DZAERIA": "DZA",
    "EGYPT": "EGY", "EGYPTIAN": "EGY",
    "IRAQ": "IRQ",
    "SYRIA": "SYR",
    "SAUDI": "SAU", "SA": "SAU", "KSA": "SAU",
    "MSA": "MSA",
    "ARE": "ARE", "UAE": "ARE", "EMIRATI": "ARE",
    "ARAB": None,
}


def resolve_dialect(x) -> Optional[str]:
    """Map a dialect label (cut.custom['dialect'], supervision language
    string, filename-derived code, ...) to a DIALECT_TOKENS code, or None
    when the dialect has no token (ARE, unknown labels)."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        # pseudo-label cuts store a float score in supervision.language
        return None
    code = str(x).strip().upper()
    if code in DIALECT_INDEX:
        return code
    return _ALIASES.get(code, None)


def num_tokens() -> int:
    return len(DIALECT_TOKENS)


# Dialect-embedding index for "no dialect": reserved slot AFTER the real
# dialects.  Used for ARE/unknown cuts and un-conditioned decoding.
NEUTRAL_ID = len(DIALECT_TOKENS)


def token_or_neutral(dialect_label) -> int:
    """Dialect embedding index for a label: DIALECT_INDEX[d] for dialects
    with training data, NEUTRAL_ID otherwise (ARE / unknown / None).  The
    ARE embedding slot exists only to keep the vocab fixed — it is never a
    training target, so injecting it would mean an untrained vector."""
    d = resolve_dialect(dialect_label)
    if d is None or d == "ARE":
        return NEUTRAL_ID
    return DIALECT_INDEX[d]


def token_offset(sp) -> int:
    """First reserved dialect token ID = tokenizer piece count."""
    return sp.get_piece_size()


def token_piece(dialect: str) -> str:
    return f"<{dialect}>"


def token_id(sp, dialect: str) -> int:
    return token_offset(sp) + DIALECT_INDEX[dialect]


def prepend_token(sp, token_ids: List[int], dialect_label) -> List[int]:
    """Prepend the dialect token ID for the given cut label (no-op when the
    dialect has no token, e.g. ARE)."""
    d = resolve_dialect(dialect_label)
    if d is None:
        return token_ids
    return [token_id(sp, d)] + token_ids


def strip_token(sp, token_ids: List[int]) -> Tuple[List[int], Optional[str]]:
    """Remove a leading dialect token from a hypothesis.
    Returns (stripped_ids, emitted_dialect_or_None)."""
    if not token_ids:
        return token_ids, None
    off = token_offset(sp)
    first = token_ids[0]
    if off <= first < off + num_tokens():
        return token_ids[1:], DIALECT_TOKENS[first - off]
    return token_ids, None


def build_id2piece(sp):
    """id -> piece dict extended with the dialect token pieces (for decode
    text conversion / logging)."""
    id2piece = {i: sp.id_to_piece(i) for i in range(sp.get_piece_size())}
    for d in DIALECT_TOKENS:
        id2piece[token_id(sp, d)] = token_piece(d)
    return id2piece

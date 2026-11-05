"""
Shared Arabic text-normalization helpers.

Only the character-level cleanup shared by all dialects remains here
(see ``_common.py``).  Dialect-level normalization lives in
``asr/textnorm/{DIALECT}.py`` and is loaded by
``asr/recipe_textnorm.py``.
"""

from ._common import normalize_char_level, remove_paralinguistic_tags

__all__ = ["normalize_char_level", "remove_paralinguistic_tags"]

"""
Auto-discover and load dialect-specific text normalization.

Each dialect provides a {DIALECT}.py with a normalize() function:
    asr/textnorm/{DIALECT}.py

Usage:
    from recipe_textnorm import get_recipe_normalizer
    norm = get_recipe_normalizer("EGY")
    if norm:
        text = norm.normalize(text)
"""

import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple

# ── Project layout ────────────────────────────────────────────────────────
_ASR_DIR = Path(__file__).resolve().parent          # asr/
_TEXTNORM_DIR = _ASR_DIR / "textnorm"

# Ensure asr/ is on sys.path so dialect scripts can import
# textnorm._common (which lives at asr/textnorm/_common.py).
_ASR_DIR_STR = str(_ASR_DIR)
if _ASR_DIR_STR not in sys.path:
    sys.path.insert(0, _ASR_DIR_STR)

# ── Cache of already-imported modules ─────────────────────────────────────
_loaded_modules: dict = {}


def _get_recipe_script_path(dialect: str) -> Optional[Path]:
    """Resolve the path to asr/textnorm/{DIALECT}.py."""
    dialect_upper = dialect.upper()
    script_path = _TEXTNORM_DIR / f"{dialect_upper}.py"
    if script_path.is_file():
        return script_path
    return None


def _load_recipe_module(dialect: str) -> Optional["module"]:
    """
    Dynamically import asr/textnorm/{DIALECT}.py as a module.

    Returns the module object, or None if the script doesn't exist.
    """
    dialect_upper = dialect.upper()

    if dialect_upper in _loaded_modules:
        return _loaded_modules[dialect_upper]

    script_path = _get_recipe_script_path(dialect_upper)
    if script_path is None:
        return None

    module_name = f"_recipe_textnorm_{dialect_upper.lower()}"

    try:
        spec = importlib.util.spec_from_file_location(module_name, str(script_path))
        if spec is None or spec.loader is None:
            logging.warning(f"Could not load spec for {script_path}")
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        _loaded_modules[dialect_upper] = module

        has_norm = hasattr(module, "normalize") and callable(module.normalize)
        has_word = hasattr(module, "arabic_text_normalize") and callable(
            module.arabic_text_normalize
        )
        extras = []
        if has_norm: extras.append("normalize()")
        if has_word: extras.append("arabic_text_normalize()")
        logging.info(
            f"Loaded text norm from {script_path} "
            f"({', '.join(extras) if extras else 'no normalize() found'})"
        )
        return module
    except Exception as e:
        logging.warning(f"Failed to load dialect textnorm {script_path}: {e}")
        return None


def has_recipe_textnorm(dialect: str) -> bool:
    """Check whether a dialect textnorm script exists for the given dialect."""
    return _get_recipe_script_path(dialect) is not None


def get_recipe_normalizer(dialect: str) -> Optional[Callable[[str], str]]:
    """
    Return the normalize() function from the dialect script, or None.

    The returned callable has signature: normalize(text: str) -> str
    """
    module = _load_recipe_module(dialect)
    if module is None:
        return None

    if hasattr(module, "normalize") and callable(module.normalize):
        return module.normalize

    logging.warning(
        f"Recipe textnorm for {dialect} has no normalize() function"
    )
    return None


def get_recipe_normalizer_with_word_level(
    dialect: str,
) -> Tuple[Optional[Callable[[str], str]], Optional[Callable]]:
    """
    Return (normalize, arabic_text_normalize) from the recipe script.

    - normalize(text: str) -> str  — full text normalization
    - arabic_text_normalize(text) -> str or list — word-level filler/stretch removal

    Either may be None if the function isn't defined.
    """
    module = _load_recipe_module(dialect)
    if module is None:
        return None, None

    norm_fn = getattr(module, "normalize", None)
    if norm_fn is not None and not callable(norm_fn):
        norm_fn = None

    word_fn = getattr(module, "arabic_text_normalize", None)
    if word_fn is not None and not callable(word_fn):
        word_fn = None

    return norm_fn, word_fn

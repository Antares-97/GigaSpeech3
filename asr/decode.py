#!/usr/bin/env python3
#
# Copyright 2021-2023 Xiaomi Corporation (Author: Fangjun Kuang,
#                                                 Zengwei Yao)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Usage:
(1) greedy search
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method greedy_search

(2) beam search (not recommended)
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method beam_search \
    --beam-size 4

(3) modified beam search
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method modified_beam_search \
    --beam-size 4

(4) fast beam search (one best)
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method fast_beam_search \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64

(5) fast beam search (nbest)
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method fast_beam_search_nbest \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64 \
    --num-paths 200 \
    --nbest-scale 0.5

(6) fast beam search (nbest oracle WER)
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method fast_beam_search_nbest_oracle \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64 \
    --num-paths 200 \
    --nbest-scale 0.5

(7) fast beam search (with LG)
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method fast_beam_search_nbest_LG \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64
"""


import argparse
import copy
import logging
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import lhotse
import regex
import sentencepiece as spm
import torch
import torch.nn as nn
from datamodule import ArabicAsrDataModule
from recipe_textnorm import get_recipe_normalizer_with_word_level
from pipeline_decode import add_pipeline_arguments, export_raw_hypotheses
from train import (
    MULTI_BPE_DIALECTS,
    add_model_arguments as add_asr_model_args,
    get_model as get_asr_model,
    get_params as get_asr_params,
    load_multi_bpe,
    _dialect_of_cut,
    _encode_text_multi_bpe,
    _multi_bpe_ctx,
)
from finetune import add_model_arguments as add_finetune_model_args, get_model as get_finetune_model, get_params as get_finetune_params
from zipformer.beam_search import (
    beam_search,
    fast_beam_search_nbest,
    fast_beam_search_nbest_LG,
    fast_beam_search_nbest_oracle,
    fast_beam_search_one_best,
    greedy_search,
    greedy_search_batch,
    modified_beam_search,
    modified_beam_search_LODR,
    modified_beam_search_lm_rescore,
    modified_beam_search_lm_rescore_LODR,
    modified_beam_search_lm_shallow_fusion,
)
from zipformer.dataset import HubertAsrDataset

from icefall import ContextGraph, LmScorer, NgramLm
from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    setup_logger,
    store_transcripts,
    str2bool,
    write_error_stats,
)

LOG_EPS = math.log(1e-10)

# ── Arabic text normalization (merged from vietasr-saudi ASR + SSL) ──────────

# Filler / disfluency words to remove before WER computation
_STOP_WORDS = frozenset({
    "ااا", "آآآ", "اممم", "إممم", "ممم", "أمم", "هممم",
    "إيه", "إي", "إييه",
    "آ", "آه", "آها", "أوه", "أوو", "إيوا", "إيوة",
    "هيه", "ههه", "هاه",
    "طيب", "يعني", "بس", "أها", "المهم", "خلاص", "تمام",
    "بصراحة", "شوف", "والله",
    "يا", "وي", "وَيْ", "وييي", "ألو", "لك",
    "شو", "هيك",
    "هاك", "باه", "هاو", "ياودي", "هاديك",
    "زين", "عد",
    "كذا", "كذا يعني", "زي", "مثل",
    "ششش", "هههه", "مممم",
    "اا", "ام", "اه", "ه", "او",
    "حاجة", "بص", "حسنا", "اوكي", "ماشي", "ايوا", "ها", "كويس",
})

# Stretched character patterns
_LONG_AAA_RE = re.compile(r"ا{3,}")
_LONG_HHH_RE = re.compile(r"ه{3,}")
_LONG_MMM_RE = re.compile(r"م{3,}")
_LONG_ALIF_RE = re.compile(r"آ{2,}")


def _normalize_arabic_script(text: str) -> str:
    """Unicode-level Arabic script cleanup (from SSL decode.py)."""
    # Remove Latin characters
    text = re.sub(r"[A-Za-zÀ-ÿ]", "", text)

    # Fullwidth → ASCII digits
    text = text.translate(str.maketrans("０１２３４５６７８９", "0123456789"))

    # Eastern Arabic → ASCII digits
    text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))

    # Remove Arabic decimal/thousands separators
    text = re.sub(r"[٬٫]", "", text)

    # Remove punctuation
    text = re.sub(
        r'[،؛؟٪-٭۔\`\·\^\$\+\=\~\%\*\#\|\.\,\!\?\:\;\-\_\(\)\[\]\{\}\"\'\/\\،؛؟…""«»]',
        "",
        text,
    )

    # Remove emoji
    text = re.sub(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        r"\U0001F1E0-\U0001F1FF\U00002700-\U000027BF\U0001F900-\U0001F9FF"
        r"\U00002600-\U000026FF\U00002B00-\U00002BFF]+",
        "",
        text,
    )

    # Remove invisible / bidi-control characters
    text = re.sub(r"[​-‏‪-‮⁦-⁩]", "", text)

    # Remove diacritics (harakat)
    text = re.sub(r"[ً-ْ]", "", text)

    # Remove tatweel + other rare diacritics
    text = re.sub(r"[ـّٕٓٔ؜٫٬ٱ]", "", text)

    # Collapse whitespace
    return re.sub(r"\s+", " ", text).strip()


def _arabic_text_normalize(words: List[str]) -> List[str]:
    """Remove filler words and compress stretched characters."""
    filtered = []
    for w in words:
        w = _LONG_AAA_RE.sub("", w)
        w = _LONG_HHH_RE.sub("", w)
        w = _LONG_MMM_RE.sub("", w)
        w = _LONG_ALIF_RE.sub("", w)
        if w and w not in _STOP_WORDS:
            filtered.append(w)
    return filtered


# Track which dialect normalizer has been logged (avoid spam in decode loop)
_normalizer_logged: set = set()


def _normalize_for_wer(params, text_or_words):
    """Full normalization pipeline for WER/CER computation.

    Uses asr/textnorm/{DIALECT}.py normalize() when available,
    falling back to the built-in pipeline (script cleanup → filler removal).

    Accepts str or list of words; returns list of words.
    """
    dialect = getattr(params, "dialect", "SAU")

    if isinstance(text_or_words, list):
        text = " ".join(text_or_words)
    else:
        text = text_or_words

    if not text or not text.strip():
        return []

    # Raw mode: keep decoded words for the final raw export and external
    # benchmark normalizer. Internal WER is normalized later by _wer_results.
    if getattr(params, "keep_raw_text", False) or getattr(
        params, "raw_transcripts", False
    ):
        return text.split()

    # ── Try recipe textnorm first ──────────────────────────────────────
    recipe_norm, recipe_word_norm = get_recipe_normalizer_with_word_level(dialect)
    if recipe_norm is not None:
        if dialect not in _normalizer_logged:
            _normalizer_logged.add(dialect)
            import recipe_textnorm as _rt
            script_path = _rt._get_recipe_script_path(dialect)
            logging.info(
                f"WER normalization: loaded text norm from {script_path}"
            )
        # Recipe script defines the complete normalization for this dialect
        text = recipe_norm(text)
        if not text.strip():
            return []
        # Apply word-level filler/stretch removal if the recipe provides it
        if recipe_word_norm is not None:
            result = recipe_word_norm(text)
            if isinstance(result, list):
                return result
            elif isinstance(result, str):
                return result.split()
            return result
        return text.split()

    # ── Fallback: built-in pipeline ────────────────────────────────────
    if dialect == "ENG":
        # No ENG recipe normalizer exists: score on raw whitespace tokens.
        return text.split()

    # Stage 1: Unicode-level Arabic cleanup
    text = _normalize_arabic_script(text)

    # Stage 2: Word-level filler/stretch cleanup
    words = _arabic_text_normalize(text.split())

    # Stage 3: dialect-specific normalization is provided by recipe scripts
    # (handled above).  Without one, keep the built-in cleanup result.
    return words


# ── Multi-BPE (union vocab) decode helpers ──────────────────────────────────
# MULTI_BPE=1 训练的词表是各方言 500-piece BPE 的去重并集（union vocab）。
# 解码时 hyp 是 union ID：文本用全局 id2piece 表逐 piece 还原（共享 piece
# 字符串相同，无需按方言反查）；词汇约束用方言 mask（只允许目标方言的
# union ID + blank）。


def _union_tokens_to_text(hyp_tokens) -> str:
    """Restore union-vocab token IDs to text via the global id2piece table."""
    id2piece = _multi_bpe_ctx["id2piece"]
    return "".join(id2piece[i] for i in hyp_tokens).replace("▁", " ").strip()


def _decode_tokens(params, sp, hyp_tokens) -> List[str]:
    """Hypothesis token IDs -> normalized word list.

    Multi-bpe mode restores text from the union id2piece table; otherwise
    uses the single SentencePiece model (original behavior).
    """
    if getattr(params, "multi_bpe", False):
        return _normalize_for_wer(params, _union_tokens_to_text(hyp_tokens).split())
    return _normalize_for_wer(params, sp.decode(hyp_tokens).split())


def is_arabic_only(cut) -> bool:
    """Filter: keep only cuts whose text contains Arabic script."""
    text = cut.supervisions[0].text.strip()
    if not text:
        return False
    if not regex.search(r"\p{Arabic}", text):
        return False
    # Remove legal Arabic + digits + punctuation + spaces; anything left is non-Arabic
    patch = r"ـًٍَُِّْ"
    allowed = rf"[\p{{Arabic}}{patch}\p{{Nd}}\p{{P}}\p{{S}}\p{{Z}}]+"
    remaining = regex.sub(allowed, "", text)
    return len(remaining) == 0


def _detect_model_type(checkpoint_path: Path) -> str:
    """Detect model type from a checkpoint state dict.

    Returns 'asr' if the checkpoint contains encoder_embed parameters
    (Conv2dSubsampling used in train.py), or 'finetune' otherwise
    (HubertModel encoder without Conv2d subsampling, from finetune.py).
    """
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    for key in state.keys():
        if key.startswith("encoder_embed."):
            return "asr"
    return "finetune"


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler="resolve",
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=15,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )

    parser.add_argument(
        "--use-averaged-model",
        type=str2bool,
        default=True,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--cuts-name",
        type=str,
        default=None,
        help="The cuts name. It is used to load the corresponding dataset",
    )

    parser.add_argument(
        "--test-cuts-only",
        type=str2bool,
        default=False,
        help="With --cuts-name test, decode ONLY the test split and skip dev "
        "(default: false — the datamodule loads every available split, i.e. "
        "dev+test when both files exist).",
    )

    parser.add_argument(
        "--cuts-path",
        type=str,
        default="",
        help="The cuts path, used when cuts_name is x",
    )

    parser.add_argument(
        "--old-cut-prefix",
        type=str,
        help="""For path mapping""",
    )

    parser.add_argument(
        "--new-cut-prefix",
        type=str,
        help="""For path mapping""",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipformer/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--decode-dir",
        type=str,
        default=None,
        help="The decode dir",
    )

    parser.add_argument(
        "--tokenizer-model",
        dest="bpe_model",
        metavar="TOKENIZER_MODEL",
        type=str,
        default="data/lang_bpe_500/tokenizer.model",
        help="Path to the SentencePiece unigram tokenizer model",
    )
    # Backward-compatible spelling; hidden from --help.
    parser.add_argument(
        "--bpe-model",
        dest="bpe_model",
        metavar="TOKENIZER_MODEL",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--lang-dir",
        type=Path,
        default="data/lang_bpe_500",
        help="The lang dir containing word table and LG graph",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="""Possible values are:
          - greedy_search
          - beam_search
          - modified_beam_search
          - modified_beam_search_LODR
          - fast_beam_search
          - fast_beam_search_nbest
          - fast_beam_search_nbest_oracle
          - fast_beam_search_nbest_LG
        If you use fast_beam_search_nbest_LG, you have to specify
        `--lang-dir`, which should contain `LG.pt`.
        """,
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
        help="""An integer indicating how many candidates we will keep for each
        frame. Used only when --decoding-method is beam_search or
        modified_beam_search.""",
    )

    parser.add_argument(
        "--beam",
        type=float,
        default=20.0,
        help="""A floating point value to calculate the cutoff score during beam
        search (i.e., `cutoff = max-score - beam`), which is the same as the
        `beam` in Kaldi.
        Used only when --decoding-method is fast_beam_search,
        fast_beam_search_nbest, fast_beam_search_nbest_LG,
        and fast_beam_search_nbest_oracle
        """,
    )

    parser.add_argument(
        "--ngram-lm-scale",
        type=float,
        default=0.01,
        help="""
        Used only when --decoding-method is fast_beam_search_nbest_LG.
        It specifies the scale for n-gram LM scores.
        """,
    )

    parser.add_argument(
        "--max-contexts",
        type=int,
        default=8,
        help="""Used only when --decoding-method is
        fast_beam_search, fast_beam_search_nbest, fast_beam_search_nbest_LG,
        and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=64,
        help="""Used only when --decoding-method is
        fast_beam_search, fast_beam_search_nbest, fast_beam_search_nbest_LG,
        and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; " "2 means tri-gram",
    )
    parser.add_argument(
        "--max-sym-per-frame",
        type=int,
        default=1,
        help="""Maximum number of symbols per frame.
        Used only when --decoding-method is greedy_search""",
    )

    parser.add_argument(
        "--num-paths",
        type=int,
        default=200,
        help="""Number of paths for nbest decoding.
        Used only when the decoding method is fast_beam_search_nbest,
        fast_beam_search_nbest_LG, and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--nbest-scale",
        type=float,
        default=0.5,
        help="""Scale applied to lattice scores when computing nbest paths.
        Used only when the decoding method is fast_beam_search_nbest,
        fast_beam_search_nbest_LG, and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--use-shallow-fusion",
        type=str2bool,
        default=False,
        help="""Use neural network LM for shallow fusion.
        If you want to use LODR, you will also need to set this to true
        """,
    )

    parser.add_argument(
        "--lm-type",
        type=str,
        default="rnn",
        help="Type of NN lm",
        choices=["rnn", "transformer"],
    )

    parser.add_argument(
        "--lm-scale",
        type=float,
        default=0.3,
        help="""The scale of the neural network LM
        Used only when `--use-shallow-fusion` is set to True.
        """,
    )

    parser.add_argument(
        "--tokens-ngram",
        type=int,
        default=2,
        help="""The order of the ngram lm.
        """,
    )

    parser.add_argument(
        "--backoff-id",
        type=int,
        default=500,
        help="ID of the backoff symbol in the ngram LM",
    )

    parser.add_argument(
        "--context-score",
        type=float,
        default=2,
        help="""
        The bonus score of each token for the context biasing words/phrases.
        Used only when --decoding-method is modified_beam_search and
        modified_beam_search_LODR.
        """,
    )

    parser.add_argument(
        "--context-file",
        type=str,
        default="",
        help="""
        The path of the context biasing lists, one word/phrase each line
        Used only when --decoding-method is modified_beam_search and
        modified_beam_search_LODR.
        """,
    )
    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=False,
        help="Whether to use half precision during decoding.",
    )
    # Register both ASR and finetune model args so all defaults are available.
    # (args with same name share defaults — finetune's registration wins for overlaps.)
    add_asr_model_args(parser)
    add_finetune_model_args(parser)
    add_pipeline_arguments(parser)

    return parser


def _normalize_hyp_list(params, hyps: List[List[str]]) -> List[List[str]]:
    """Apply full normalization pipeline to each hypothesis in a list."""
    return [_normalize_for_wer(params, words) for words in hyps]


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    batch: dict,
    word_table: Optional[k2.SymbolTable] = None,
    decoding_graph: Optional[k2.Fsa] = None,
    context_graph: Optional[ContextGraph] = None,
    LM: Optional[LmScorer] = None,
    ngram_lm=None,
    ngram_lm_scale: float = 0.0,
) -> Dict[str, List[List[str]]]:
    """Decode one batch and return the result in a dict. The dict has the
    following format:

        - key: It indicates the setting used for decoding. For example,
               if greedy_search is used, it would be "greedy_search"
               If beam search with a beam size of 7 is used, it would be
               "beam_7"
        - value: It contains the decoding result. `len(value)` equals to
                 batch size. `value[i]` is the decoding result for the i-th
                 utterance in the given batch.
    Args:
      params:
        It's the return value of :func:`get_params`.
      model:
        The neural model.
      sp:
        The SentencePiece tokenizer.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
      word_table:
        The word symbol table.
      decoding_graph:
        The decoding graph. Can be either a `k2.trivial_graph` or HLG, Used
        only when --decoding-method is fast_beam_search, fast_beam_search_nbest,
        fast_beam_search_nbest_oracle, and fast_beam_search_nbest_LG.
      LM:
        A neural network language model.
      ngram_lm:
        A ngram language model
      ngram_lm_scale:
        The scale for the ngram language model.
    Returns:
      Return the decoding result. See above description for the format of
      the returned dict.
    """
    multi_bpe = getattr(params, "multi_bpe", False)
    if multi_bpe and params.decoding_method not in (
        "greedy_search",
        "beam_search",
        "modified_beam_search",
    ):
        raise ValueError(
            f"Decoding method '{params.decoding_method}' is not supported "
            f"with multi-bpe (union vocab). Use greedy_search or "
            f"modified_beam_search."
        )
    allowed_ids = getattr(params, "decode_allowed_ids", None)

    device = next(model.parameters()).device
    supervisions = batch["supervisions"]
    use_amp = getattr(params, "use_fp16", False)

    # Auto-detect batch format: ASR (inputs + num_frames) vs Finetune (audio + padding_mask)
    if "inputs" in batch:
        feature = batch["inputs"]
        assert feature.ndim == 3
        feature = feature.to(device)
        feature_lens = supervisions["num_frames"].to(device)
    elif "audio" in batch:
        feature = batch["audio"]
        feature = feature.to(device)
        if "padding_mask" in batch:
            padding_mask = batch["padding_mask"]
            if isinstance(padding_mask, torch.Tensor):
                padding_mask = padding_mask.to(device)
                feature_lens = (~padding_mask).sum(dim=1).to(device)
            else:
                feature_lens = torch.tensor(padding_mask, device=device)
        elif "feature_lens" in batch:
            feature_lens = torch.tensor(batch["feature_lens"], device=device)
        else:
            raise RuntimeError("Finetune batch missing padding_mask or feature_lens")
    else:
        raise RuntimeError("Batch missing both 'inputs' and 'audio' keys")

    if params.causal:
        pad_len = 30
        feature_lens += pad_len
        feature = torch.nn.functional.pad(
            feature, pad=(0, 0, 0, pad_len), value=LOG_EPS,
        )

    with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
        encoder_out, encoder_out_lens = model.forward_encoder(feature, feature_lens)

    hyps = []

    if params.decoding_method == "fast_beam_search":
        hyp_tokens = fast_beam_search_one_best(
            model=model,
            decoding_graph=decoding_graph,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam,
            max_contexts=params.max_contexts,
            max_states=params.max_states,
        )
        for hyp in sp.decode(hyp_tokens):
            hyps.append(_normalize_for_wer(params, hyp.split()))
    elif params.decoding_method == "fast_beam_search_nbest_LG":
        hyp_tokens = fast_beam_search_nbest_LG(
            model=model,
            decoding_graph=decoding_graph,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam,
            max_contexts=params.max_contexts,
            max_states=params.max_states,
            num_paths=params.num_paths,
            nbest_scale=params.nbest_scale,
        )
        for hyp in hyp_tokens:
            hyps.append([word_table[i] for i in hyp])  # LG path: keep word symbols as-is
    elif params.decoding_method == "fast_beam_search_nbest":
        hyp_tokens = fast_beam_search_nbest(
            model=model,
            decoding_graph=decoding_graph,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam,
            max_contexts=params.max_contexts,
            max_states=params.max_states,
            num_paths=params.num_paths,
            nbest_scale=params.nbest_scale,
        )
        for hyp in sp.decode(hyp_tokens):
            hyps.append(_normalize_for_wer(params, hyp.split()))
    elif params.decoding_method == "fast_beam_search_nbest_oracle":
        hyp_tokens = fast_beam_search_nbest_oracle(
            model=model,
            decoding_graph=decoding_graph,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam,
            max_contexts=params.max_contexts,
            max_states=params.max_states,
            num_paths=params.num_paths,
            ref_texts=sp.encode(supervisions["text"]),
            nbest_scale=params.nbest_scale,
        )
        for hyp in sp.decode(hyp_tokens):
            hyps.append(_normalize_for_wer(params, hyp.split()))
    elif params.decoding_method == "greedy_search" and params.max_sym_per_frame == 1:
        hyp_tokens = greedy_search_batch(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            allowed_ids=allowed_ids,
        )
        for h in hyp_tokens:
            hyps.append(_decode_tokens(params, sp, h))
    elif params.decoding_method == "modified_beam_search":
        hyp_tokens = modified_beam_search(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam_size,
            context_graph=context_graph,
            allowed_ids=allowed_ids,
        )
        for h in hyp_tokens:
            hyps.append(_decode_tokens(params, sp, h))
    elif params.decoding_method == "modified_beam_search_lm_shallow_fusion":
        hyp_tokens = modified_beam_search_lm_shallow_fusion(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam_size,
            LM=LM,
        )
        for hyp in sp.decode(hyp_tokens):
            hyps.append(_normalize_for_wer(params, hyp.split()))
    elif params.decoding_method == "modified_beam_search_LODR":
        hyp_tokens = modified_beam_search_LODR(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam_size,
            LODR_lm=ngram_lm,
            LODR_lm_scale=ngram_lm_scale,
            LM=LM,
            context_graph=context_graph,
        )
        for hyp in sp.decode(hyp_tokens):
            hyps.append(_normalize_for_wer(params, hyp.split()))
    elif params.decoding_method == "modified_beam_search_lm_rescore":
        lm_scale_list = [0.01 * i for i in range(10, 50)]
        ans_dict = modified_beam_search_lm_rescore(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam_size,
            LM=LM,
            lm_scale_list=lm_scale_list,
        )
    elif params.decoding_method == "modified_beam_search_lm_rescore_LODR":
        lm_scale_list = [0.02 * i for i in range(2, 30)]
        ans_dict = modified_beam_search_lm_rescore_LODR(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam_size,
            LM=LM,
            LODR_lm=ngram_lm,
            sp=sp,
            lm_scale_list=lm_scale_list,
        )
    else:
        batch_size = encoder_out.size(0)

        for i in range(batch_size):
            # fmt: off
            encoder_out_i = encoder_out[i:i+1, :encoder_out_lens[i]]
            # fmt: on
            if params.decoding_method == "greedy_search":
                hyp = greedy_search(
                    model=model,
                    encoder_out=encoder_out_i,
                    max_sym_per_frame=params.max_sym_per_frame,
                    allowed_ids=allowed_ids,
                )
            elif params.decoding_method == "beam_search":
                hyp = beam_search(
                    model=model,
                    encoder_out=encoder_out_i,
                    beam=params.beam_size,
                    allowed_ids=allowed_ids,
                )
            else:
                raise ValueError(
                    f"Unsupported decoding method: {params.decoding_method}"
                )
            hyps.append(_decode_tokens(params, sp, hyp))

    if params.decoding_method == "greedy_search":
        return {"greedy_search": hyps}
    elif "fast_beam_search" in params.decoding_method:
        key = f"beam_{params.beam}_"
        key += f"max_contexts_{params.max_contexts}_"
        key += f"max_states_{params.max_states}"
        if "nbest" in params.decoding_method:
            key += f"_num_paths_{params.num_paths}_"
            key += f"nbest_scale_{params.nbest_scale}"
            if "LG" in params.decoding_method:
                key += f"_ngram_lm_scale_{params.ngram_lm_scale}"

        return {key: hyps}
    elif "modified_beam_search" in params.decoding_method:
        prefix = f"beam_size_{params.beam_size}"
        if params.decoding_method in (
            "modified_beam_search_lm_rescore",
            "modified_beam_search_lm_rescore_LODR",
        ):
            ans = dict()
            assert ans_dict is not None
            for key, hyps in ans_dict.items():
                hyps = [_normalize_for_wer(params, sp.decode(hyp).split()) for hyp in hyps]
                ans[f"{prefix}_{key}"] = hyps
            return ans
        else:
            if params.has_contexts:
                prefix += f"-context-score-{params.context_score}"
            return {prefix: hyps}
    else:
        return {f"beam_size_{params.beam_size}": hyps}


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    word_table: Optional[k2.SymbolTable] = None,
    decoding_graph: Optional[k2.Fsa] = None,
    context_graph: Optional[ContextGraph] = None,
    LM: Optional[LmScorer] = None,
    ngram_lm=None,
    ngram_lm_scale: float = 0.0,
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      sp:
        The SentencePiece tokenizer.
      word_table:
        The word symbol table.
      decoding_graph:
        The decoding graph. Can be either a `k2.trivial_graph` or HLG, Used
        only when --decoding-method is fast_beam_search, fast_beam_search_nbest,
        fast_beam_search_nbest_oracle, and fast_beam_search_nbest_LG.
    Returns:
      Return a dict, whose key may be "greedy_search" if greedy search
      is used, or it may be "beam_7" if beam size of 7 is used.
      Its value is a list of tuples. Each tuple contains two elements:
      The first is the reference transcript, and the second is the
      predicted result.
    """
    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    if params.decoding_method == "greedy_search":
        log_interval = 50
    else:
        log_interval = 20

    results = defaultdict(list)
    for batch_idx, batch in enumerate(dl):
        texts = batch["supervisions"]["text"]
        # HubertAsrDataset puts cuts in batch["cuts"]; K2SpeechRecognitionDataset
        # puts them in batch["supervisions"]["cut"].  Handle both.
        if "cut" in batch["supervisions"]:
            cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]
        elif "cuts" in batch:
            cut_ids = [cut.id for cut in batch["cuts"]]
        else:
            cut_ids = [f"utt_{i}" for i in range(len(texts))]

        hyps_dict = decode_one_batch(
            params=params,
            model=model,
            sp=sp,
            decoding_graph=decoding_graph,
            context_graph=context_graph,
            word_table=word_table,
            batch=batch,
            LM=LM,
            ngram_lm=ngram_lm,
            ngram_lm_scale=ngram_lm_scale,
        )

        for name, hyps in hyps_dict.items():
            this_batch = []
            assert len(hyps) == len(texts)
            for cut_id, hyp_words, ref_text in zip(cut_ids, hyps, texts):
                ref_words = _normalize_for_wer(params, ref_text)
                this_batch.append((cut_id, ref_words, hyp_words))

            results[name].extend(this_batch)

        num_cuts += len(texts)

        if batch_idx % log_interval == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
    return results


def resolve_test_sets(args, datamodule):
    """Resolve evaluation cuts, giving an explicit --cuts-path precedence.

    A named split (``test``/``dev``) or a manifest-dir name is only used when
    no explicit file was supplied. Missing explicit files are errors; they are
    never silently replaced by another split.
    """
    cuts_name = getattr(args, "cuts_name", None)
    cuts_path = getattr(args, "cuts_path", None)

    if cuts_path:
        path = Path(cuts_path)
        if not path.is_file():
            raise FileNotFoundError(f"Explicit evaluation cuts not found: {path}")
        key = cuts_name or path.stem
        return {key: lhotse.load_manifest_lazy(path)}

    if cuts_name in ("dev", "test"):
        source = datamodule.dev_cuts() if cuts_name == "dev" else datamodule.test_cuts()
        if isinstance(source, dict):
            return dict(source)
        return {cuts_name: source}

    if cuts_name:
        if "/" in cuts_name or cuts_name.endswith(".jsonl.gz"):
            path = Path(cuts_name)
        else:
            path = Path(args.manifest_dir) / f"{cuts_name}.jsonl.gz"
        if not path.is_file():
            raise FileNotFoundError(
                f"Evaluation cuts not found for --cuts-name {cuts_name!r}: {path}"
            )
        return {cuts_name: lhotse.load_manifest_lazy(path)}

    return datamodule.test_cuts()


def _wer_results(
    params: AttributeDict,
    results: List[Tuple[str, List[str], List[str]]],
):
    """Normalize raw words for internal WER while keeping the raw export raw."""
    if not (getattr(params, "keep_raw_text", False)
            or getattr(params, "raw_transcripts", False)):
        return results
    wer_params = copy.copy(params)
    wer_params.keep_raw_text = False
    wer_params.raw_transcripts = False
    return [
        (cut_id,
         _normalize_for_wer(wer_params, reference),
         _normalize_for_wer(wer_params, hypothesis))
        for cut_id, reference, hypothesis in results
    ]


def save_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str]]]],
):
    test_set_wers = dict()
    for key, results in results_dict.items():
        recog_path = (
            params.res_dir / f"recogs-{test_set_name}-{key}-{params.suffix}.txt"
        )
        raw_results = sorted(results)
        # Internal recogs/errs always use the current asr/textnorm rules;
        # the raw text is additionally exported for inspection/benchmarking.
        results = _wer_results(params, raw_results)
        export_raw_hypotheses(params, test_set_name, key, raw_results, results)
        store_transcripts(filename=recog_path, texts=results)
        logging.info(f"The transcripts are stored in {recog_path}")

        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = (
            params.res_dir / f"errs-{test_set_name}-{key}-{params.suffix}.txt"
        )
        with open(errs_filename, "w") as f:
            wer = write_error_stats(
                f, f"{test_set_name}-{key}", results, enable_log=True
            )
            test_set_wers[key] = wer

        logging.info("Wrote detailed error stats to {}".format(errs_filename))

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])
    errs_info = (
        params.res_dir / f"wer-summary-{test_set_name}-{key}-{params.suffix}.txt"
    )
    with open(errs_info, "w") as f:
        print("settings\tWER", file=f)
        for key, val in test_set_wers:
            print("{}\t{}".format(key, val), file=f)

    s = "\nFor {}, WER of different settings are:\n".format(test_set_name)
    note = "\tbest for {}".format(test_set_name)
    for key, val in test_set_wers:
        s += "{}\t{}{}\n".format(key, val, note)
        note = ""
    logging.info(s)


@torch.inference_mode()
def main():
    parser = get_parser()
    parser.add_argument("--device", type=int)
    parser.add_argument(
        "--model-type",
        type=str,
        default="auto",
        choices=["auto", "asr", "finetune"],
        help="Model architecture type. 'auto' detects from checkpoint.",
    )

    ArabicAsrDataModule.add_arguments(parser)
    LmScorer.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    # ── Find a checkpoint (for model-type detection + metadata) ──────────
    ref_ckpt = None
    for pattern in [
        args.exp_dir / f"epoch-{args.epoch}.pt" if args.epoch else None,
        args.exp_dir / "epoch-1.pt",
        args.exp_dir / "best-train-loss.pt",
        args.exp_dir / "best-valid-loss.pt",
    ]:
        if pattern and pattern.is_file():
            ref_ckpt = pattern
            break
    if ref_ckpt is None:
        ckpts = sorted(args.exp_dir.glob("*.pt"))
        ref_ckpt = ckpts[0] if ckpts else None

    # ── Auto-detect model type ────────────────────────────────────────────
    if args.model_type == "auto":
        if ref_ckpt:
            model_type = _detect_model_type(ref_ckpt)
            logging.info(f"Auto-detected model type: {model_type} (from {ref_ckpt})")
        else:
            logging.warning("No checkpoint found in exp_dir, defaulting to 'asr'")
            model_type = "asr"
    else:
        model_type = args.model_type

    # ── Select model / params based on type ───────────────────────────────
    if model_type == "finetune":
        get_model_fn, get_params_fn = get_finetune_model, get_finetune_params
        use_hubert_dataset = True
    else:
        get_model_fn, get_params_fn = get_asr_model, get_asr_params
        use_hubert_dataset = False

    params = get_params_fn()
    params.update(vars(args))

    # ── Override params with checkpoint metadata ──────────────────────────
    # The checkpoint stores all training hyperparameters.  Load them so the
    # model architecture matches exactly what was trained.
    _CLI_KEPT_KEYS = frozenset({
        "bpe_model", "lang_dir", "manifest_dir", "exp_dir", "decode_dir",
        "epoch", "avg", "iter", "start_epoch", "start_batch", "num_epochs",
        "device", "world_size", "master_port", "cuts_name", "cuts_path",
        "decoding_method", "beam_size", "beam", "max_duration",
        # decode-time dialect choice (test set dialect / textnorm / vocab
        # mask target) must come from the CLI, not the training checkpoint.
        "dialect",
    })
    if ref_ckpt and ref_ckpt.is_file():
        _ckpt = torch.load(str(ref_ckpt), map_location="cpu", weights_only=False)
        _ckpt_params = {k: v for k, v in _ckpt.items()
                        if not callable(v) and not k.startswith("_")
                        and k not in ("model", "model_avg", "optimizer", "scheduler",
                                      "grad_scaler", "sampler")}
        for k, v in _ckpt_params.items():
            if k in params and k not in _CLI_KEPT_KEYS:
                params[k] = v
        logging.info(f"Loaded {len(_ckpt_params)} metadata keys from checkpoint")

    # Keep the pre-textnorm hypothesis for the final raw export. Internal WER
    # still applies asr/textnorm via _wer_results().
    params.keep_raw_text = True

    assert params.decoding_method in (
        "greedy_search",
        "beam_search",
        "fast_beam_search",
        "fast_beam_search_nbest",
        "fast_beam_search_nbest_LG",
        "fast_beam_search_nbest_oracle",
        "modified_beam_search",
        "modified_beam_search_LODR",
        "modified_beam_search_lm_shallow_fusion",
        "modified_beam_search_lm_rescore",
        "modified_beam_search_lm_rescore_LODR",
    )
    if params.decode_dir is not None:
        params.decode_dir = Path(params.decode_dir)
        params.res_dir = params.decode_dir / params.decoding_method
    else:
        params.res_dir = params.exp_dir / params.decoding_method

    if os.path.exists(params.context_file):
        params.has_contexts = True
    else:
        params.has_contexts = False

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    if params.causal:
        assert (
            "," not in params.chunk_size
        ), "chunk_size should be one value in decoding."
        assert (
            "," not in params.left_context_frames
        ), "left_context_frames should be one value in decoding."
        params.suffix += f"-chunk-{params.chunk_size}"
        params.suffix += f"-left-context-{params.left_context_frames}"

    if "fast_beam_search" in params.decoding_method:
        params.suffix += f"-beam-{params.beam}"
        params.suffix += f"-max-contexts-{params.max_contexts}"
        params.suffix += f"-max-states-{params.max_states}"
        if "nbest" in params.decoding_method:
            params.suffix += f"-nbest-scale-{params.nbest_scale}"
            params.suffix += f"-num-paths-{params.num_paths}"
            if "LG" in params.decoding_method:
                params.suffix += f"-ngram-lm-scale-{params.ngram_lm_scale}"
    elif "beam_search" in params.decoding_method:
        params.suffix += f"-{params.decoding_method}-beam-size-{params.beam_size}"
        if params.decoding_method in (
            "modified_beam_search",
            "modified_beam_search_LODR",
        ):
            if params.has_contexts:
                params.suffix += f"-context-score-{params.context_score}"
    else:
        params.suffix += f"-context-{params.context_size}"
        params.suffix += f"-max-sym-per-frame-{params.max_sym_per_frame}"

    if params.use_shallow_fusion:
        params.suffix += f"-{params.lm_type}-lm-scale-{params.lm_scale}"

        if "LODR" in params.decoding_method:
            params.suffix += (
                f"-LODR-{params.tokens_ngram}gram-scale-{params.ngram_lm_scale}"
            )

    if params.use_averaged_model:
        params.suffix += "-use-averaged-model"

    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}")
    logging.info("Decoding started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{params.device}")

    logging.info(f"Device: {device}")

    if getattr(params, "multi_bpe", False):
        # Union vocab: load id2piece / dialect_id_map tables + per-dialect
        # tokenizers; sets params.blank_id / sos_id / eos_id / vocab_size
        # (union size, e.g. 1317 — must NOT be clobbered by a single
        # dialect sp of 500 pieces).
        load_multi_bpe(params)
        sp = None
        logging.info(
            f"Multi-BPE decode: union vocab size {params.vocab_size}, "
            f"dialects {MULTI_BPE_DIALECTS}"
        )
    else:
        sp = spm.SentencePieceProcessor()
        sp.load(params.bpe_model)

        # <blk> and <unk> are defined by scripts/build_tokenizer.py
        params.blank_id = sp.piece_to_id("<blk>")
        params.unk_id = sp.piece_to_id("<unk>")
        params.vocab_size = sp.get_piece_size()

    # ── Dialect vocab mask for multi-bpe decode ──────────────────────────
    # Restrict decode logits to {blank} ∪ {target dialect's union IDs}.
    # params.dialect is the CLI decode dialect (kept from checkpoint
    # override). Mixed test sets (e.g. dialect=ARAB) decode unmasked.
    params.decode_allowed_ids = None
    if getattr(params, "multi_bpe", False):
        id_maps = _multi_bpe_ctx["id_maps"]
        decode_dialect = str(getattr(params, "dialect", "")).upper()
        if decode_dialect in id_maps:
            # specials share union IDs 0/1/2; drop <sos/eos> and <unk> so
            # the search can only emit blank + real pieces.
            allowed = set(id_maps[decode_dialect]) - {1, 2}
            params.decode_allowed_ids = torch.tensor(
                sorted(allowed), dtype=torch.long, device=device
            )
            logging.info(
                f"Decode vocab mask: dialect={decode_dialect}, "
                f"{len(allowed)} allowed union IDs of {params.vocab_size}"
            )
        else:
            logging.warning(
                f"No union id_map for decode dialect '{decode_dialect}': "
                f"decoding with the full union vocab (no dialect mask)"
            )

    logging.info(params)

    # Finetune models may have minor architecture mismatches with the
    # current code (e.g. Decoder.conv is absent from older checkpoints).
    strict_load = (model_type == "asr")

    logging.info(f"Creating model (type={model_type}, strict_load={strict_load})")
    model = get_model_fn(params)

    if not params.use_averaged_model:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device), strict=strict_load)
        elif params.avg == 1:
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if i >= 1:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device), strict=strict_load)
    else:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg + 1
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg + 1:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            filename_start = filenames[-1]
            filename_end = filenames[0]
            logging.info(
                "Calculating the averaged model over iteration checkpoints"
                f" from {filename_start} (excluded) to {filename_end}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                ),
                strict=strict_load,
            )
        else:
            assert params.avg > 0, params.avg
            start = params.epoch - params.avg
            assert start >= 1, start
            filename_start = f"{params.exp_dir}/epoch-{start}.pt"
            filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
            logging.info(
                f"Calculating the averaged model over epoch range from "
                f"{start} (excluded) to {params.epoch}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                ),
                strict=strict_load,
            )

    model.to(device)
    model.eval()

    # only load the neural network LM if required
    if params.use_shallow_fusion or params.decoding_method in (
        "modified_beam_search_lm_rescore",
        "modified_beam_search_lm_rescore_LODR",
        "modified_beam_search_lm_shallow_fusion",
        "modified_beam_search_LODR",
    ):
        LM = LmScorer(
            lm_type=params.lm_type,
            params=params,
            device=device,
            lm_scale=params.lm_scale,
        )
        LM.to(device)
        LM.eval()
    else:
        LM = None

    # only load N-gram LM when needed
    if params.decoding_method == "modified_beam_search_lm_rescore_LODR":
        try:
            import kenlm
        except ImportError:
            print("Please install kenlm first. You can use")
            print(" pip install https://github.com/kpu/kenlm/archive/master.zip")
            print("to install it")
            import sys

            sys.exit(-1)
        ngram_file_name = str(params.lang_dir / f"{params.tokens_ngram}gram.arpa")
        logging.info(f"lm filename: {ngram_file_name}")
        ngram_lm = kenlm.Model(ngram_file_name)
        ngram_lm_scale = None  # use a list to search

    elif params.decoding_method == "modified_beam_search_LODR":
        lm_filename = f"{params.tokens_ngram}gram.fst.txt"
        logging.info(f"Loading token level lm: {lm_filename}")
        ngram_lm = NgramLm(
            str(params.lang_dir / lm_filename),
            backoff_id=params.backoff_id,
            is_binary=False,
        )
        logging.info(f"num states: {ngram_lm.lm.num_states}")
        ngram_lm_scale = params.ngram_lm_scale
    else:
        ngram_lm = None
        ngram_lm_scale = None

    if "fast_beam_search" in params.decoding_method:
        if params.decoding_method == "fast_beam_search_nbest_LG":
            lexicon = Lexicon(params.lang_dir)
            word_table = lexicon.word_table
            lg_filename = params.lang_dir / "LG.pt"
            logging.info(f"Loading {lg_filename}")
            decoding_graph = k2.Fsa.from_dict(
                torch.load(lg_filename, map_location=device)
            )
            decoding_graph.scores *= params.ngram_lm_scale
        else:
            word_table = None
            decoding_graph = k2.trivial_graph(params.vocab_size - 1, device=device)
    else:
        decoding_graph = None
        word_table = None

    if "modified_beam_search" in params.decoding_method:
        if os.path.exists(params.context_file):
            contexts = []
            for line in open(params.context_file).readlines():
                if getattr(params, "multi_bpe", False):
                    # context pieces must be union IDs of the decode dialect
                    d = str(params.dialect).upper()
                    if d not in _multi_bpe_ctx["id_maps"]:
                        d = "MSA"
                    ids = _encode_text_multi_bpe(line.strip(), d)
                else:
                    ids = sp.encode(line.strip())
                contexts.append((ids, 0.0))
            context_graph = ContextGraph(params.context_score)
            context_graph.build(contexts)
        else:
            context_graph = None
    else:
        context_graph = None

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True
    datamodule = ArabicAsrDataModule(args)

    def remove_short_and_long_utt(c):
        if c.duration < 1.0 or c.duration > 40.0:
            return False
        if c.supervisions[0].text.strip() == "":
            return False
        T = ((c.num_frames - 7) // 2 + 1) // 2
        if getattr(params, "multi_bpe", False):
            tokens = _encode_text_multi_bpe(
                c.supervisions[0].text, _dialect_of_cut(c)
            )
        else:
            tokens = sp.encode(c.supervisions[0].text, out_type=str)
        if T < len(tokens):
            return False
        return True

    # ── Test set selection ────────────────────────────────────────────────
    test_sets_cuts: Dict[str, CutSet] = resolve_test_sets(args, datamodule)

    for k, v in list(test_sets_cuts.items()):
        if not getattr(params, "keep_eval_cuts", False):
            test_sets_cuts[k] = v.filter(remove_short_and_long_utt)
            if getattr(params, "dialect", "SAU") != "ENG":
                test_sets_cuts[k] = test_sets_cuts[k].filter(is_arabic_only)
        test_sets_cuts[k].describe()

    test_sets = list(test_sets_cuts.keys())

    if use_hubert_dataset:
        # SSL finetune models expect HubertAsrDataset (provides batch["audio"] = fbank)
        from torch.utils.data import DataLoader
        from lhotse.dataset import DynamicBucketingSampler

        test_dl = []
        for name in test_sets:
            dataset = HubertAsrDataset()
            sampler = DynamicBucketingSampler(
                test_sets_cuts[name],
                max_duration=args.max_duration,
                shuffle=False,
            )
            test_dl.append(DataLoader(
                dataset, batch_size=None, sampler=sampler, num_workers=args.num_workers,
                pin_memory=True, persistent_workers=False,
            ))
        logging.info(f"Using HubertAsrDataset for finetune model ({len(test_sets)} test sets)")
    else:
        test_dl = [
            datamodule.test_dataloaders(test_sets_cuts[name])
            for name in test_sets
        ]

    for test_set, test_dl in zip(test_sets, test_dl):
        results_dict = decode_dataset(
            dl=test_dl,
            params=params,
            model=model,
            sp=sp,
            word_table=word_table,
            decoding_graph=decoding_graph,
            context_graph=context_graph,
            LM=LM,
            ngram_lm=ngram_lm,
            ngram_lm_scale=ngram_lm_scale,
        )

        save_results(
            params=params,
            test_set_name=test_set,
            results_dict=results_dict,
        )

    logging.info("Done!")


if __name__ == "__main__":
    main()

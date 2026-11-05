#!/usr/bin/env python3
# The SentencePiece training portion is adapted from icefall's
# train_bpe_model.py, Copyright 2021 Xiaomi Corp., Apache-2.0.
"""Build a SentencePiece unigram tokenizer from the training cuts.

The script performs both tokenizer preparation steps in one place:

1. write normalized training text to ``lang/train.txt``;
2. train ``lang/tokenizer.model`` and export ``lang/tokens.txt``.

Only the training split is used. Evaluation references are not part of this
stage and are supplied separately when decoding.
"""

import argparse
import importlib.util
import json
from pathlib import Path

from common import read_jsonl, sha256

USER_DEFINED_SYMBOLS = ["<blk>", "<sos/eos>"]
MODEL_TYPE = "unigram"
META_NAME = "tokenizer.meta.json"


def write_training_text(cuts, output):
    """Normalize the training transcripts and write one sentence per line."""
    source = Path(__file__).resolve().parents[1] / "asr/textnorm/_common.py"
    spec = importlib.util.spec_from_file_location("training_char_normalizer", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with Path(output).open("w", encoding="utf-8") as stream:
        for cut in read_jsonl(cuts):
            text = module.normalize_char_level(cut["supervisions"][0]["text"])
            if text.strip():
                stream.write(" ".join(text.split()) + "\n")
                count += 1
    if not count:
        raise ValueError("No training text remains after character normalization")
    print(f"Tokenizer training sentences: {count}")
    return count


def generate_tokens(model_file, tokens_file):
    """Write the token-id table used by icefall training and decoding."""
    import sentencepiece as spm

    processor = spm.SentencePieceProcessor(model_file=str(model_file))
    with Path(tokens_file).open("w", encoding="utf-8") as stream:
        for index in range(processor.get_piece_size()):
            stream.write(f"{processor.id_to_piece(index)} {index}\n")


def _model_type(processor):
    """Read the actual SentencePiece model type from its serialized proto."""
    from sentencepiece import sentencepiece_model_pb2 as model_pb

    proto = model_pb.ModelProto()
    proto.ParseFromString(processor.serialized_model_proto())
    return proto.trainer_spec.model_type


def _validate_existing(model_file, meta_file, text_file, vocab_size):
    """Return the model size only when the existing model is the exact match."""
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2 as model_pb

    if not meta_file.is_file():
        raise ValueError(
            f"{model_file} exists without {meta_file.name}; cannot verify its "
            "model type, vocabulary or training text. Use a new output directory "
            "or pass --overwrite."
        )
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Cannot read {meta_file}; use a new output directory or --overwrite"
        ) from error

    if meta.get("model_type") != MODEL_TYPE:
        raise ValueError(
            f"Existing tokenizer model type is {meta.get('model_type')!r}, "
            f"expected {MODEL_TYPE!r}; use --overwrite."
        )
    if meta.get("requested_vocab_size") != vocab_size:
        raise ValueError(
            f"Existing tokenizer was built with vocab_size="
            f"{meta.get('requested_vocab_size')}, requested {vocab_size}; "
            "use --overwrite."
        )
    processor = spm.SentencePieceProcessor(model_file=str(model_file))
    actual_type = _model_type(processor)
    expected_type = model_pb.ModelProto().trainer_spec.UNIGRAM
    if actual_type != expected_type:
        raise ValueError(
            f"Existing tokenizer model type is not unigram; use --overwrite."
        )
    actual_size = processor.get_piece_size()
    if meta.get("actual_vocab_size") != actual_size:
        raise ValueError(
            f"Existing tokenizer model size ({actual_size}) does not match "
            f"its metadata ({meta.get('actual_vocab_size')}); use --overwrite."
        )
    if meta.get("model_sha256") != sha256(model_file):
        raise ValueError(
            "Existing tokenizer model file does not match its metadata; "
            "use --overwrite."
        )
    input_hash = sha256(text_file)
    if meta.get("input_sha256") != input_hash:
        raise ValueError(
            "Existing tokenizer was trained on different input text; "
            "use a new output directory or --overwrite."
        )
    return actual_size


def build_tokenizer(text_file, lang_dir, vocab_size, overwrite=False):
    """Train a unigram model when needed and always refresh tokens.txt.

    Reuse is allowed only when the model metadata confirms the same model
    type, requested vocabulary and input-text hash. ``overwrite=True``
    explicitly discards an incompatible model and rebuilds it.
    """
    import sentencepiece as spm

    lang_dir = Path(lang_dir)
    lang_dir.mkdir(parents=True, exist_ok=True)
    model_file = lang_dir / "tokenizer.model"
    meta_file = lang_dir / META_NAME
    if vocab_size < 1:
        raise ValueError("vocab_size must be positive")

    if model_file.is_file() and not overwrite:
        actual_size = _validate_existing(model_file, meta_file, text_file, vocab_size)
        print(f"{model_file} exists with matching metadata (vocab={actual_size}) - reusing it")
        generate_tokens(model_file, lang_dir / "tokens.txt")
        print(f"Tokenizer model: {model_file}")
        return

    if overwrite:
        for path in (model_file, lang_dir / "tokenizer.vocab", meta_file,
                     lang_dir / "tokens.txt"):
            path.unlink(missing_ok=True)

    # ``tokenizer.vocab`` is emitted by SentencePiece alongside the model.
    try:
        spm.SentencePieceTrainer.train(
            input=str(text_file),
            vocab_size=vocab_size,
            model_type=MODEL_TYPE,
            model_prefix=str(lang_dir / "tokenizer"),
            input_sentence_size=1_000_000,
            character_coverage=1.0,
            user_defined_symbols=USER_DEFINED_SYMBOLS,
            unk_id=len(USER_DEFINED_SYMBOLS),
            bos_id=-1,
            eos_id=-1,
        )
    except Exception:
        # SentencePiece can leave a partial file behind when the requested
        # vocabulary is impossible for a tiny corpus.
        model_file.unlink(missing_ok=True)
        (lang_dir / "tokenizer.vocab").unlink(missing_ok=True)
        raise

    processor = spm.SentencePieceProcessor(model_file=str(model_file))
    actual_size = processor.get_piece_size()
    meta_file.write_text(json.dumps({
        "model_type": MODEL_TYPE,
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": actual_size,
        "model_sha256": sha256(model_file),
        "input_sha256": sha256(text_file),
    }, indent=2) + "\n", encoding="utf-8")
    generate_tokens(model_file, lang_dir / "tokens.txt")
    print(f"Tokenizer model: {model_file} (vocab={actual_size})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuts", type=Path, required=True,
                        help="Training raw-cut manifest (JSONL or JSONL.gz)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for train.txt, tokenizer.model and tokens.txt")
    parser.add_argument("--vocab-size", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true",
                        help="Discard an incompatible tokenizer and rebuild it")
    args = parser.parse_args()

    output_dir = args.output_dir
    text_file = output_dir / "train.txt"
    temporary_text = output_dir / "train.txt.tmp"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        write_training_text(args.cuts, temporary_text)
        build_tokenizer(
            temporary_text, output_dir, args.vocab_size, overwrite=args.overwrite
        )
        temporary_text.replace(text_file)
    finally:
        temporary_text.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

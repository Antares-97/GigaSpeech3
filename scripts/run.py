"""Prepare training data and a SentencePiece unigram tokenizer.

This dispatcher stops at the tokenizer. Training and decoding are separate
entrypoints (scripts/train.py and scripts/decode.py) because they take
different arguments and run on different machines.

Only the training split is processed. Outputs are created under
data/<DIALECT>/{audio,raw_cuts,fbank,lang}. DZA is used consistently for
Algerian Arabic.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from common import DIALECTS

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
ASR = ROOT / "asr"
STAGES = ("prepare", "features", "tokenizer")


def _substitute(value, code):
    if isinstance(value, str):
        return value.replace("{dialect}", code)
    if isinstance(value, list):
        return [_substitute(v, code) for v in value]
    if isinstance(value, dict):
        return {k: _substitute(v, code) for k, v in value.items()}
    return value


def load_config(path, overrides):
    """Resolve the data/tokenizer recipe for one dialect."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    config.update({key: value for key, value in overrides.items() if value is not None})
    code = str(config["dialect"]).upper()
    if code not in DIALECTS:
        raise ValueError(f"Expected one of {DIALECTS}, got {code}")
    config = {key: _substitute(value, code) for key, value in config.items()}
    config["dialect"] = code
    config.setdefault("data_root", "data")
    config.setdefault("max_shards", 0)
    config.setdefault("num_workers", 2)
    config.setdefault("tokenizer_vocab_size", 500)
    if int(config["max_shards"]) < 0:
        raise ValueError("max_shards must be non-negative; 0 means all shards")
    if int(config["num_workers"]) < 1:
        raise ValueError("num_workers must be positive")
    if int(config["tokenizer_vocab_size"]) < 1:
        raise ValueError("tokenizer_vocab_size must be positive")
    return config


def absolute(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/dialect-specific.example.json")
    parser.add_argument("--dialect", type=str.upper, choices=DIALECTS)
    parser.add_argument("--repo-id")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--local-dataset", type=Path)
    parser.add_argument("--data-root")
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--tokenizer-vocab-size", type=int)
    parser.add_argument("--stage", choices=STAGES, default="prepare",
                        help="Run one stage at a time: prepare, features, tokenizer")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, {
        "dialect": args.dialect,
        "repo_id": args.repo_id,
        "dataset_revision": args.dataset_revision,
        "local_dataset": args.local_dataset,
        "data_root": args.data_root,
        "max_shards": args.max_shards,
        "num_workers": args.num_workers,
        "tokenizer_vocab_size": args.tokenizer_vocab_size,
    })
    code = config["dialect"]

    data_root = absolute(config["data_root"])
    data_dir = data_root / code
    raw = data_dir / "raw_cuts"
    fbank = data_dir / "fbank"
    lang = data_dir / "lang"
    audio = data_dir / "audio"
    prefix = f"gigaspeech3_{code.lower()}_cuts"

    py = config.get("python", sys.executable)
    env = os.environ.copy()
    path_entries = [ASR, ASR / "zipformer", ROOT, SCRIPTS]
    env["PYTHONPATH"] = os.pathsep.join(map(str, path_entries)) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PIPELINE_ROOT"] = str(ROOT)
    jobs = {}

    prepare = [py, SCRIPTS / "prepare.py", "hf", "--repo-id", config["repo_id"],
               "--revision", config.get("dataset_revision", "main"), "--dialects", code,
               "--audio-dir", audio, "--output", raw / f"{prefix}_train_raw.jsonl.gz"]
    if config.get("local_dataset"):
        prepare += ["--local-dataset", absolute(config["local_dataset"])]
    if config["max_shards"]:
        prepare += ["--max-shards", str(config["max_shards"])]
    jobs["prepare"] = prepare

    jobs["features"] = [
        py, SCRIPTS / "features.py", "--dialect", code, "--partition", "train",
        "--data-dir", fbank, "--raw-cuts-dir", raw,
        "--num-workers", config["num_workers"],
    ]
    jobs["tokenizer"] = [
        py, SCRIPTS / "build_tokenizer.py",
        "--cuts", raw / f"{prefix}_train_raw.jsonl.gz",
        "--output-dir", lang,
        "--vocab-size", config["tokenizer_vocab_size"],
    ]

    command = jobs[args.stage]
    print(f"[{args.stage}] {shlex.join([str(item) for item in command])}", flush=True)
    if args.dry_run:
        return

    for directory in (data_root, data_dir, raw, fbank, lang, audio):
        directory.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(item) for item in command], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()

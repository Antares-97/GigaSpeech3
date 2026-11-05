# Install the dialect-specific recipe

Run on Linux or WSL2 with an NVIDIA GPU for training. Native Windows is not
supported by the training core (`fcntl`). Python 3.10 or 3.11 is a sensible
starting point; the exact wheel versions must match your CUDA stack.

## System packages

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs ffmpeg libsndfile1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## Python dependencies

1. Install **torch and torchaudio together**, selecting a matched CUDA build
   from the PyTorch installation instructions.
2. Install a **CUDA-enabled k2 wheel** compatible with that Python, torch and
   CUDA combination.
3. Install **kaldifeat** for the same torch environment.
4. Install the data/tokenizer dependencies:

```bash
python -m pip install -r requirements.txt
```

The training/decoding code and the upstream `icefall` Python package are
vendored under `asr/`; no separate training-code or `icefall` checkout is
required. Do not install the legacy upstream `requirements.txt` from an older
checkout — it may contain a Git-package icefall entry and dependencies that do
not describe this setup.

## Environment check

```bash
python -m pip check
python - <<'PY'
import sys
sys.path.insert(0, "asr")
import torch, torchaudio, k2, kaldifeat, lhotse, soundfile, pyarrow, sentencepiece
import icefall, kaldialign, packaging
print("torch", torch.__version__, "CUDA", torch.version.cuda)
print("GPU available:", torch.cuda.is_available())
PY
python scripts/run.py --dialect DZA --stage prepare --dry-run
python scripts/train.py --help
python scripts/decode.py --help
```

## Configure and run the data stages

```bash
cp configs/dialect-specific.example.json configs/DZA.json
# edit configs/DZA.json: max_shards (0 = full data), data_root,
# tokenizer_vocab_size
```

Run one stage at a time. Only the training split is processed:

```bash
python scripts/run.py --config configs/DZA.json --dialect DZA --stage prepare
python scripts/run.py --config configs/DZA.json --dialect DZA --stage features
python scripts/run.py --config configs/DZA.json --dialect DZA --stage tokenizer
```

The code creates `data/<DIALECT>/` as needed. There is no automatic `all`
stage; training and decoding are separate. Tokenizer reuse is rejected when
the model type, vocabulary size or training-text hash changes; use a new
output directory or `python scripts/build_tokenizer.py ... --overwrite`.

## Train and decode separately

Training requires an independent dev CutSet at
`data/<DIALECT>/fbank/gigaspeech3_<dialect>_cuts_dev.jsonl.gz` (80-bin
Kaldifeat fbank, 16 kHz mono, human transcripts). It is not created by the data
stages; see [Development data](#development-data) for the reference JSONL
schema and conversion commands. Training fails if it is missing and never
falls back to train cuts.

`scripts/train.py` and `scripts/decode.py` pass every argument through to the
ASR entrypoints. Use `--help` to see the complete argument lists.

```bash
python scripts/train.py \
  --dialect DZA \
  --manifest-dir data/DZA/fbank \
  --exp-dir exp/DZA/run1 \
  --tokenizer-model data/DZA/lang/tokenizer.model \
  --cuts-name gigaspeech3_dza_cuts_train \
  --world-size 8 --num-epochs 30 --max-duration 600 --use-fp16 1

python scripts/decode.py \
  --dialect DZA \
  --manifest-dir data/DZA/fbank \
  --exp-dir exp/DZA/run1 \
  --tokenizer-model data/DZA/lang/tokenizer.model \
  --cuts-name eval \
  --cuts-path /path/to/eval_cuts.jsonl.gz \
  --epoch 30 --avg 15 --decoding-method greedy_search \
  --keep-eval-cuts
```

Every decode writes `hyp-*.jsonl` with `raw_text` and `text`.
`--raw-transcripts` is retained only for compatibility; raw export is always
on. `--keep-eval-cuts` preserves every supplied evaluation segment ID.

Training output is created under the `--exp-dir` you pass (for example
`exp/<DIALECT>/<run_name>/`); decoding output goes to its `decode/` subdirectory. Evaluation cuts are supplied separately and
are not created by the data stages.

Do not wrap the training command in `torchrun`: `train.py` already spawns
workers with `--world-size`.

## Tests

```bash
python -m unittest discover -s tests -v
```

These tests cover the data and tokenizer workflow and transcript export contracts.

## Development data

Create an independent human-reference JSONL with one record per segment:

```json
{"id":"DZA-dev-1","audio_path":"audio/recording.wav","start":2.0,"duration":3.0,"text":"النص المرجعي","dialect":"DZA","label_type":"human"}
```

Audio paths are absolute or relative to the JSONL file. Preserve original IDs,
transcripts and segment boundaries. Use separate development and test sets.

```bash
python scripts/prepare.py references --dialects DZA \
  --input data/references/DZA-dev.jsonl \
  --output data/DZA/raw_cuts/gigaspeech3_dza_cuts_dev_raw.jsonl.gz
python scripts/features.py --dialect DZA --partition dev \
  --raw-cuts-dir data/DZA/raw_cuts --data-dir data/DZA/fbank
```

Repeat for test if needed, keeping dev independent from test and training.

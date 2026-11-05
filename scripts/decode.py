#!/usr/bin/env python3
"""Run decoding with the full ``asr/decode.py`` argument set.

All arguments are passed through unchanged, for example::

    python scripts/decode.py --dialect DZA --manifest-dir data/DZA/fbank \
      --exp-dir exp/DZA/run1 --tokenizer-model data/DZA/lang/tokenizer.model \
      --cuts-path /path/to/eval_cuts.jsonl.gz --epoch 30 --avg 15 \
      --keep-eval-cuts

Evaluation cuts are supplied separately; the training-data stages do not
create dev/test manifests. WER normalization uses the current pipeline
``asr/textnorm/<DIALECT>.py``. Every decode exports ``raw_text`` and
current-textnorm ``text`` in ``hyp-*.jsonl``; ``--keep-eval-cuts``
preserves all explicitly supplied segment IDs.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASR = ROOT / "asr"


def main():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ASR), str(ASR / "zipformer"), str(ROOT), str(ROOT / "scripts")]
    ) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    command = [sys.executable, str(ASR / "decode.py"), *sys.argv[1:]]
    return subprocess.call(command, cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())

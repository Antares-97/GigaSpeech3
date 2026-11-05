# Third-party notices

This repository vendors code needed to run the pipeline:

- `asr/icefall/` — the upstream icefall Python package, licensed under Apache-2.0.
  Source: https://github.com/k2-fsa/icefall
- `asr/` training/decoding scripts — adapted from icefall example recipes and the
  GigaSpeech 3 research code, licensed under Apache-2.0.
- `scripts/build_tokenizer.py` — SentencePiece training portion adapted from
  icefall's `train_bpe_model.py`, Copyright 2021 Xiaomi Corp., Apache-2.0.
- `asr/textnorm/{DZA,EGY,IRQ,SAU,SYR}.py` — GigaSpeech 3 dialect text
  normalization scripts, licensed under Apache-2.0.

The root `LICENSE` (Apache-2.0) applies to this repository's original code.
Upstream copyright notices are retained in the vendored files and the
`asr/LICENSE` file.

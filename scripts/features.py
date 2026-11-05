# Copyright 2021 Johns Hopkins University (Piotr Zelasko)
# Copyright 2021 Xiaomi Corp. (Fangjun Kuang)
# SPDX-License-Identifier: Apache-2.0
"""Extract Fbank features from the prepared dialect-specific Lhotse cuts."""

import argparse
import logging
from pathlib import Path

from common import DIALECTS


def compute_features(dialect, partition, raw_cuts_dir, data_dir, num_workers=2):
    code = dialect.upper()
    prefix = f"gigaspeech3_{code.lower()}_cuts_{partition}"
    source = Path(raw_cuts_dir) / f"{prefix}_raw.jsonl.gz"
    target = Path(data_dir) / f"{prefix}.jsonl.gz"
    if not source.is_file():
        raise FileNotFoundError(f"Missing raw cuts: {source}; run the prepare stage first")
    if target.is_file():
        logging.info("Using existing features: %s", target)
        return
    if num_workers < 1:
        raise ValueError("num_workers must be positive")

    import torch
    from lhotse import CutSet, KaldifeatFbank, KaldifeatFbankConfig

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    target.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    extractor = KaldifeatFbank(KaldifeatFbankConfig(device=device))
    cuts = CutSet.from_file(source).compute_and_store_features_batch(
        extractor=extractor,
        storage_path=str(target.parent / f"gigaspeech3_{code.lower()}_feats_{partition}"),
        num_workers=num_workers,
        batch_duration=1000,
        overwrite=True,
    )
    # Prepared cuts already contain exactly one supervision covering the cut
    # and the cut audio is the segment itself; no trim is needed.
    cuts.to_file(target)
    logging.info("Saved %s", target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dialect", choices=DIALECTS, required=True)
    parser.add_argument("--partition", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--raw-cuts-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    compute_features(**vars(args))


if __name__ == "__main__":
    main()

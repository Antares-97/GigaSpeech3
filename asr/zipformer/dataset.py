# Copyright      2021-2024  Xiaomi Corp.        (authors: Yifan Yang, et al.)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
#
# MERGED from:
#   - vietasr-saudi/ASR/zipformer/dataset.py     (PseudoRecognitionDataset, validate_for_asr)
#   - vietasr-saudi/SSL/zipformer_fbank/dataset.py (HubertDataset, HubertAsrDataset)
#
# Three dataset classes:
#   PseudoRecognitionDataset  — Stage 1 supervised ASR training
#   HubertDataset             — Stage 2 HuBERT SSL pretraining
#   HubertAsrDataset          — Stage 2 HuBERT finetuning on ASR data

import sys
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from lhotse import validate
from lhotse.cut import CutSet
from lhotse.dataset.input_strategies import BatchIO, PrecomputedFeatures
from lhotse.utils import compute_num_frames, ifnone
from lhotse.workarounds import Hdf5MemoryIssueFix
from torch.utils.data.dataloader import default_collate


def disc_to_label(text):
    """Deduplicate consecutive identical tokens in a space-delimited string."""
    pre = ""
    label = []
    for item in text.split():
        if item != pre:
            label.append(item)
            pre = item
    return " ".join(label)


def validate_for_asr(cuts: CutSet) -> None:
    validate(cuts)
    tol = 2e-3  # 1ms
    for cut in cuts:
        for supervision in cut.supervisions:
            assert supervision.start >= -tol, (
                f"Supervisions starting before the cut are not supported for ASR"
                f" (sup id: {supervision.id}, cut id: {cut.id})"
            )
            assert supervision.end <= cut.duration + tol, (
                f"Supervisions ending after the cut "
                f"are not supported for ASR"
                f" (sup id: {supervision.id}, cut id: {cut.id})"
            )


class PseudoRecognitionDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for k2 speech recognition.

    This dataset expects to be queried with lists of cut IDs,
    for which it loads features and automatically collates/batches them.

    To use it with a PyTorch DataLoader, set ``batch_size=None``
    and provide a :class:`SimpleCutSampler` sampler.

    Each item in this dataset is a dict of:

    .. code-block::

        {
            'inputs': float tensor with shape determined by :attr:`input_strategy`:
                      - features: (B, T, F)
                      - audio: (B, T)
            'supervisions': [
                {
                    'sequence_idx': Tensor[int] of shape (S,)
                    'text': List[str] of len S
                    'start_frame': Tensor[int] of shape (S,)
                    'num_frames': Tensor[int] of shape (S,)
                    'start_sample': Tensor[int] of shape (S,)
                    'num_samples': Tensor[int] of shape (S,)
                    # Optionally, when return_cuts=True
                    'cut': List[AnyCut] of len S
                }
            ]
        }

    Dimension symbols legend:
    * ``B`` - batch size (number of Cuts)
    * ``S`` - number of supervision segments
    * ``T`` - number of frames of the longest Cut
    * ``F`` - number of features
    """

    def __init__(
        self,
        return_cuts: bool = False,
        cut_transforms: List[Callable[[CutSet], CutSet]] = None,
        input_transforms: List[Callable[[torch.Tensor], torch.Tensor]] = None,
        input_strategy: BatchIO = PrecomputedFeatures(),
    ):
        super().__init__()
        self.return_cuts = return_cuts
        self.cut_transforms = ifnone(cut_transforms, [])
        self.input_transforms = ifnone(input_transforms, [])
        self.input_strategy = input_strategy
        self.hdf5_fix = Hdf5MemoryIssueFix(reset_interval=100)

    def __getitem__(self, cuts: CutSet) -> Dict[str, Union[torch.Tensor, List[str]]]:
        validate_for_asr(cuts)
        self.hdf5_fix.update()

        cuts = cuts.sort_by_duration(ascending=False)

        for tnfm in self.cut_transforms:
            cuts = tnfm(cuts)

        cuts = cuts.sort_by_duration(ascending=False)

        input_tpl = self.input_strategy(cuts)
        if len(input_tpl) == 3:
            inputs, _, cuts = input_tpl
        else:
            inputs, _ = input_tpl

        supervision_intervals = self.input_strategy.supervision_intervals(cuts)

        segments = torch.stack(list(supervision_intervals.values()), dim=1)
        for tnfm in self.input_transforms:
            inputs = tnfm(inputs, supervision_segments=segments)

        batch = {
            "inputs": inputs,
            "supervisions": default_collate(
                [
                    {
                        "text": disc_to_label(cut.custom["kmeans"]),
                    }
                    for sequence_idx, cut in enumerate(cuts)
                    for supervision in cut.supervisions
                ]
            ),
        }
        batch["supervisions"].update(supervision_intervals)
        if self.return_cuts:
            batch["supervisions"]["cut"] = [
                cut for cut in cuts for sup in cut.supervisions
            ]

        has_word_alignments = all(
            s.alignment is not None and "word" in s.alignment
            for c in cuts
            for s in c.supervisions
        )
        if has_word_alignments:
            words, starts, ends = [], [], []
            frame_shift = cuts[0].frame_shift
            sampling_rate = cuts[0].sampling_rate
            if frame_shift is None:
                try:
                    frame_shift = self.input_strategy.extractor.frame_shift
                except AttributeError:
                    raise ValueError(
                        "Can't determine the frame_shift -- it is not present "
                        "either in cuts or the input_strategy."
                    )
            for c in cuts:
                for s in c.supervisions:
                    words.append([aliword.symbol for aliword in s.alignment["word"]])
                    starts.append(
                        [
                            compute_num_frames(
                                aliword.start,
                                frame_shift=frame_shift,
                                sampling_rate=sampling_rate,
                            )
                            for aliword in s.alignment["word"]
                        ]
                    )
                    ends.append(
                        [
                            compute_num_frames(
                                aliword.end,
                                frame_shift=frame_shift,
                                sampling_rate=sampling_rate,
                            )
                            for aliword in s.alignment["word"]
                        ]
                    )
            batch["supervisions"]["word"] = words
            batch["supervisions"]["word_start"] = starts
            batch["supervisions"]["word_end"] = ends

        return batch


# ── SSL / HuBERT datasets ─────────────────────────────────────────────────


class HubertDataset(torch.utils.data.Dataset):
    """
    Dataset for HuBERT-style SSL pretraining.

    Returns:
        {
            'cuts': CutSet,
            'features': (B, T, F) float tensor,
            'padding_mask': (B, T) bool tensor,
            'kmeans': (B, T') int64 tensor (k-means pseudo-labels),
        }
    """

    def __init__(
        self,
        max_sample_size: Optional[int] = None,
        sample_rate: float = 100,
        label_rate: float = 50,
        random_crop: bool = True,
        pad_audio: bool = False,
        num_classes: list = [504],
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.label_rate = label_rate
        self.random_crop = random_crop
        self.pad_feature = pad_audio
        self.num_classes = num_classes
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.hdf5_fix = Hdf5MemoryIssueFix(reset_interval=100)

    def __getitem__(self, cuts: CutSet) -> Dict[str, Any]:
        self._validate(cuts)
        self.hdf5_fix.update()

        cuts = cuts.sort_by_duration(ascending=False)

        features = [torch.from_numpy(cut.load_features()) for cut in cuts]
        feature_lens = [cut.num_frames for cut in cuts]

        if self.pad_feature:
            feature_size = min(max(feature_lens), self.max_sample_size)
        else:
            feature_size = min(min(feature_lens), self.max_sample_size)

        features, padding_mask, feature_starts = self.collater_feature(
            features, feature_lens, feature_size
        )

        kmeans = [cut.custom["kmeans"] for cut in cuts]
        if isinstance(kmeans[0], list):
            kmeans = [
                torch.tensor([int(item) for item in label], dtype=torch.int64)
                for label in kmeans
            ]
        else:
            kmeans = [
                torch.tensor([int(item) for item in label.split()], dtype=torch.int64)
                for label in kmeans
            ]
        kmeans, kmeans_lens = self.collater_frm_label(
            kmeans, feature_size, feature_starts
        )

        return {
            "cuts": cuts,
            "features": features,
            "padding_mask": padding_mask,
            "kmeans": kmeans,
        }

    def _validate(self, cuts: CutSet) -> None:
        validate(cuts)
        assert all(cut.has_recording for cut in cuts)

    def crop_to_max_size(self, feature, target_size):
        size = len(feature)
        diff = size - target_size
        if diff <= 0:
            return feature, 0
        start, end = 0, target_size
        if self.random_crop:
            start = np.random.randint(0, diff + 1)
            end = size - diff + start
        return feature[start:end, :], start

    def collater_feature(self, features, feature_lens, feature_size):
        feature_dim = features[0].shape[-1]
        collated_features = features[0].new_zeros(
            len(features), feature_size, feature_dim
        )
        padding_mask = torch.BoolTensor(collated_features.shape[:-1]).fill_(False)
        feature_starts = [0 for _ in features]
        for i, (feature, feature_len) in enumerate(zip(features, feature_lens)):
            diff = feature_len - feature_size
            if diff == 0:
                collated_features[i] = feature
            elif diff < 0:
                assert self.pad_feature
                collated_features[i] = torch.cat(
                    [feature, feature.new_full((-diff, feature_dim), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_features[i], feature_starts[i] = self.crop_to_max_size(
                    feature, feature_size
                )
        return collated_features, padding_mask, feature_starts

    @staticmethod
    def collate_tokens(
        values,
        pad_idx,
        eos_idx=None,
        left_pad=False,
        move_eos_to_beginning=False,
        pad_to_length=None,
        pad_to_multiple=1,
        pad_to_bsz=None,
    ):
        """Convert a list of 1d tensors into a padded 2d tensor."""
        size = max(v.size(0) for v in values)
        size = size if pad_to_length is None else max(size, pad_to_length)
        if pad_to_multiple != 1 and size % pad_to_multiple != 0:
            size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

        batch_size = (
            len(values) if pad_to_bsz is None else max(len(values), pad_to_bsz)
        )
        res = values[0].new(batch_size, size).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            if move_eos_to_beginning:
                if eos_idx is None:
                    dst[0] = src[-1]
                else:
                    dst[0] = eos_idx
                dst[1:] = src[:-1]
            else:
                dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][: len(v)])
        return res

    def collater_frm_label(self, kmeans, feature_size, feature_starts):
        assert self.label_rate > 0
        pad = self.num_classes[0] - 1
        s2f = self.label_rate / self.sample_rate

        frm_starts = [int(round(s * s2f)) for s in feature_starts]
        frm_size = int(round(feature_size * s2f))
        if not self.pad_feature:
            rem_size = [len(t) - s for t, s in zip(kmeans, frm_starts)]
            frm_size = min(frm_size, *rem_size)
        # Never allow a zero/negative label size: a cut whose kmeans do not
        # cover its random crop offset would produce an empty label tensor,
        # and the model would trim the whole batch's features to 0 frames,
        # crashing the zipformer position-embedding reshape.
        frm_size = max(frm_size, 1)
        kmeans = [t[s: s + frm_size] for t, s in zip(kmeans, frm_starts)]

        lengths = torch.LongTensor([len(t) for t in kmeans])
        kmeans = self.collate_tokens(kmeans, pad_idx=pad, left_pad=False)
        return kmeans, lengths


class HubertAsrDataset(torch.utils.data.Dataset):
    """
    Dataset for HuBERT finetuning on ASR data.

    Returns:
        {
            'cuts': CutSet,
            'audio': (B, T, F) float tensor (precomputed fbank),
            'padding_mask': (B, T) bool tensor,
            'feature_lens': list of int,
            'supervisions': collated supervision dict with 'text' field,
        }
    """

    def __init__(
        self,
        max_sample_size: Optional[int] = None,
        sample_rate: float = 16000,
        random_crop: bool = True,
        pad_audio: bool = True,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.random_crop = random_crop
        self.pad_feature = pad_audio
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.hdf5_fix = Hdf5MemoryIssueFix(reset_interval=100)

    def __getitem__(self, cuts: CutSet) -> Dict[str, Any]:
        self._validate(cuts)
        self.hdf5_fix.update()

        cuts = cuts.sort_by_duration(ascending=False)

        features = [torch.from_numpy(cut.load_features()) for cut in cuts]
        feature_lens = [cut.num_frames for cut in cuts]

        if self.pad_feature:
            feature_size = min(max(feature_lens), self.max_sample_size)
        else:
            feature_size = min(min(feature_lens), self.max_sample_size)

        features, padding_mask, feature_starts = self.collater_feature(
            features, feature_lens, feature_size
        )

        return {
            "cuts": cuts,
            "audio": features,
            "padding_mask": padding_mask,
            "feature_lens": feature_lens,
            "supervisions": default_collate(
                [
                    {"text": supervision.text}
                    for sequence_idx, cut in enumerate(cuts)
                    for supervision in cut.supervisions
                ]
            ),
        }

    def _validate(self, cuts: CutSet) -> None:
        validate(cuts)
        assert all(cut.has_recording for cut in cuts)

    def crop_to_max_size(self, feature, target_size):
        size = len(feature)
        diff = size - target_size
        if diff <= 0:
            return feature, 0
        start, end = 0, target_size
        if self.random_crop:
            start = np.random.randint(0, diff + 1)
            end = size - diff + start
        return feature[start:end, :], start

    def collater_feature(self, features, feature_lens, feature_size):
        feature_dim = features[0].shape[-1]
        collated_features = features[0].new_zeros(
            len(features), feature_size, feature_dim
        )
        padding_mask = torch.BoolTensor(collated_features.shape[:-1]).fill_(False)
        feature_starts = [0 for _ in features]
        for i, (feature, feature_len) in enumerate(zip(features, feature_lens)):
            diff = feature_len - feature_size
            if diff == 0:
                collated_features[i] = feature
            elif diff < 0:
                assert self.pad_feature
                collated_features[i] = torch.cat(
                    [feature, feature.new_full((-diff, feature_dim), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_features[i], feature_starts[i] = self.crop_to_max_size(
                    feature, feature_size
                )
        return collated_features, padding_mask, feature_starts

# Copyright      2022-2024  Xiaomi Corporation
#
# ADAPTED from:
#   - vietasr-saudi/ASR/zipformer/arabic_datamodule.py  (generic AsrDataModule base)
#   - vietasr-saudi/ASR/zipformer/asr_datamodule.py      (DataOceanAIArabicAsrDataModule — DDP-aware loaders)
#
# Changes from source:
#   - Dialect-parameterized cut loading (replaces hardcoded sada_cuts_*.jsonl.gz)
#   - Uses naming convention: gigaspeech3_{dialect}_cuts_{partition}.jsonl.gz
#   - Adds --dialect argument (ARE/EGY/IRQ/SAU/SYR)

import argparse
import fcntl
import gzip
import inspect
import io
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import lhotse
from lhotse import CutSet, Fbank, FbankConfig, load_manifest, load_manifest_lazy


def _log_describe(cuts, log_msg: str, enabled: bool = True):
    """Capture lhotse CutSet.describe() stdout into logging.

    Only runs when enabled (--log-describe); describe() iterates the full
    dataset, which is expensive for large manifests.
    """
    if not enabled:
        return
    buf = io.StringIO()
    try:
        import sys
        _stdout = sys.stdout
        sys.stdout = buf
        cuts.describe()
        sys.stdout = _stdout
        logging.info(f"{log_msg}\n{buf.getvalue().rstrip()}")
    finally:
        sys.stdout = sys.__stdout__
from lhotse.dataset import (
    CutConcatenate,
    CutMix,
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    PrecomputedFeatures,
    SimpleCutSampler,
    SpecAugment,
)
from lhotse.dataset.input_strategies import (
    OnTheFlyFeatures,
)
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader

from icefall.utils import str2bool
from recipe_textnorm import get_recipe_normalizer, has_recipe_textnorm


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class _RoundRobinCuts:
    """Lazily round-robins several CutSets (one cut per source per round).

    Each __iter__ call creates fresh iterators, so the stream is fully
    re-iterable (describe/map/sampler all iterate independently).  This
    mixes dialects WITHIN any finite window of the stream — single-dialect
    windows (file-concatenated order) make the LID head's CE trivial
    (a constant prediction matches an all-same-target batch), and also
    weaken the first-token LID supervision.
    """

    def __init__(self, cutsets):
        self.cutsets = [cs for cs in cutsets if cs is not None]

    def __iter__(self):
        iters = [iter(cs) for cs in self.cutsets]
        active = list(range(len(iters)))
        while active:
            nxt = []
            for i in active:
                try:
                    yield next(iters[i])
                except StopIteration:
                    continue
                nxt.append(i)
            active = nxt


def interleave_cutsets(cutsets):
    """Round-robin interleave several CutSets into one lazy, re-iterable
    stream.  Single input is returned unchanged; empty input returns None."""
    cutsets = [cs for cs in cutsets if cs is not None]
    if not cutsets:
        return None
    if len(cutsets) == 1:
        return cutsets[0]
    return CutSet(_RoundRobinCuts(cutsets))


# ══════════════════════════════════════════════════════════════════════
# Dialect oversampling and LID class weighting
# ══════════════════════════════════════════════════════════════════════

# Cut counts of the ARAB training stream, measured on the manifests actually
# loaded by semi_supervised_with_pseudo_label_train_cuts()
# (data/unsupervised/*_iter1_cleaned.jsonl.gz, 2026-10-11; DZA and SYR were
# replaced that day — DZA's *_cleaned_wavs set became *_cleaned and the old
# file became *_cleaned_full, SYR's _novideo set became *_cleaned).  These are
# the per-file line counts BEFORE the per-dialect score filter, matching what
# the loader logs as "<DIALECT>/<file>: N before".  Kept as a table so the
# stream composition and the inverse-frequency LID class weights can be
# derived without iterating ~12M lazy cuts.  REFRESH whenever those manifests
# are regenerated, or the derived weights silently go stale.
ARAB_TRAIN_COUNTS = {
    "DZA": 2_555_693,
    "EGY": 1_912_374,
    "IRQ": 2_542_807,
    "SYR": 708_248,
    "SAU": 2_627_488,
    "MSA": 1_549_152,
}


def parse_repeat_spec(spec) -> Dict[str, int]:
    """Parse a dialect-oversampling spec such as "SYR=2" or "SYR=3,EGY=2".

    A bare integer entry (no "=") applies to every dialect.  '', None and
    'none' all mean "no oversampling".
    """
    out: Dict[str, int] = {}
    if spec is None or str(spec).strip().lower() in ("", "none"):
        return out
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        name, _, k = item.partition("=")
        if not k:
            name, k = "*", name
        try:
            out[name.strip().upper()] = int(k)
        except ValueError:
            logging.warning(f"ignoring malformed --dialect-repeat entry {item!r}")
    return out


def repeat_named_cutsets(named_cutsets, spec):
    """Expand [(dialect, CutSet), ...] into the flat list given to
    interleave_cutsets, listing a dialect `spec[dialect]` times.

    _RoundRobinCuts consumes each list entry exactly once, so listing the
    same CutSet k times multiplies that dialect's exposure by k without
    duplicating anything on disk (the k iterators are independent).
    """
    if not spec:
        return [c for _, c in named_cutsets]
    out = []
    for name, cuts in named_cutsets:
        k = spec.get(str(name).upper(), spec.get("*", 1))
        out.extend([cuts] * max(int(k), 0))
    return out


def arab_lid_class_weights(repeat_spec=None):
    """Inverse-frequency CE weights for the LID head, in DIALECT_TOKENS order.

    w_c = N / (K * n_c) over the dialects that actually occur in training, so
    a predictor that outputs the class prior gets the same weighted loss as
    it would unweighted — which keeps --lid-loss-weight's meaning intact.

    n_c is the EFFECTIVE count, i.e. after --dialect-repeat, so oversampling
    and class weighting do not compound into a double correction.
    """
    from dialect_tokens import DIALECT_TOKENS

    spec = parse_repeat_spec(repeat_spec)
    default_k = spec.get("*", 1)
    eff = {d: n * spec.get(d, default_k) for d, n in ARAB_TRAIN_COUNTS.items()}
    present = {d: n for d, n in eff.items() if n > 0}
    if not present:
        return None
    k, total = len(present), sum(present.values())
    return [
        (total / (k * eff[d])) if eff.get(d, 0) > 0 else 1.0
        for d in DIALECT_TOKENS
    ]


class ArabicAsrDataModule:
    """
    DataModule for dialect-parameterized k2 ASR experiments.

    Uses the naming convention established by prepare_haitian_label.py:
        data/fbank/gigaspeech3_{dialect}_cuts_{partition}.jsonl.gz

    Supports 5 Arabic dialects: ARE, EGY, IRQ, SAU, SYR.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="ASR data related options",
            description="These options are used for the preparation of "
            "PyTorch DataLoaders from Lhotse CutSet's.",
        )

        group.add_argument(
            "--dialect",
            type=str,
            default="SAU",
            choices=["ARAB", "ARE", "DZA", "EGY", "ENG", "IRQ", "MSA", "SAU", "SYR"],
            help="Arabic dialect code. ARAB = multi-dialect combined model.",
        )

        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path("data/fbank"),
            help="Path to directory with train/valid/test cut manifests.",
        )

        group.add_argument(
            "--unsupervised-dir",
            type=Path,
            default=Path("data/unsupervised"),
            help="Directory scanned for *iter{N}*pseudolabeled*.jsonl.gz files "
            "when --train-with-pseudo-label is used.",
        )

        group.add_argument(
            "--dialect-textnorm",
            type=str2bool,
            default=False,
            help="Apply dialect-level recipe textnorm (word-level dialect "
            "mappings, filler removal, number standardization) to texts at "
            "data-load time. Default: shared character-level normalization "
            "only (see textnorm._common.normalize_char_level). Decode-side "
            "WER normalization always uses the dialect-level normalizer "
            "regardless of this flag.",
        )

        group.add_argument(
            "--max-duration",
            type=int,
            default=200,
            help="Maximum pooled recordings duration (seconds) in a single batch.",
        )

        group.add_argument(
            "--bucketing-sampler",
            type=str2bool,
            default=True,
            help="When enabled, use DynamicBucketingSampler.",
        )

        group.add_argument(
            "--num-buckets",
            type=int,
            default=30,
            help="Number of buckets for DynamicBucketingSampler.",
        )

        group.add_argument(
            "--concatenate-cuts",
            type=str2bool,
            default=False,
            help="Concatenate cuts to minimize padding.",
        )

        group.add_argument(
            "--duration-factor",
            type=float,
            default=1.0,
            help="Max duration factor for cut concatenation.",
        )

        group.add_argument(
            "--gap",
            type=float,
            default=1.0,
            help="Padding gap (seconds) between concatenated cuts.",
        )

        group.add_argument(
            "--on-the-fly-feats",
            type=str2bool,
            default=False,
            help="Compute features on-the-fly instead of using precomputed.",
        )

        group.add_argument(
            "--shuffle",
            type=str2bool,
            default=True,
            help="Shuffle training data for each epoch.",
        )

        group.add_argument(
            "--drop-last",
            type=str2bool,
            default=True,
            help="Drop the last incomplete batch.",
        )

        group.add_argument(
            "--return-cuts",
            type=str2bool,
            default=True,
            help="Include cut objects in batch['supervisions']['cut'].",
        )

        group.add_argument(
            "--num-workers",
            type=int,
            default=16,
            help="Number of DataLoader workers.",
        )

        group.add_argument(
            "--log-describe",
            type=str2bool,
            default=False,
            help="Run CutSet.describe() at startup (iterates the full dataset; "
            "slow for large manifests). Disabled by default.",
        )

        group.add_argument(
            "--enable-spec-aug",
            type=str2bool,
            default=True,
            help="Enable SpecAugment for training.",
        )

        group.add_argument(
            "--spec-aug-time-warp-factor",
            type=int,
            default=80,
            help="Time warping factor for SpecAugment (<1 to disable).",
        )

        group.add_argument(
            "--enable-musan",
            type=str2bool,
            default=False,
            help="Mix noise from MUSAN into training data.",
        )

        group.add_argument(
            "--input-strategy",
            type=str,
            default="PrecomputedFeatures",
            help="AudioSamples or PrecomputedFeatures.",
        )

    @property
    def prefix(self) -> str:
        return f"gigaspeech3_{self.args.dialect.lower()}"

    def _uppercase_text(self, cuts: CutSet) -> CutSet:
        """Uppercase all supervision texts in the CutSet (used for ENG dialect)."""

        def _upper_supervision_text(cut):
            for sup in cut.supervisions:
                if sup.text:
                    sup.text = sup.text.upper()
                if sup.custom and sup.custom.get("text_norm"):
                    sup.custom["text_norm"] = sup.custom["text_norm"].upper()
            return cut

        cuts = cuts.map(_upper_supervision_text)
        return cuts

    def _apply_recipe_textnorm(self, cuts: CutSet) -> CutSet:
        """Apply text normalization to all supervision texts.

        Default (--dialect-textnorm false): shared CHARACTER-LEVEL
        normalization (textnorm._common.normalize_char_level) — common to all
        dialects, removes diacritics/punctuation/tatweel, unifies alef
        variants, Persian letters and Eastern digits.  Does NOT rewrite word
        surface forms (keeps ال, ة and dialect words as spoken).

        With --dialect-textnorm true: additionally applies the dialect-level
        recipe normalize() (asr/textnorm/{DIALECT}.py) — word-level
        dialect mappings, filler removal, number standardization.  Decode-side
        WER normalization always uses the dialect-level normalizer regardless
        of this flag.
        """
        dialect = self.args.dialect.upper()

        # ARAB multi-dialect: detect per-cut dialect and apply correct textnorm
        if dialect == "ARAB":
            return self._apply_multidialect_textnorm(cuts)

        # Skip ENG — it uses _uppercase_text instead
        if dialect == "ENG":
            return cuts

        if not getattr(self.args, "dialect_textnorm", False):
            # ── Default: shared character-level normalization only ──
            from textnorm._common import normalize_char_level

            logging.info(
                "Normalizing text: shared character-level normalization "
                "(dialect-level textnorm disabled; pass --dialect-textnorm "
                "true to enable)"
            )

            def _normalize_char_level(cut):
                for sup in cut.supervisions:
                    if sup.text and sup.text.strip():
                        sup.text = normalize_char_level(sup.text.strip())
                return cut

            return cuts.map(_normalize_char_level)

        # ── --dialect-textnorm: dialect-level recipe normalize ──
        # Try recipe script first
        recipe_norm = get_recipe_normalizer(dialect)
        if recipe_norm is not None:
            import recipe_textnorm as _rt
            script_path = _rt._get_recipe_script_path(dialect)
            logging.info(
                f"Normalizing text: loaded dialect text norm from {script_path}"
            )

            def _normalize_with_recipe(cut):
                for sup in cut.supervisions:
                    if sup.text and sup.text.strip():
                        sup.text = recipe_norm(sup.text.strip())
                return cut

            cuts = cuts.map(_normalize_with_recipe)
            return cuts

        # No recipe script found: leave the text at its current
        # normalization.  Dialect-level textnorm is only provided by
        # asr/textnorm/{DIALECT}.py.
        logging.warning(
            f"No recipe textnorm found for dialect {dialect}; "
            "leaving text unchanged"
        )

        return cuts

    # ── ARAB multi-dialect support ──────────────────────────────────────
    ARAB_SUB_DIALECTS = ["ARE", "EGY", "SYR", "SAU", "IRQ", "DZA", "MSA"]

    @staticmethod
    def _tag_dialect(cut, dialect: str):
        """Tag a cut with its dialect in cut.custom['dialect']."""
        if cut.custom is None:
            cut.custom = {}
        cut.custom['dialect'] = dialect
        return cut

    def _apply_multidialect_textnorm(self, cuts: CutSet) -> CutSet:
        """For ARAB multi-dialect: apply per-dialect textnorm based on
        cut.custom['dialect'] (set during train_cuts() loading).

        Default: shared character-level normalization (all dialects).
        With --dialect-textnorm: per-dialect recipe normalize().
        """
        if not getattr(self.args, "dialect_textnorm", False):
            from textnorm._common import normalize_char_level

            logging.info(
                "ARAB textnorm: shared character-level normalization "
                "(dialect-level textnorm disabled; pass --dialect-textnorm "
                "true to enable)"
            )

            def _normalize_char_level(cut):
                for sup in cut.supervisions:
                    if sup.text and sup.text.strip():
                        sup.text = normalize_char_level(sup.text.strip())
                return cut

            return cuts.map(_normalize_char_level)

        # Load all available dialect normalizers
        normalizers = {}
        for dialect in self.ARAB_SUB_DIALECTS:
            norm = get_recipe_normalizer(dialect)
            if norm is not None:
                normalizers[dialect] = norm
                logging.info(f"  ARAB textnorm: loaded {dialect} normalizer")
            else:
                logging.warning(f"  ARAB textnorm: no normalizer for {dialect}")

        if not normalizers:
            logging.warning("ARAB: no multi-dialect normalizers available")
            return cuts

        def _normalize(cut):
            dialect = cut.custom.get('dialect', None) if cut.custom else None
            if dialect is None:
                # Fallback: use supervision language tag (string only;
                # pseudo-label cuts carry a float score there instead).
                lang = getattr(cut.supervisions[0], 'language', None) \
                    if cut.supervisions else None
                if isinstance(lang, str):
                    dialect = lang
            norm_fn = normalizers.get(dialect) if dialect else None
            if norm_fn is not None:
                for sup in cut.supervisions:
                    if sup.text and sup.text.strip():
                        try:
                            sup.text = norm_fn(sup.text.strip())
                        except Exception:
                            pass
            return cut

        logging.info(
            f"ARAB multi-dialect textnorm: applying per-dialect normalization "
            f"({len(normalizers)} normalizers loaded, using cut.custom['dialect'])"
        )
        return cuts.map(_normalize)

    @lru_cache()
    def train_cuts(self) -> CutSet:
        # ── ARAB: load each sub-dialect separately and tag with language ID ──
        if self.args.dialect.upper() == "ARAB":
            # MSA is always added via train.py (after the if-elif-else block)
            # to work with both supervised and pseudo-label modes.
            subs = [s for s in self.ARAB_SUB_DIALECTS if s != "MSA"]
            all_cuts = []
            for sub in subs:
                prefix = f"gigaspeech3_{sub.lower()}"
                path = self.args.manifest_dir / f"{prefix}_cuts_train.jsonl.gz"
                if path.is_file():
                    logging.info(f"Loading {sub} train cuts from {path}")
                    cuts = load_manifest_lazy(path)

                    # Duration filter — applied early to reduce textnorm work
                    _min_dur = getattr(self.args, "min_cut_duration", 1.0)
                    _max_dur = getattr(self.args, "max_cut_duration", 50.0)
                    cuts = cuts.filter(
                        lambda c, lo=_min_dur, hi=_max_dur: lo <= c.duration <= hi
                    )

                    # Apply textnorm inline (before mixing): dialect-level
                    # recipe normalize with --dialect-textnorm, otherwise the
                    # shared character-level normalization.
                    if getattr(self.args, "dialect_textnorm", False):
                        norm_fn = get_recipe_normalizer(sub)
                    else:
                        from textnorm._common import normalize_char_level
                        norm_fn = normalize_char_level
                    if norm_fn is not None:
                        _norm = norm_fn  # capture per-iteration
                        def _apply_norm(cut, fn=_norm):
                            for sup in cut.supervisions:
                                if sup.text and sup.text.strip():
                                    sup.text = fn(sup.text.strip())
                            return cut
                        cuts = cuts.map(_apply_norm)
                        logging.info(f"  {sub}: textnorm applied")
                    else:
                        logging.warning(f"  {sub}: no textnorm found")

                    all_cuts.append((sub, cuts))
                    _log_describe(cuts, f"  {sub} describe:", enabled=self.args.log_describe)
                else:
                    logging.warning(
                        f"{sub} train cuts not found at {path}, skipping"
                    )

            if not all_cuts:
                raise FileNotFoundError(
                    f"No sub-dialect train cuts found in {self.args.manifest_dir}"
                )

            # Round-robin interleave all dialects into one stream (mixed
            # batches; see interleave_cutsets for the rationale), applying
            # --dialect-repeat (e.g. "SYR=2") so a rare dialect gets more
            # exposure than its share of the corpus.
            _repeat = parse_repeat_spec(getattr(self.args, "dialect_repeat", None))
            result = interleave_cutsets(repeat_named_cutsets(all_cuts, _repeat))

            logging.info(
                f"ARAB combined train cuts: {len(all_cuts)} dialects loaded "
                f"(textnorm applied per-dialect); dialect-repeat="
                f"{_repeat or 'none'}"
            )
            _log_describe(result, "ARAB combined describe:", enabled=self.args.log_describe)
            return result

        # ── Single-dialect path ──
        # Duration filter applied before textnorm (mirrors the ARAB
        # sub-dialect path).  Describe is logged by train.py (final
        # train_cuts.describe, gated by --log-describe) to avoid iterating
        # the dataset twice.
        _min_dur = getattr(self.args, "min_cut_duration", 1.0)
        _max_dur = getattr(self.args, "max_cut_duration", 50.0)
        path = self.args.manifest_dir / f"{self.prefix}_cuts_train.jsonl.gz"
        if not path.is_file():
            # Fallback: try any cut file in manifest_dir
            candidates = sorted(Path(self.args.manifest_dir).glob("*.jsonl.gz"))
            if candidates:
                logging.warning(f"Train cuts not found at {path}, using {candidates[0].name}")
                cuts = load_manifest_lazy(candidates[0])
                cuts = cuts.filter(
                    lambda c, lo=_min_dur, hi=_max_dur: lo <= c.duration <= hi
                )
                if self.args.dialect == "ENG":
                    cuts = self._uppercase_text(cuts)
                cuts = self._apply_recipe_textnorm(cuts)
                return cuts
            raise FileNotFoundError(f"No cut files found in {self.args.manifest_dir}")
        logging.info(f"Loading train cuts from {path}")
        cuts = load_manifest_lazy(path)
        cuts = cuts.filter(
            lambda c, lo=_min_dur, hi=_max_dur: lo <= c.duration <= hi
        )
        if self.args.dialect == "ENG":
            cuts = self._uppercase_text(cuts)
        cuts = self._apply_recipe_textnorm(cuts)
        return cuts

    @lru_cache()
    def dev_cuts(self) -> CutSet:
        # Validation must use an independent, explicitly supplied dev split.
        # Never fall back to train: that would silently report training-set
        # metrics as validation metrics.
        path = self.args.manifest_dir / f"{self.prefix}_cuts_dev.jsonl.gz"
        if not path.is_file():
            raise FileNotFoundError(
                f"Independent dev cuts are required at {path}. "
                "Prepare a human-labeled dev manifest with fbank features; "
                "do not reuse train or test cuts as dev."
            )
        logging.info(f"Loading dev cuts from {path}")
        cuts = load_manifest_lazy(path)
        if self.args.dialect == "ENG":
            cuts = self._uppercase_text(cuts)
        if self.args.dialect != "ARAB":
            cuts = self._apply_recipe_textnorm(cuts)
        return cuts

    @lru_cache()
    def test_cuts(self) -> Dict[str, CutSet]:
        result = {}
        # Try to load dev + test.  Key includes dialect prefix so that
        # multi-dialect decode jobs sharing one decode_dir don't overwrite
        # each other's WER files (e.g. "SAU_test" vs "EGY_test").
        # --test-cuts-only restricts this to the test split (used by the
        # packaged dialect decode scripts, whose dev files
        # would otherwise be decoded too).
        _splits = ("test",) if getattr(self.args, "test_cuts_only", False) else ("dev", "test")
        for name in _splits:
            path = self.args.manifest_dir / f"{self.prefix}_cuts_{name}.jsonl.gz"
            if path.is_file():
                logging.info(f"Loading {name} cuts from {path}")
                cuts = load_manifest_lazy(path)
                if self.args.dialect == "ENG":
                    cuts = self._uppercase_text(cuts)
                cuts = self._apply_recipe_textnorm(cuts)
                key = f"{self.args.dialect}_{name}" if self.args.dialect != "ARAB" else name
                result[key] = cuts
        
        if self.args.dialect == "ENG":
            for name in ("dev", "test"):
                path = self.args.manifest_dir / f"spgispeech_cuts_{name}.jsonl.gz"
                if path.is_file():
                    logging.info(f"Loading spgispeech {name} cuts from {path}")
                    cuts = load_manifest_lazy(path)
                    cuts = self._uppercase_text(cuts)
                    result[f'spgispeech_{name}'] = cuts
                
                path = self.args.manifest_dir / f"librispeech_cuts_{name}-clean.jsonl.gz"
                logging.info(f"Loading librispeech {name} clean cuts from {path}")
                cuts = load_manifest_lazy(path)
                cuts = self._uppercase_text(cuts)
                result[f'librispeech_{name}_clean'] = cuts
                
                path = self.args.manifest_dir / f"librispeech_cuts_{name}-other.jsonl.gz"
                logging.info(f"Loading librispeech {name} other cuts from {path}")
                cuts = load_manifest_lazy(path)
                cuts = self._uppercase_text(cuts)
                result[f'librispeech_{name}_other'] = cuts
        
        if not result:
            raise FileNotFoundError(
                f"No dev/test cut manifests found in {self.args.manifest_dir}. "
                "Supply an explicit evaluation manifest or prepare the "
                f"{self.prefix}_cuts_{{dev,test}}.jsonl.gz files."
            )
        return result

    def train_dataloaders(
        self,
        cuts_train: Optional[CutSet] = None,
        sampler_state_dict: Optional[Dict[str, Any]] = None,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> DataLoader:
        if cuts_train is None:
            cuts_train = self.train_cuts()

        transforms = []
        if self.args.enable_musan:
            logging.info("Enable MUSAN")
            musan_path = self.args.manifest_dir / "musan_cuts.jsonl.gz"
            if musan_path.is_file():
                cuts_musan = load_manifest(musan_path)
                transforms.append(
                    CutMix(cuts=cuts_musan, p=0.5, snr=(10, 20), preserve_id=True)
                )

        if self.args.concatenate_cuts:
            logging.info(
                f"Using cut concatenation with duration_factor={self.args.duration_factor}, gap={self.args.gap}"
            )
            transforms = [
                CutConcatenate(
                    duration_factor=self.args.duration_factor, gap=self.args.gap
                )
            ] + transforms

        input_transforms = []
        if self.args.enable_spec_aug:
            logging.info("Enable SpecAugment")
            logging.info(f"Time warp factor: {self.args.spec_aug_time_warp_factor}")
            num_frame_masks = 10
            num_frame_masks_parameter = inspect.signature(
                SpecAugment.__init__
            ).parameters["num_frame_masks"]
            if num_frame_masks_parameter.default == 1:
                num_frame_masks = 2
            logging.info(f"Num frame mask: {num_frame_masks}")
            input_transforms.append(
                SpecAugment(
                    time_warp_factor=self.args.spec_aug_time_warp_factor,
                    num_frame_masks=num_frame_masks,
                    features_mask_size=27,
                    num_feature_masks=2,
                    frames_mask_size=100,
                )
            )

        logging.info("About to create train dataset")
        train = K2SpeechRecognitionDataset(
            input_strategy=eval(self.args.input_strategy)(),
            cut_transforms=transforms,
            input_transforms=input_transforms,
            return_cuts=self.args.return_cuts,
        )

        if self.args.on_the_fly_feats:
            train = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
                input_transforms=input_transforms,
                return_cuts=self.args.return_cuts,
            )

        if self.args.bucketing_sampler:
            logging.info("Using DynamicBucketingSampler.")
            train_sampler = DynamicBucketingSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                num_buckets=self.args.num_buckets,
                buffer_size=self.args.num_buckets * 2000,
                shuffle_buffer_size=self.args.num_buckets * 5000,
                drop_last=self.args.drop_last,
                world_size=world_size,
                rank=rank,
            )
        else:
            logging.info("Using SimpleCutSampler.")
            train_sampler = SimpleCutSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                world_size=world_size,
                rank=rank,
            )

        logging.info("About to create train dataloader")

        if sampler_state_dict is not None:
            logging.info("Loading sampler state dict")
            train_sampler.load_state_dict(sampler_state_dict)

        seed = torch.randint(0, 100000, ()).item()
        worker_init_fn = _SeedWorkers(seed)

        train_dl = DataLoader(
            train,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=True,
            prefetch_factor=4,
            worker_init_fn=worker_init_fn,
        )
        return train_dl

    def valid_dataloaders(
        self,
        cuts_valid: Optional[CutSet] = None,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> DataLoader:
        if cuts_valid is None:
            cuts_valid = self.dev_cuts()

        transforms = []
        if self.args.concatenate_cuts:
            transforms = [
                CutConcatenate(
                    duration_factor=self.args.duration_factor, gap=self.args.gap
                )
            ] + transforms

        logging.info("About to create dev dataset")
        if self.args.on_the_fly_feats:
            validate = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
                return_cuts=self.args.return_cuts,
            )
        else:
            validate = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                return_cuts=self.args.return_cuts,
            )

        valid_sampler = DynamicBucketingSampler(
            cuts_valid,
            max_duration=self.args.max_duration,
            shuffle=False,
            world_size=world_size,
            rank=rank,
        )

        logging.info("About to create dev dataloader")
        valid_dl = DataLoader(
            validate,
            sampler=valid_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
        )
        return valid_dl

    def test_dataloaders(self, cuts: CutSet) -> DataLoader:
        logging.debug("About to create test dataset")
        test = K2SpeechRecognitionDataset(
            input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)))
            if self.args.on_the_fly_feats
            else eval(self.args.input_strategy)(),
            return_cuts=self.args.return_cuts,
        )
        sampler = DynamicBucketingSampler(
            cuts,
            max_duration=self.args.max_duration,
            shuffle=False,
        )
        logging.debug("About to create test dataloader")
        test_dl = DataLoader(
            test,
            batch_size=None,
            sampler=sampler,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        return test_dl

    # ── Special data modes (used by finetune.py) ─────────────────────────

    @lru_cache()
    def semi_supervised_with_pseudo_label_train_cuts(self) -> CutSet:
        """Load pseudo-labeled unsupervised cuts for semi-supervised training.

        Scans data/unsupervised/ (CWD-relative) for *iter{N}_cleaned*.jsonl.gz
        files (duration + empty-text cleaned; SAU/SYR are the r4-filtered sets),
        falling back to *pseudolabeled*.jsonl.gz when none exist.  Optionally
        filtered by iteration and score range.

        Supports per-dialect score thresholds via --pseudo-label-score-dialects
        (JSON map), e.g. '{"egypt":[10,16],"saudi":[8,18],"iraq":[14,17]}'.
        Dialects not listed fall back to --pseudo-label-score-min / --max.
        """
        import json as _json

        iteration = getattr(self.args, "iteration", 1)
        score_min = getattr(self.args, "pseudo_label_score_min", 7.0)
        score_max = getattr(self.args, "pseudo_label_score_max", 16.0)

        # Per-dialect thresholds: parse JSON if provided
        dialect_scores: dict = {}
        raw = getattr(self.args, "pseudo_label_score_dialects", None)
        if raw and raw != "None":
            try:
                dialect_scores = _json.loads(raw)
                logging.info(f"Per-dialect pseudo-label score thresholds: {dialect_scores}")
            except _json.JSONDecodeError:
                logging.warning(f"Invalid pseudo-label-score-dialects JSON: {raw}")

        def _dialect_score_range(dialect: str) -> tuple:
            if dialect and dialect in dialect_scores:
                v = dialect_scores[dialect]
                return (v[0], v[1])
            return (score_min, score_max)

        # Detect source dialect from filename (long name → short code)
        _DIALECT_NAME_MAP = {
            "egypt": "EGY", "algeria": "DZA", "syria": "SYR",
            "saudi": "SAU", "iraq": "IRQ", "msa": "MSA",
        }
        def _detect_dialect_code(path: Path) -> str:
            for name, code in _DIALECT_NAME_MAP.items():
                if name in path.name:
                    return code
            return ""

        unsup_dir = Path(getattr(self.args, "unsupervised_dir", "data/unsupervised"))

        # (2026-08-17) The old *_textnormed file preference is commented out:
        # those files were pre-processed with the dialect-level textnorm, which
        # is now opt-in (--dialect-textnorm).  Textnorm is applied at load time
        # below, so we always load the *_pseudolabeled originals and skip stale
        # _textnormed/_filtered copies.
        #
        # (2026-08-31) Prefer the *_iter{N}_cleaned files: duration + empty-text
        # cleaned per dialect (SAU/SYR are the user's r4-filtered sets with
        # fixed feature storage paths).  Fall back to *_pseudolabeled when no
        # cleaned files exist for this iteration.
        def _find_cuts(search_dir: Path) -> list:
            """Return the training cut files for this iteration.

            Prefer *_cleaned (duration/empty-text filtered); fall back to
            *_pseudolabeled originals.  Stale _textnormed/_filtered copies
            are always skipped.
            """
            cleaned = sorted(search_dir.glob(
                f"*iter{iteration}*cleaned*.jsonl.gz"
            ))
            if cleaned:
                return [p for p in cleaned
                        if "_filtered_" not in p.name and "_textnormed" not in p.name]
            matches = sorted(search_dir.glob(
                f"*iter{iteration}*pseudolabeled*.jsonl.gz"
            ))
            return [p for p in matches
                    if "_filtered_" not in p.name and "_textnormed" not in p.name]

        cut_list = []
        if unsup_dir.is_dir():
            cut_list = _find_cuts(unsup_dir)
        if not cut_list:
            fallback_dir = Path("../../data/unsupervised")
            if fallback_dir.is_dir():
                cut_list = _find_cuts(fallback_dir)

        if not cut_list:
            logging.warning("No pseudo-label cuts found, falling back to train_cuts()")
            return self.train_cuts()

        # Single-dialect recipes may share one unsupervised/ directory holding
        # EVERY dialect's pseudo-labels (e.g. recipe/EGY/data/unsupervised has
        # algeria/egypt/iraq/saudi/syria files).  For a non-ARAB recipe keep
        # only files belonging to this recipe's dialect; files whose name
        # carries no dialect keyword (e.g. DZA/IRQ's ssl_split_*.jsonl.gz)
        # belong to this recipe by construction and are kept as-is.
        from collections import Counter
        _recipe_dialect = str(getattr(self.args, "dialect", "")).upper()
        # Some packaging dirs use a different code for the same dialect
        # (repro/recipe/DZA holds the Algerian data, whose files are named
        # ..._algeria_...): accept the equivalent codes too.
        _DIALECT_EQUIV = {}
        _accepted = {_recipe_dialect} | _DIALECT_EQUIV.get(_recipe_dialect, set())
        if _recipe_dialect and _recipe_dialect != "ARAB":
            kept_files, dropped_files = [], Counter()
            for p in cut_list:
                code = _detect_dialect_code(p)
                if code and code not in _accepted:
                    dropped_files[code] += 1
                    continue
                kept_files.append(p)
            if dropped_files:
                logging.info(
                    f"Pseudo-label dialect filter (recipe={_recipe_dialect}): "
                    f"kept {len(kept_files)} file(s), dropped "
                    f"{sum(dropped_files.values())} file(s) from other dialects "
                    f"{dict(dropped_files)}"
                )
            cut_list = kept_files
            if not cut_list:
                logging.warning(
                    f"No pseudo-label files for dialect {_recipe_dialect} "
                    f"in {unsup_dir.resolve()}, falling back to train_cuts()"
                )
                return self.train_cuts()

        # Log per-dialect file counts
        dialect_counts = Counter(_detect_dialect_code(p) for p in cut_list)
        logging.info(
            f"Semi-supervised pseudo-label cuts: {len(cut_list)} files "
            f"from {cut_list[0].parent.resolve()}"
        )
        logging.info(f"  Per-dialect: {dict(dialect_counts)}")

        # Load per-file: score-filter, then apply textnorm, then combine
        processed_parts = []

        # Textnorm selection (2026-08-17): shared character-level normalization
        # by default; per-dialect recipe normalizers only with
        # --dialect-textnorm true.  (The old _textnormed-file shortcut, where
        # pre-processed files skipped normalization, is commented out — see
        # _find_cuts above.)
        from textnorm._common import normalize_char_level
        use_dialect_norm = getattr(self.args, "dialect_textnorm", False)
        _dialect_norms = {}
        if use_dialect_norm:
            for d_code in set(_DIALECT_NAME_MAP.values()):
                norm = get_recipe_normalizer(d_code)
                if norm is not None:
                    _dialect_norms[d_code] = norm

        for p in cut_list:
            dialect = _detect_dialect_code(p)
            d_min, d_max = _dialect_score_range(dialect)
            if use_dialect_norm:
                # Fall back to character-level when no recipe normalizer
                norm_fn = _dialect_norms.get(dialect) or normalize_char_level
            else:
                norm_fn = normalize_char_level

            cuts = lhotse.load_manifest_lazy(p)
            _dmin, _dmax = d_min, d_max  # capture per-iteration values

            def _filter(cut, _min=_dmin, _max=_dmax):
                for sup in cut.supervisions:
                    score = getattr(sup, 'language', None)
                    if score is not None and not (_min <= score <= _max):
                        return None
                return cut

            # Score filter + duration filter (no textnorm yet — keep describe fast)
            filtered = cuts.map(_filter).filter(lambda c: c is not None)
            _min_dur = getattr(self.args, "min_cut_duration", 1.0)
            _max_dur = getattr(self.args, "max_cut_duration", 50.0)
            filtered = filtered.filter(
                lambda c, lo=_min_dur, hi=_max_dur: lo <= c.duration <= hi
            )

            # Per-cut dialect tag (pseudo-label cuts carry neither
            # custom['dialect'] nor a string language — the dialect is known
            # from the filename).  Needed by multi-bpe encoding, dialect-token
            # training, and --dialect-textnorm; tag whenever we detected one.
            if dialect:
                _d = dialect  # capture per-iteration value

                def _tag(cut, _d=_d):
                    return self._tag_dialect(cut, _d)

                filtered = filtered.map(_tag)

            processed_parts.append((dialect, filtered, norm_fn))
            try:
                n_before = sum(1 for _ in gzip.open(p, 'rt'))
            except Exception:
                n_before = "?"
            logging.info(
                f"  {dialect}/{p.name}: {n_before} before "
                f"(score [{d_min},{d_max}])"
            )

        if not processed_parts:
            logging.warning("All pseudo-label cuts filtered out, falling back to train_cuts()")
            return self.train_cuts()

        # ── MSA source (supervised) ──────────────────────────────────────
        # Fold MSA into the SAME flat round-robin as the pseudo-label
        # dialects, so that:
        #   * every cut is tagged custom['dialect']='MSA' — the LID head CE
        #     and the dialect embedding consume this tag directly instead of
        #     relying on the supervision-language fallback;
        #   * every sampler window contains MSA alongside the dialects;
        #   * MSA bypasses the pseudo-label SCORE filter (its supervision
        #     language is the string 'MSA', not a numeric score).
        if getattr(self.args, "train_with_msa", False):
            _msa_tn = self.args.manifest_dir / "gigaspeech3_msa_cuts_train_textnormed.jsonl.gz"
            _msa_raw = self.args.manifest_dir / "gigaspeech3_msa_cuts_train.jsonl.gz"
            if _msa_tn.is_file():
                msa_path, msa_norm_fn = _msa_tn, None  # already textnormed
            elif _msa_raw.is_file():
                from textnorm._common import normalize_char_level
                msa_path, msa_norm_fn = _msa_raw, normalize_char_level
            else:
                msa_path = None
                logging.warning(
                    f"train_with_msa=True but no MSA manifest in {self.args.manifest_dir}"
                )
            if msa_path is not None:
                _min_dur = getattr(self.args, "min_cut_duration", 1.0)
                _max_dur = getattr(self.args, "max_cut_duration", 50.0)
                msa_cuts = load_manifest_lazy(msa_path)
                msa_cuts = msa_cuts.map(lambda c: self._tag_dialect(c, "MSA"))
                msa_cuts = msa_cuts.filter(
                    lambda c, lo=_min_dur, hi=_max_dur: lo <= c.duration <= hi
                )
                processed_parts.append(("MSA", msa_cuts, msa_norm_fn))
                logging.info(
                    f"MSA folded into the training stream: {msa_path.name} "
                    f"(tagged custom['dialect']='MSA', duration [{_min_dur},{_max_dur}])"
                )

        # Phase 1: combine per-dialect, describe (no textnorm yet — fast)
        from itertools import groupby
        unnormed_parts = []  # (dialect, unnormed_cuts, norm_fn)
        for dialect, parts in groupby(processed_parts, key=lambda x: x[0]):
            parts_list = list(parts)
            dialect_cuts = parts_list[0][1]
            for p in parts_list[1:]:
                dialect_cuts = dialect_cuts + p[1]
            _log_describe(dialect_cuts, f"  === {dialect} pseudo-label describe ===", enabled=self.args.log_describe)
            fn = parts_list[0][2]
            unnormed_parts.append((dialect, dialect_cuts, fn))

        # Phase 2: combined describe (no textnorm yet)
        all_unnormed = unnormed_parts[0][1]
        for _, cuts, _ in unnormed_parts[1:]:
            all_unnormed = all_unnormed + cuts
        logging.info(
            f"Pseudo-label all files loaded, textnorm mode: "
            f"{'dialect-level' if use_dialect_norm else 'character-level'}"
            f" ({len(_dialect_norms)} dialect normalizers ready)"
        )
        _log_describe(all_unnormed, "=== Pseudo-label combined describe ===", enabled=self.args.log_describe)

        # Phase 3: apply textnorm (lazy — will execute during training)
        normed_parts = []
        for dialect, cuts, fn in unnormed_parts:
            if fn is not None:
                _norm = fn
                def _apply_norm(cut, fn=_norm):
                    for sup in cut.supervisions:
                        if sup.text and sup.text.strip():
                            sup.text = fn(sup.text.strip())
                    return cut
                cuts = cuts.map(_apply_norm)
            normed_parts.append((dialect, cuts))
        # Round-robin interleave the per-dialect streams instead of
        # concatenating them: any sampler window then contains all dialects,
        # which the LID head / first-token LID need to learn real
        # discrimination (single-dialect windows make the CE trivial).
        #
        # --dialect-repeat oversamples a rare dialect by listing its CutSet
        # more than once (see repeat_named_cutsets).  Scoped to the ARAB
        # multi-dialect run: for a single-dialect recipe the spec would match
        # that recipe's own dialect and merely double the epoch length.
        _repeat = (
            parse_repeat_spec(getattr(self.args, "dialect_repeat", None))
            if self.args.dialect.upper() == "ARAB"
            else {}
        )
        if _repeat:
            logging.info(f"Dialect oversampling (--dialect-repeat): {_repeat}")
        return interleave_cutsets(repeat_named_cutsets(normed_parts, _repeat))

    @lru_cache()
    def transfer_learning_filtered_train_cuts(self) -> CutSet:
        """Load filtered transfer-learning datasets.

        Scans ../../data/transfer_datasets/ (project-level) for files
        matching *filtered*train*.jsonl.gz. Falls back to train_cuts().
        """
        transfer_dir = Path("../../data/transfer_datasets")
        if transfer_dir.is_dir():
            cut_list = sorted(
                p for p in transfer_dir.iterdir()
                if p.suffix == ".gz" and p.name.endswith(".jsonl.gz")
                and "filtered" in p.name and "train" in p.name
            )
            if cut_list:
                logging.info(
                    f"Transfer-learning cuts: {len(cut_list)} files"
                )
                return self._apply_recipe_textnorm(
                    lhotse.combine(lhotse.load_manifest_lazy(p) for p in cut_list)
                )

        logging.warning("No transfer-learning cuts found, falling back to train_cuts()")
        return self.train_cuts()

    @lru_cache()
    def tts_train_cuts(self) -> CutSet:
        """Load TTS-generated training cuts.

        Loads a named TTS cut file from manifest_dir, or falls back to
        train_cuts().
        """
        tts_name = getattr(self.args, "tts_cuts_name", None)
        if tts_name:
            path = self.args.manifest_dir / f"{tts_name}.jsonl.gz"
            if path.is_file():
                logging.info(f"TTS train cuts: {path}")
                return self._apply_recipe_textnorm(load_manifest_lazy(path))

        # Fallback: look for common TTS cut files
        for name in [
            "sada_cuts_train_21h_real_23h_tts_sp",
            "merged_best_wer_tts_114h",
            "merged_best_wer_cuts_train",
        ]:
            path = self.args.manifest_dir / f"{name}.jsonl.gz"
            if path.is_file():
                logging.info(f"TTS train cuts (auto-detected): {path}")
                return self._apply_recipe_textnorm(load_manifest_lazy(path))

        logging.warning("No TTS cuts found, falling back to train_cuts()")
        return self.train_cuts()

    @lru_cache()
    def multiasr_saudi_cuts(self) -> CutSet:
        """Load MultiASR benchmark test set for Saudi Arabic."""
        path = self.args.manifest_dir / "multiasr_bench_cuts_SAU.jsonl.gz"
        if path.is_file():
            return self._apply_recipe_textnorm(load_manifest_lazy(path))
        logging.warning(f"MultiASR benchmark not found at {path}")
        return self.test_cuts().get("test", self.dev_cuts())

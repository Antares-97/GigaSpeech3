#!/usr/bin/env python3
# Copyright    2021-2023  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Mingshuang Luo,
#                                                       Zengwei Yao,
#                                                       Daniel Povey)
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
It supports training with:
  - transducer loss (default)
  - ctc loss
  - attention decoder loss
  - cr-ctc loss (should use half the max-duration compared to regular ctc)
"""


import argparse
import copy
import io
import logging
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union

import k2
import optim
import sentencepiece as spm
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import dialect_tokens
from datamodule import ArabicAsrDataModule, _log_describe, interleave_cutsets
from zipformer.attention_decoder import AttentionDecoderModel
from zipformer.decoder import Decoder
from zipformer.joiner import Joiner
from lhotse import load_manifest_lazy
from lhotse.cut import Cut
from lhotse.dataset import SpecAugment
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from zipformer.model import AsrModel
from zipformer.optim import Eden, ScaledAdam
from zipformer.tri_scheduler import TriStageLRSchedule
from zipformer.scaling import ScheduledFloat
from zipformer.subsampling import Conv2dSubsampling
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from zipformer.zipformer import Zipformer2

from icefall import diagnostics
from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import (
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.err import raise_grad_scale_is_too_small_error
from icefall.hooks import register_inf_check_hooks
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    get_parameter_groups_with_lrs,
    setup_logger,
    str2bool,
)

LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, optim.LRScheduler]

# ── Multi-BPE (union vocab) support ─────────────────────────────────────────
# MULTI_BPE=1 时，各方言用自己的 lang_bpe_500 编码，piece 经去重映射表折算到
# 统一 union 词表（由 recipe/ARAB/scripts/build_union_vocab.py 生成）。
# SentencePieceProcessor 不可序列化，不放进 params（checkpoint 只保存路径与
# 词表大小），运行时通过本模块级上下文访问。

MULTI_BPE_DIALECTS = ["IRQ", "SAU", "SYR", "EGY", "MSA"]

# Manifest language codes -> canonical dialect codes.
# dev/test manifests tag Algerian cuts as "DZA"; supervised train manifests
# use lowercase long names ("algeria", "iraq"). ARE has no union dialect and
# is intentionally left unmapped (falls back to MSA in _encode_text_multi_bpe).
_DIALECT_ALIASES = {
    "DZA": "DZA",
    "DZ": "DZA",
    "ALGERIA": "DZA",
    "IRAQ": "IRQ",
}


def _canonical_dialect(dialect) -> str:
    d = str(dialect).upper()
    return _DIALECT_ALIASES.get(d, d)


_multi_bpe_ctx = {
    "sps": {},       # dialect -> SentencePieceProcessor
    "id_maps": {},   # dialect -> list[int] (piece_id -> union_id)
    "id2piece": [],  # union_id -> piece string
}


def load_multi_bpe(params: AttributeDict) -> None:
    """Load union vocab tables + per-dialect tokenizers.

    Fills the module-level context and sets params.blank_id / sos_id / eos_id
    / vocab_size from the union vocab.
    """
    import json as _json

    union_dir = Path(params.union_dir)
    id2piece = _json.loads(
        (union_dir / "id2piece.json").read_text(encoding="utf-8")
    )
    dialect_id_map = _json.loads(
        (union_dir / "dialect_id_map.json").read_text(encoding="utf-8")
    )
    missing = [d for d in MULTI_BPE_DIALECTS if d not in dialect_id_map]
    assert not missing, f"union vocab missing dialects: {missing}"

    sps = {}
    for d in MULTI_BPE_DIALECTS:
        path = getattr(params, f"bpe_model_{d.lower()}")
        sp_d = spm.SentencePieceProcessor()
        sp_d.load(path)
        sps[d] = sp_d

    blanks = {d: s.piece_to_id("<blk>") for d, s in sps.items()}
    sos_eos = {d: s.piece_to_id("<sos/eos>") for d, s in sps.items()}
    assert len(set(blanks.values())) == 1, f"blank ids differ: {blanks}"
    assert len(set(sos_eos.values())) == 1, f"sos/eos ids differ: {sos_eos}"

    _multi_bpe_ctx["sps"] = sps
    _multi_bpe_ctx["id_maps"] = dialect_id_map
    _multi_bpe_ctx["id2piece"] = id2piece

    params.blank_id = blanks[MULTI_BPE_DIALECTS[0]]
    params.sos_id = params.eos_id = sos_eos[MULTI_BPE_DIALECTS[0]]
    params.vocab_size = len(id2piece)
    logging.info(
        f"Multi-BPE enabled: union vocab size {len(id2piece)}, "
        f"dialects {MULTI_BPE_DIALECTS}, blank_id={params.blank_id}"
    )


def _dialect_of_cut(cut) -> str:
    """Determine the dialect of a cut.

    Priority: cut.custom['dialect'] -> supervision language (string only,
    pseudo-label cuts carry a float score there) -> MSA default.
    Returns canonical dialect codes (DZA stays DZA).
    """
    d = cut.custom.get("dialect") if cut.custom else None
    if d:
        return _canonical_dialect(d)
    if cut.supervisions:
        lang = cut.supervisions[0].language
        if isinstance(lang, str):
            return _canonical_dialect(lang)
    return "MSA"


_unknown_dialect_warned = set()


def _encode_text_multi_bpe(text: str, dialect: str) -> list:
    """Encode text with its dialect's BPE and map pieces to union IDs."""
    d = _canonical_dialect(dialect)
    if d not in _multi_bpe_ctx["id_maps"]:
        if d not in _unknown_dialect_warned:
            _unknown_dialect_warned.add(d)
            logging.warning(
                f"No union id_map for dialect '{d}': encoding with MSA BPE "
                f"(check manifests for unexpected language codes)"
            )
        d = "MSA"
    ids = _multi_bpe_ctx["sps"][d].encode(text, out_type=int)
    return [_multi_bpe_ctx["id_maps"][d][i] for i in ids]


class DualWriter:
    """A writer that simultaneously logs to TensorBoard and/or WandB.

    Implements the same ``add_scalar`` interface as ``SummaryWriter`` so it
    can be passed to ``MetricsTracker.write_summary()`` as a drop-in.
    Supports WandB offline mode for air-gapped GPU machines — sync logs
    later with ``wandb sync <dir>`` from a networked machine.
    """

    def __init__(
        self,
        tb_writer: Optional[SummaryWriter] = None,
        use_wandb: bool = False,
        wandb_project: str = "gigaspeech3-arabic",
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[dict] = None,
        wandb_dir: str = ".",
    ):
        self.tb_writer = tb_writer
        self.use_wandb = use_wandb
        if use_wandb:
            import wandb

            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config=wandb_config,
                mode="offline",
                dir=wandb_dir,
            )

    def add_scalar(self, tag: str, scalar_value: float, global_step: int):
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(tag, scalar_value, global_step)
        if self.use_wandb:
            import wandb

            wandb.log({tag: scalar_value}, step=global_step)

    def close(self):
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self.use_wandb:
            import wandb

            wandb.finish()


def get_adjusted_batch_count(params: AttributeDict) -> float:
    # returns the number of batches we would have used so far if we had used the reference
    # duration.  This is for purposes of set_batch_count().
    return (
        params.batch_idx_train
        * (params.max_duration * params.world_size)
        / params.ref_duration
    )


def set_batch_count(model: Union[nn.Module, DDP], batch_count: float) -> None:
    if isinstance(model, DDP):
        # get underlying nn.Module
        model = model.module
    for name, module in model.named_modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count
        if hasattr(module, "name"):
            module.name = name


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--num-encoder-layers",
        type=str,
        default="2,2,3,4,3,2",
        help="Number of zipformer encoder layers per stack, comma separated.",
    )

    parser.add_argument(
        "--downsampling-factor",
        type=str,
        default="1,2,4,8,4,2",
        help="Downsampling factor for each stack of encoder layers.",
    )

    parser.add_argument(
        "--feedforward-dim",
        type=str,
        default="512,768,1024,1536,1024,768",
        help="Feedforward dimension of the zipformer encoder layers, per stack, comma separated.",
    )

    parser.add_argument(
        "--num-heads",
        type=str,
        default="4,4,4,8,4,4",
        help="Number of attention heads in the zipformer encoder layers: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--encoder-dim",
        type=str,
        default="192,256,384,512,384,256",
        help="Embedding dimension in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--query-head-dim",
        type=str,
        default="32",
        help="Query/key dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--value-head-dim",
        type=str,
        default="12",
        help="Value dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--pos-head-dim",
        type=str,
        default="4",
        help="Positional-encoding dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--pos-dim",
        type=int,
        default="48",
        help="Positional-encoding embedding dimension",
    )

    parser.add_argument(
        "--encoder-unmasked-dim",
        type=str,
        default="192,192,256,256,256,192",
        help="Unmasked dimensions in the encoders, relates to augmentation during training.  "
        "A single int or comma-separated list.  Must be <= each corresponding encoder_dim.",
    )

    parser.add_argument(
        "--cnn-module-kernel",
        type=str,
        default="31,31,15,15,15,31",
        help="Sizes of convolutional kernels in convolution modules in each encoder stack: "
        "a single int or comma-separated list.",
    )

    parser.add_argument(
        "--decoder-dim",
        type=int,
        default=512,
        help="Embedding dimension in the decoder model.",
    )

    parser.add_argument(
        "--joiner-dim",
        type=int,
        default=512,
        help="""Dimension used in the joiner model.
        Outputs from the encoder and decoder model are projected
        to this dimension before adding.
        """,
    )

    parser.add_argument(
        "--attention-decoder-dim",
        type=int,
        default=512,
        help="""Dimension used in the attention decoder""",
    )

    parser.add_argument(
        "--attention-decoder-num-layers",
        type=int,
        default=6,
        help="""Number of transformer layers used in attention decoder""",
    )

    parser.add_argument(
        "--attention-decoder-attention-dim",
        type=int,
        default=512,
        help="""Attention dimension used in attention decoder""",
    )

    parser.add_argument(
        "--attention-decoder-num-heads",
        type=int,
        default=8,
        help="""Number of attention heads used in attention decoder""",
    )

    parser.add_argument(
        "--attention-decoder-feedforward-dim",
        type=int,
        default=2048,
        help="""Feedforward dimension used in attention decoder""",
    )

    parser.add_argument(
        "--causal",
        type=str2bool,
        default=False,
        help="If True, use causal version of model.",
    )

    parser.add_argument(
        "--chunk-size",
        type=str,
        default="16,32,64,-1",
        help="Chunk sizes (at 50Hz frame rate) will be chosen randomly from this list during training. "
        " Must be just -1 if --causal=False",
    )

    parser.add_argument(
        "--left-context-frames",
        type=str,
        default="64,128,256,-1",
        help="Maximum left-contexts for causal training, measured in frames which will "
        "be converted to a number of chunks.  If splitting into chunks, "
        "chunk left-context frames will be chosen randomly from this list; else not relevant.",
    )

    parser.add_argument(
        "--use-transducer",
        type=str2bool,
        default=True,
        help="If True, use Transducer head.",
    )

    parser.add_argument(
        "--use-ctc",
        type=str2bool,
        default=False,
        help="If True, use CTC head.",
    )

    parser.add_argument(
        "--use-attention-decoder",
        type=str2bool,
        default=False,
        help="If True, use attention-decoder head.",
    )

    parser.add_argument(
        "--use-cr-ctc",
        type=str2bool,
        default=False,
        help="If True, use consistency-regularized CTC.",
    )

    parser.add_argument(
        "--encoder-path",
        type=str,
        default=None,
        help="load RNN-T encoder from this path",
    )


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--final-downsample",
        type=str2bool,
        default=False,
        help="Whether to use final downsample in the encoder.",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--use-wandb",
        type=str2bool,
        default=False,
        help="Enable WandB logging in offline mode (sync later with wandb sync).",
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        default="gigaspeech3-arabic",
        help="WandB project name.",
    )

    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="WandB run name (auto-generated if not set).",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=30,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="""Resume training from this epoch. It should be positive.
        If larger than 1, it will load checkpoint from
        exp-dir/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--start-batch",
        type=int,
        default=0,
        help="""If positive, --start-epoch is ignored and
        it loads the checkpoint from exp-dir/checkpoint-{start_batch}.pt
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipformer/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
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
        "--multi-bpe",
        type=str2bool,
        default=False,
        help="Use per-dialect BPE models merged into one deduplicated union "
        "vocab (MULTI_BPE=1). Requires --union-dir tables built by "
        "recipe/ARAB/scripts/build_union_vocab.py.",
    )

    parser.add_argument(
        "--union-dir",
        type=str,
        default="data/lang_bpe_union",
        help="Directory with union vocab tables "
        "(tokens.txt, dialect_id_map.json, id2piece.json).",
    )

    for _d in MULTI_BPE_DIALECTS:
        parser.add_argument(
            f"--bpe-model-{_d.lower()}",
            type=str,
            default=f"../{_d}/data/lang_bpe_500/unigram_500.model",
            help=f"Path to the {_d} BPE model (multi-bpe mode).",
        )

    parser.add_argument(
        "--base-lr", type=float, default=0.045, help="The base learning rate."
    )

    parser.add_argument(
        "--lr-batches",
        type=float,
        default=7500,
        help="""Number of steps that affects how rapidly the learning rate
        decreases. We suggest not to change this.""",
    )

    parser.add_argument(
        "--lr-epochs",
        type=float,
        default=3.5,
        help="""Number of epochs that affects how rapidly the learning rate decreases.
        """,
    )

    parser.add_argument(
        "--max-lr-update",
        type=int,
        default=80000,
        help="For tri-stage scheduler (used when --phase-ratio is set)",
    )

    parser.add_argument(
        "--phase-ratio",
        type=str,
        default=None,
        help="Tri-stage scheduler phase ratio, e.g. (0.1, 0.4, 0.5). "
        "When set, uses TriStageLRSchedule instead of Eden.",
    )

    parser.add_argument(
        "--ref-duration",
        type=float,
        default=600,
        help="Reference batch duration for purposes of adjusting batch counts for setting various "
        "schedules inside the model",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )

    parser.add_argument(
        "--prune-range",
        type=int,
        default=5,
        help="The prune range for rnnt loss, it means how many symbols(context)"
        "we are using to compute the loss",
    )

    parser.add_argument(
        "--lm-scale",
        type=float,
        default=0.25,
        help="The scale to smooth the loss with lm "
        "(output of prediction network) part.",
    )

    parser.add_argument(
        "--am-scale",
        type=float,
        default=0.0,
        help="The scale to smooth the loss with am (output of encoder network) part.",
    )

    parser.add_argument(
        "--simple-loss-scale",
        type=float,
        default=0.5,
        help="To get pruning ranges, we will calculate a simple version"
        "loss(joiner is just addition), this simple loss also uses for"
        "training (as a regularization item). We will scale the simple loss"
        "with this parameter before adding to the final loss.",
    )

    parser.add_argument(
        "--ctc-loss-scale",
        type=float,
        default=0.2,
        help="Scale for CTC loss.",
    )

    parser.add_argument(
        "--cr-loss-scale",
        type=float,
        default=0.2,
        help="Scale for consistency-regularization loss.",
    )

    parser.add_argument(
        "--time-mask-ratio",
        type=float,
        default=2.5,
        help="When using cr-ctc, we increase the amount of time-masking in SpecAugment.",
    )

    parser.add_argument(
        "--attention-decoder-loss-scale",
        type=float,
        default=0.8,
        help="Scale for attention-decoder loss.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--print-diagnostics",
        type=str2bool,
        default=False,
        help="Accumulate stats on activations, print them and exit.",
    )

    parser.add_argument(
        "--inf-check",
        type=str2bool,
        default=False,
        help="Add hooks to check for infinite module outputs and gradients.",
    )

    parser.add_argument(
        "--save-every-n",
        type=int,
        default=10000,
        help="""Save checkpoint after processing this number of batches"
        periodically. We save checkpoint to exp-dir/ whenever
        params.batch_idx_train % save_every_n == 0. The checkpoint filename
        has the form: f'exp-dir/checkpoint-{params.batch_idx_train}.pt'
        Note: It also saves checkpoint to `exp-dir/epoch-xxx.pt` at the
        end of each epoch where `xxx` is the epoch number counting from 1.
        """,
    )

    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=5,
        help="""Only keep this number of checkpoints on disk.
        For instance, if it is 3, there are only 3 checkpoints
        in the exp-dir with filenames `checkpoint-xxx.pt`.
        It does not affect checkpoints with name `epoch-xxx.pt`.
        """,
    )

    parser.add_argument(
        "--average-period",
        type=int,
        default=200,
        help="""Update the averaged model, namely `model_avg`, after processing
        this number of batches. `model_avg` is a separate version of model,
        in which each floating-point parameter is the average of all the
        parameters from the start of training. Each time we take the average,
        we do: `model_avg = model * (average_period / batch_idx_train) +
            model_avg * ((batch_idx_train - average_period) / batch_idx_train)`.
        """,
    )

    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=False,
        help="Whether to use half precision training.",
    )

    parser.add_argument(
        "--use-bf16",
        type=str2bool,
        default=False,
        help="Whether to use bf16 in AMP.",
    )


    parser.add_argument(
        "--freeze-encoder",
        type=str2bool,
        default=False,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--freeze-encoder-epoch",
        type=int,
        default=0,
        help="Freeze encoder before training reach this epoch",
    )

    parser.add_argument(
        "--freeze-encoder-step",
        type=int,
        default=0,
        help="freeze encoder in an epoch's training before this step",
    )

    parser.add_argument(
        "--warmup-encoder-step",
        type=int,
        default=0,
        help="After freeze-encoder-step, encoder-scale will increase to 1 linearly after warmup-step",
    )

    add_model_arguments(parser)

    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    are saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

        - best_train_loss: Best training loss so far. It is used to select
                           the model that has the lowest training loss. It is
                           updated during the training.

        - best_valid_loss: Best validation loss so far. It is used to select
                           the model that has the lowest validation loss. It is
                           updated during the training.

        - best_train_epoch: It is the epoch that has the best training loss.

        - best_valid_epoch: It is the epoch that has the best validation loss.

        - batch_idx_train: Used to writing statistics to tensorboard. It
                           contains number of batches trained so far across
                           epochs.

        - log_interval:  Print training loss if batch_idx % log_interval` is 0

        - reset_interval: Reset statistics if batch_idx % reset_interval is 0

        - valid_interval:  Run validation if batch_idx % valid_interval is 0

        - feature_dim: The model input dim. It has to match the one used
                       in computing features.

        - subsampling_factor:  The subsampling factor for the model.

        - warm_step: The warmup period that dictates the decay of the
              scale on "simple" (un-pruned) loss.
    """
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            # multi-bpe union vocab (MULTI_BPE=1); paths kept so checkpoints
            # carry enough info to rebuild the context on resume
            "multi_bpe": False,
            "union_dir": "data/lang_bpe_union",
            **{f"bpe_model_{d.lower()}": f"../{d}/data/lang_bpe_500/unigram_500.model"
               for d in MULTI_BPE_DIALECTS},
            "log_interval": 50,
            "reset_interval": 200,
            # "valid_interval": 3000,  # For the 100h subset, use 800
            "valid_interval": 20000,
            # parameters for zipformer
            "feature_dim": 80,
            "subsampling_factor": 4,  # not passed in, this is fixed.
            # parameters for attention-decoder
            "ignore_id": -1,
            "label_smoothing": 0.1,
            "warm_step": 2000,
            "env_info": get_env_info(),
        }
    )

    return params


def _to_int_tuple(s: str):
    return tuple(map(int, s.split(",")))


def get_encoder_embed(params: AttributeDict) -> nn.Module:
    # encoder_embed converts the input of shape (N, T, num_features)
    # to the shape (N, (T - 7) // 2, encoder_dims).
    # That is, it does two things simultaneously:
    #   (1) subsampling: T -> (T - 7) // 2
    #   (2) embedding: num_features -> encoder_dims
    # In the normal configuration, we will downsample once more at the end
    # by a factor of 2, and most of the encoder stacks will run at a lower
    # sampling rate.
    encoder_embed = Conv2dSubsampling(
        in_channels=params.feature_dim,
        out_channels=_to_int_tuple(params.encoder_dim)[0],
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
    )
    return encoder_embed


def get_encoder_model(params: AttributeDict) -> nn.Module:
    encoder = Zipformer2(
        output_downsampling_factor=2,
        downsampling_factor=_to_int_tuple(params.downsampling_factor),
        num_encoder_layers=_to_int_tuple(params.num_encoder_layers),
        encoder_dim=_to_int_tuple(params.encoder_dim),
        encoder_unmasked_dim=_to_int_tuple(params.encoder_unmasked_dim),
        query_head_dim=_to_int_tuple(params.query_head_dim),
        pos_head_dim=_to_int_tuple(params.pos_head_dim),
        value_head_dim=_to_int_tuple(params.value_head_dim),
        pos_dim=params.pos_dim,
        num_heads=_to_int_tuple(params.num_heads),
        feedforward_dim=_to_int_tuple(params.feedforward_dim),
        cnn_module_kernel=_to_int_tuple(params.cnn_module_kernel),
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        warmup_batches=4000.0,
        causal=params.causal,
        chunk_size=_to_int_tuple(params.chunk_size),
        left_context_frames=_to_int_tuple(params.left_context_frames),
    )
    return encoder


def get_decoder_model(params: AttributeDict) -> nn.Module:
    decoder = Decoder(
        vocab_size=params.vocab_size,
        decoder_dim=params.decoder_dim,
        blank_id=params.blank_id,
        context_size=params.context_size,
    )
    return decoder


def get_joiner_model(params: AttributeDict) -> nn.Module:
    joiner = Joiner(
        encoder_dim=max(_to_int_tuple(params.encoder_dim)),
        decoder_dim=params.decoder_dim,
        joiner_dim=params.joiner_dim,
        vocab_size=params.vocab_size,
    )
    return joiner


def get_attention_decoder_model(params: AttributeDict) -> nn.Module:
    decoder = AttentionDecoderModel(
        vocab_size=params.vocab_size,
        decoder_dim=params.attention_decoder_dim,
        num_decoder_layers=params.attention_decoder_num_layers,
        attention_dim=params.attention_decoder_attention_dim,
        num_heads=params.attention_decoder_num_heads,
        feedforward_dim=params.attention_decoder_feedforward_dim,
        memory_dim=max(_to_int_tuple(params.encoder_dim)),
        sos_id=params.sos_id,
        eos_id=params.eos_id,
        ignore_id=params.ignore_id,
        label_smoothing=params.label_smoothing,
    )
    return decoder


def load_encoder_from_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    ckpt_dict = checkpoint['model']
    model_dict = model.state_dict()

    encoder_dict = {k: v for k, v in ckpt_dict.items() if k.startswith('encoder')}
    model_encoder_dict = {k: v for k, v in model_dict.items() if k.startswith('encoder')}
    for k in encoder_dict:
        if k not in model_encoder_dict:
            raise KeyError(f"model lacking param: {k}")
        if encoder_dict[k].shape != model_encoder_dict[k].shape:
            raise ValueError(f"dimension not applicable: {k} ({encoder_dict[k].shape} vs {model_encoder_dict[k].shape})")

    model_dict.update(encoder_dict)
    model.load_state_dict(model_dict, strict=True)
    logging.info(f"loaded {len(encoder_dict)} params")

    for name, param_a in model.state_dict().items():
        if not name.startswith('encoder'):
            continue
        param_b = ckpt_dict[name]
        if not torch.all(param_a == param_b):
            logging.info(f'{name} not equal!')
            logging.info(f'A: {param_a}')
            logging.info(f'B: {param_b}')
            raise ValueError
    return model


def get_model(params: AttributeDict) -> nn.Module:
    assert params.use_transducer or params.use_ctc, (
        f"At least one of them should be True, "
        f"but got params.use_transducer={params.use_transducer}, "
        f"params.use_ctc={params.use_ctc}"
    )

    encoder_embed = get_encoder_embed(params)
    encoder = get_encoder_model(params)

    if params.use_transducer:
        decoder = get_decoder_model(params)
        joiner = get_joiner_model(params)
    else:
        decoder = None
        joiner = None

    if params.use_attention_decoder:
        attention_decoder = get_attention_decoder_model(params)
    else:
        attention_decoder = None

    model = AsrModel(
        encoder_embed=encoder_embed,
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        attention_decoder=attention_decoder,
        encoder_dim=max(_to_int_tuple(params.encoder_dim)),
        decoder_dim=params.decoder_dim,
        vocab_size=params.vocab_size,
        use_transducer=params.use_transducer,
        use_ctc=params.use_ctc,
        use_attention_decoder=params.use_attention_decoder,
        num_dialect_tokens=(
            dialect_tokens.num_tokens() + 1  # + neutral slot
            if getattr(params, "dialect_token", False)
            else 0
        ),
    )
    return model


def get_spec_augment(params: AttributeDict) -> SpecAugment:
    num_frame_masks = int(10 * params.time_mask_ratio)
    max_frames_mask_fraction = 0.15 * params.time_mask_ratio
    logging.info(
        f"num_frame_masks: {num_frame_masks}, "
        f"max_frames_mask_fraction: {max_frames_mask_fraction}"
    )
    spec_augment = SpecAugment(
        time_warp_factor=0,  # Do time warping in model.py
        num_frame_masks=num_frame_masks,  # default: 10
        features_mask_size=27,
        num_feature_masks=2,
        frames_mask_size=100,
        max_frames_mask_fraction=max_frames_mask_fraction,  # default: 0.15
    )
    return spec_augment


def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    model_avg: nn.Module = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
) -> Optional[Dict[str, Any]]:
    """Load checkpoint from file.

    If params.start_batch is positive, it will load the checkpoint from
    `params.exp_dir/checkpoint-{params.start_batch}.pt`. Otherwise, if
    params.start_epoch is larger than 1, it will load the checkpoint from
    `params.start_epoch - 1`.

    Apart from loading state dict for `model` and `optimizer` it also updates
    `best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
    and `best_valid_loss` in `params`.

    Args:
      params:
        The return value of :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer that we are using.
      scheduler:
        The scheduler that we are using.
    Returns:
      Return a dict containing previously saved training info.
    """
    if params.start_batch > 0:
        filename = params.exp_dir / f"checkpoint-{params.start_batch}.pt"
    elif params.start_epoch > 1:
        filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
    else:
        return None

    assert filename.is_file(), f"{filename} does not exist!"

    saved_params = load_checkpoint(
        filename,
        model=model,
        model_avg=model_avg,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    keys = [
        "best_train_epoch",
        "best_valid_epoch",
        "batch_idx_train",
        "best_train_loss",
        "best_valid_loss",
    ]
    for k in keys:
        params[k] = saved_params[k]

    if params.start_batch > 0:
        if "cur_epoch" in saved_params:
            params["start_epoch"] = saved_params["cur_epoch"]

    return saved_params


def save_checkpoint(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    model_avg: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    sampler: Optional[CutSampler] = None,
    scaler: Optional[GradScaler] = None,
    rank: int = 0,
) -> None:
    """Save model, optimizer, scheduler and training stats to file.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer used in the training.
      sampler:
       The sampler for the training dataset.
      scaler:
        The scaler used for mix precision training.
    """
    if rank != 0:
        return
    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
    save_checkpoint_impl(
        filename=filename,
        model=model,
        model_avg=model_avg,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        scaler=scaler,
        rank=rank,
    )

    if params.best_train_epoch == params.cur_epoch:
        best_train_filename = params.exp_dir / "best-train-loss.pt"
        copyfile(src=filename, dst=best_train_filename)

    if params.best_valid_epoch == params.cur_epoch:
        best_valid_filename = params.exp_dir / "best-valid-loss.pt"
        copyfile(src=filename, dst=best_valid_filename)


def compute_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    sp: spm.SentencePieceProcessor,
    batch: dict,
    is_training: bool,
    spec_augment: Optional[SpecAugment] = None,
    freeze_encoder: bool=False,
    encoder_grad_scale: float = 1,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute loss given the model and its inputs.

    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training. It is an instance of Zipformer in our case.
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      is_training:
        True for training. False for validation. When it is True, this
        function enables autograd during computation; when it is False, it
        disables autograd.
      spec_augment:
        The SpecAugment instance used only when use_cr_ctc is True.
    """
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    feature = batch["inputs"]
    # at entry, feature is (N, T, C)
    assert feature.ndim == 3
    feature = feature.to(device)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)
    # if not is_training and (feature_lens <= 0).any():
    assert torch.all(feature_lens > 0), (batch)
        # logging.info(batch)
        # exit

    batch_idx_train = params.batch_idx_train
    warm_step = params.warm_step

    texts = batch["supervisions"]["text"]
    if params.multi_bpe:
        cuts = batch["supervisions"]["cut"]
        y = [
            _encode_text_multi_bpe(t, _dialect_of_cut(c))
            for t, c in zip(texts, cuts)
        ]
    else:
        y = sp.encode(texts, out_type=int)
    y = k2.RaggedTensor(y)

    use_cr_ctc = params.use_cr_ctc
    use_spec_aug = use_cr_ctc and is_training
    if use_spec_aug:
        supervision_intervals = batch["supervisions"]
        supervision_segments = torch.stack(
            [
                supervision_intervals["sequence_idx"],
                supervision_intervals["start_frame"],
                supervision_intervals["num_frames"],
            ],
            dim=1,
        )  # shape: (S, 3)
    else:
        supervision_segments = None

    with torch.set_grad_enabled(is_training):
        simple_loss, pruned_loss, ctc_loss, attention_decoder_loss, cr_loss, _lid_ce = model(
            x=feature,
            x_lens=feature_lens,
            y=y,
            prune_range=params.prune_range,
            am_scale=params.am_scale,
            lm_scale=params.lm_scale,
            use_cr_ctc=use_cr_ctc,
            use_spec_aug=use_spec_aug,
            spec_augment=spec_augment,
            supervision_segments=supervision_segments,
            time_warp_factor=params.spec_aug_time_warp_factor,
            freeze_encoder=freeze_encoder,
            encoder_grad_scale=encoder_grad_scale,
        )

        loss = 0.0

        if params.use_transducer:
            s = params.simple_loss_scale
            # take down the scale on the simple loss from 1.0 at the start
            # to params.simple_loss scale by warm_step.
            simple_loss_scale = (
                s
                if batch_idx_train >= warm_step
                else 1.0 - (batch_idx_train / warm_step) * (1.0 - s)
            )
            pruned_loss_scale = (
                1.0
                if batch_idx_train >= warm_step
                else 0.1 + 0.9 * (batch_idx_train / warm_step)
            )
            loss += simple_loss_scale * simple_loss + pruned_loss_scale * pruned_loss

        if params.use_ctc:
            loss += params.ctc_loss_scale * ctc_loss
            if use_cr_ctc:
                loss += params.cr_loss_scale * cr_loss

        if params.use_attention_decoder:
            loss += params.attention_decoder_loss_scale * attention_decoder_loss

    assert loss.requires_grad == is_training

    info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info["frames"] = (feature_lens // params.subsampling_factor).sum().item()

    # Note: We use reduction=sum while computing the loss.
    info["loss"] = loss.detach().cpu().item()
    if params.use_transducer:
        info["simple_loss"] = simple_loss.detach().cpu().item()
        info["pruned_loss"] = pruned_loss.detach().cpu().item()
    if params.use_ctc:
        info["ctc_loss"] = ctc_loss.detach().cpu().item()
        if params.use_cr_ctc:
            info["cr_loss"] = cr_loss.detach().cpu().item()
    if params.use_attention_decoder:
        info["attn_decoder_loss"] = attention_decoder_loss.detach().cpu().item()

    return loss, info


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    sp: spm.SentencePieceProcessor,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process."""
    model.eval()

    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(valid_dl):
        loss, loss_info = compute_loss(
            params=params,
            model=model,
            sp=sp,
            batch=batch,
            is_training=False,
        )
        assert loss.requires_grad is False
        tot_loss = tot_loss + loss_info

    if world_size > 1:
        tot_loss.reduce(loss.device)

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return tot_loss


def train_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    optimizer: torch.optim.Optimizer,
    scheduler: LRSchedulerType,
    sp: spm.SentencePieceProcessor,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    scaler: GradScaler,
    spec_augment: Optional[SpecAugment] = None,
    model_avg: Optional[nn.Module] = None,
    writer: Optional[DualWriter] = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:
    """Train the model for one epoch.

    The training loss from the mean of all frames is saved in
    `params.train_loss`. It runs the validation process every
    `params.valid_interval` batches.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The model for training.
      optimizer:
        The optimizer we are using.
      scheduler:
        The learning rate scheduler, we call step() every step.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
      scaler:
        The scaler used for mix precision training.
      spec_augment:
        The SpecAugment instance used only when use_cr_ctc is True.
      model_avg:
        The stored model averaged from the start of training.
      writer:
        Writer to log metrics to tensorboard and/or wandb.
      world_size:
        Number of nodes in DDP training. If it is 1, DDP is disabled.
      rank:
        The rank of the node in DDP training. If no DDP is used, it should
        be set to 0.
    """
    model.train()

    tot_loss = MetricsTracker()

    saved_bad_model = False

    def save_bad_model(suffix: str = ""):
        save_checkpoint_impl(
            filename=params.exp_dir / f"bad-model{suffix}-{rank}.pt",
            model=model,
            model_avg=model_avg,
            params=params,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=0,
        )

    # Per-stage timing accumulators (seconds). Reset at each log interval.
    # data_time: waiting for the next batch (sampler + workers + feature I/O)
    # fwd_time:  compute_loss forward pass
    # bwd_time:  backward + optimizer step
    timing_prev = time.time()
    data_time = fwd_time = bwd_time = opt_time = 0.0
    timing_batches = 0
    # Throughput: cumulative encoder frames (loss_info["frames"]) and the
    # wall-clock anchor for the previous log line.
    cumulative_frames = 0
    last_log_time = time.time()
    last_log_frames = 0

    for batch_idx, batch in enumerate(train_dl):
        cur_time = time.time()
        data_time += cur_time - timing_prev

        if batch_idx % 10 == 0:
            set_batch_count(model, get_adjusted_batch_count(params))

        params.batch_idx_train += 1
        batch_size = len(batch["supervisions"]["text"])

        try:
            if params.warmup_encoder_step == 0:
                encoder_grad_scale = 1
            else:
                encoder_grad_scale = min(1, (params.batch_idx_train-params.freeze_encoder_step)/params.warmup_encoder_step)

            freeze_encoder = params.freeze_encoder or \
                    (params.cur_epoch < params.freeze_encoder_epoch) or \
                    (params.batch_idx_train < params.freeze_encoder_step)

            with torch.cuda.amp.autocast(
                enabled=params.use_autocast, dtype=params.dtype
            ):
                loss, loss_info = compute_loss(
                    params=params,
                    model=model,
                    sp=sp,
                    batch=batch,
                    is_training=True,
                    spec_augment=spec_augment,
                    freeze_encoder=freeze_encoder,
                    encoder_grad_scale=encoder_grad_scale
                )
            t_fwd = time.time()
            fwd_time += t_fwd - cur_time
            # summary stats
            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info
            cumulative_frames += loss_info["frames"]

            # NOTE: We use reduction==sum and loss is computed over utterances
            # in the batch and there is no normalization to it so far.
            scaler.scale(loss).backward()
            t_bwd = time.time()
            bwd_time += t_bwd - t_fwd

            scheduler.step_batch(params.batch_idx_train)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            opt_time += time.time() - t_bwd
            timing_prev = time.time()
            timing_batches += 1
        except Exception as e:
            logging.info(f"Caught exception: {e}.")
            save_bad_model()
            display_and_save_batch(batch, params=params, sp=sp)
            raise

        if params.print_diagnostics and batch_idx == 5:
            return

        if (
            rank == 0
            and params.batch_idx_train > 0
            and params.batch_idx_train % params.average_period == 0
        ):
            update_averaged_model(
                params=params,
                model_cur=model,
                model_avg=model_avg,
            )

        if (
            params.batch_idx_train > 0
            and params.batch_idx_train % params.save_every_n == 0
        ):
            save_checkpoint_with_global_batch_idx(
                out_dir=params.exp_dir,
                global_batch_idx=params.batch_idx_train,
                model=model,
                model_avg=model_avg,
                params=params,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=train_dl.sampler,
                scaler=scaler,
                rank=rank,
            )
            remove_checkpoints(
                out_dir=params.exp_dir,
                topk=params.keep_last_k,
                rank=rank,
            )

        if params.use_autocast:
            cur_grad_scale = scaler._scale.item()

            if cur_grad_scale < 0.01:
                if not saved_bad_model:
                    save_bad_model(suffix="-first-warning")
                    saved_bad_model = True
                    if not params.inf_check:
                        register_inf_check_hooks(model)
                logging.warning(f"Grad scale is small: {cur_grad_scale}")

            if cur_grad_scale < 1.0e-05:
                save_bad_model()
                raise_grad_scale_is_too_small_error(cur_grad_scale)

            # If the grad scale was less than 1, try increasing it.    The _growth_interval
            # of the grad scaler is configurable, but we can't configure it to have different
            # behavior depending on the current grad scale.
            if (
                batch_idx % 25 == 0
                and cur_grad_scale < 2.0
                or batch_idx % 100 == 0
                and cur_grad_scale < 8.0
                or batch_idx % 400 == 0
                and cur_grad_scale < 32.0
            ):
                scaler.update(cur_grad_scale * 2.0)

        if batch_idx % params.log_interval == 0:
            cur_lr = max(scheduler.get_last_lr())
            cur_grad_scale = scaler._scale.item() if params.use_autocast else 1.0

            if timing_batches > 0:
                n = timing_batches
                d = data_time / n
                f = fwd_time / n
                b = bwd_time / n
                o = opt_time / n
                total = d + f + b + o
                timing_str = (
                    f"timing per batch: data {d:.3f}s ({d/total*100:.0f}%), "
                    f"fwd {f:.3f}s ({f/total*100:.0f}%), "
                    f"bwd {b:.3f}s ({b/total*100:.0f}%), "
                    f"opt {o:.3f}s ({o/total*100:.0f}%)"
                )
            else:
                timing_str = "timing per batch: n/a"

            # Throughput: encoder frames per wall-clock second since the
            # previous log line, plus the equivalent audio duration
            # (feature rate 100Hz / subsampling_factor = encoder frame rate).
            cur_log_time = time.time()
            elapsed = cur_log_time - last_log_time
            fps = (
                (cumulative_frames - last_log_frames) / elapsed
                if elapsed > 0
                else 0.0
            )
            audio_sec_s = fps / (100.0 / params.subsampling_factor)
            last_log_time = cur_log_time
            last_log_frames = cumulative_frames

            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {batch_idx}, loss[{loss_info}], "
                f"tot_loss[{tot_loss}], batch size: {batch_size}, "
                f"fps: {fps:.0f}, audio-sec/s: {audio_sec_s:.0f}, "
                f"lr: {cur_lr:.2e}, "
                + (f"grad_scale: {scaler._scale.item()}, " if params.use_autocast else "")
                + timing_str
            )

            data_time = fwd_time = bwd_time = opt_time = 0.0
            timing_batches = 0

            if writer is not None:
                writer.add_scalar(
                    "train/learning_rate", cur_lr, params.batch_idx_train
                )

                loss_info.write_summary(
                    writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(writer, "train/tot_", params.batch_idx_train)
                if params.use_autocast:
                    writer.add_scalar(
                        "train/grad_scale", cur_grad_scale, params.batch_idx_train
                    )

        if batch_idx % params.valid_interval == 0 and not params.print_diagnostics:
            logging.info("Computing validation loss")
            valid_info = compute_validation_loss(
                params=params,
                model=model,
                sp=sp,
                valid_dl=valid_dl,
                world_size=world_size,
            )
            model.train()
            logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}")
            logging.info(
                f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
            )
            if writer is not None:
                valid_info.write_summary(
                    writer, "train/valid_", params.batch_idx_train
                )

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss


def run(rank, world_size, args):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = get_params()
    params.update(vars(args))

    fix_random_seed(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")

    if rank == 0:
        tb = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard") if args.tensorboard else None
        writer = DualWriter(
            tb_writer=tb,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            wandb_config=vars(args),
            wandb_dir=str(params.exp_dir),
        )
    else:
        writer = None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    logging.info(f"Device: {device}")

    if params.multi_bpe:
        sp = None
        load_multi_bpe(params)  # sets blank_id / sos_id / eos_id / vocab_size
    else:
        sp = spm.SentencePieceProcessor()
        sp.load(params.bpe_model)

        # <blk> is defined by scripts/build_tokenizer.py
        params.blank_id = sp.piece_to_id("<blk>")
        params.sos_id = params.eos_id = sp.piece_to_id("<sos/eos>")
        params.vocab_size = sp.get_piece_size()

    if not params.use_transducer:
        if not params.use_attention_decoder:
            params.ctc_loss_scale = 1.0
        else:
            assert params.ctc_loss_scale + params.attention_decoder_loss_scale == 1.0, (
                params.ctc_loss_scale,
                params.attention_decoder_loss_scale,
            )

    if params.use_bf16:  # amp + bf16
        assert torch.cuda.is_bf16_supported(), "Your GPU does not support bf16!"
        assert not params.use_fp16, "You can only use either fp16 or bf16"
        params.dtype = torch.bfloat16
        params.use_autocast = True
    elif params.use_fp16:  # amp + fp16
        params.dtype = torch.float16
        params.use_autocast = True
    else:  # fp32
        params.dtype = torch.float32
        params.use_autocast = False

    logging.info(f"Using dtype={params.dtype}")
    logging.info(f"Use AMP={params.use_autocast}")

    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)
    if params.encoder_path is not None:
        load_encoder_from_checkpoint(model, params.encoder_path)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    if params.use_cr_ctc:
        assert params.use_ctc
        assert not params.enable_spec_aug  # we will do spec_augment in model.py
        spec_augment = get_spec_augment(params)
    else:
        spec_augment = None

    assert params.save_every_n >= params.average_period
    model_avg: Optional[nn.Module] = None
    if rank == 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model).to(torch.float64)

    assert params.start_epoch > 0, params.start_epoch
    checkpoints = load_checkpoint_if_available(
        params=params, model=model, model_avg=model_avg
    )

    model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = ScaledAdam(
        get_parameter_groups_with_lrs(model, lr=params.base_lr, include_names=True),
        lr=params.base_lr,  # should have no effect
        clipping_scale=2.0,
    )

    if params.phase_ratio is not None:
        scheduler = TriStageLRSchedule(
            optimizer,
            final_lr_scale=0.05,
            max_update=params.max_lr_update,
            phase_ratio=eval(params.phase_ratio),
        )
    else:
        scheduler = Eden(optimizer, params.lr_batches, params.lr_epochs, warmup_start=0.1)

    if checkpoints and "optimizer" in checkpoints:
        logging.info("Loading optimizer state dict")
        optimizer.load_state_dict(checkpoints["optimizer"])

    if (
        checkpoints
        and "scheduler" in checkpoints
        and checkpoints["scheduler"] is not None
    ):
        logging.info("Loading scheduler state dict")
        scheduler.load_state_dict(checkpoints["scheduler"])

    if params.print_diagnostics:
        opts = diagnostics.TensorDiagnosticOptions(
            512
        )  # allow 4 megabytes per sub-module
        diagnostic = diagnostics.attach_diagnostics(model, opts)

    if params.inf_check:
        register_inf_check_hooks(model)

    dataoceanai_arabic = ArabicAsrDataModule(args)

    if args.cuts_name is not None:
        cuts_path = args.manifest_dir / f"{args.cuts_name}.jsonl.gz"
        logging.info(f"Loading training cuts from {cuts_path}")
        train_cuts = load_manifest_lazy(cuts_path)
        _min_dur = getattr(args, "min_cut_duration", 1.0)
        _max_dur = getattr(args, "max_cut_duration", 50.0)
        train_cuts = train_cuts.filter(
            lambda c, lo=_min_dur, hi=_max_dur: lo <= c.duration <= hi
        )
        if args.dialect == "ENG":
            train_cuts = dataoceanai_arabic._uppercase_text(train_cuts)
        train_cuts = dataoceanai_arabic._apply_recipe_textnorm(train_cuts)
    elif args.train_with_pseudo_label:
        train_cuts = dataoceanai_arabic.semi_supervised_with_pseudo_label_train_cuts()
    elif args.transfer_learning:
        train_cuts = dataoceanai_arabic.transfer_learning_filtered_train_cuts()
    elif args.train_with_tts_data:
        train_cuts = dataoceanai_arabic.tts_train_cuts()
    else:
        train_cuts = dataoceanai_arabic.train_cuts()

    # TRAIN_WITH_MSA: standalone MSA append, for the non-pseudo-label paths.
    # With --train-with-pseudo-label the datamodule already folds MSA into the
    # interleaved stream (tagged custom['dialect']='MSA'), so adding it here
    # would duplicate it.
    if args.train_with_msa and not args.train_with_pseudo_label:
        msa_path = args.manifest_dir / "gigaspeech3_msa_cuts_train.jsonl.gz"
        textnormed_msa_path = args.manifest_dir / "gigaspeech3_msa_cuts_train_textnormed.jsonl.gz"
        msa_cuts = None
        if textnormed_msa_path.is_file():
            # Pre-textnormed manifest (see scripts/preprocess_msa_textnorm.py):
            # textnorm already applied + empty-text cuts dropped, so we skip
            # the per-epoch lazy textnorm entirely.
            logging.info(f"Loading pre-textnormed MSA train cuts from {textnormed_msa_path}")
            msa_cuts = load_manifest_lazy(textnormed_msa_path)
        elif msa_path.is_file():
            logging.info(f"Loading MSA train cuts from {msa_path}")
            msa_cuts = load_manifest_lazy(msa_path)
            # Tag dialect so _apply_recipe_textnorm picks the MSA normalizer
            # (MSA manifests carry no custom['dialect'] tag by default).
            msa_cuts = msa_cuts.map(
                lambda c: dataoceanai_arabic._tag_dialect(c, "MSA")
            )
            msa_cuts = dataoceanai_arabic._apply_recipe_textnorm(msa_cuts)
        else:
            logging.warning(f"MSA train cuts not found at {msa_path}")

        if msa_cuts is not None:
            # Duration filter applied early so describe reflects filtered data
            _min_d = getattr(args, "min_cut_duration", 1.0)
            _max_d = getattr(args, "max_cut_duration", 50.0)
            msa_cuts = msa_cuts.filter(
                lambda c, lo=_min_d, hi=_max_d: lo <= c.duration <= hi
            )
            if args.log_describe:
                buf = io.StringIO()
                try:
                    stdout = sys.stdout
                    sys.stdout = buf
                    msa_cuts.describe()
                    sys.stdout = stdout
                    logging.info(f"MSA train cuts describe (duration [{_min_d},{_max_d}]):\n{buf.getvalue().rstrip()}")
                finally:
                    sys.stdout = sys.__stdout__
            # Interleave MSA into the dialect stream (appending would create
            # pure-MSA windows — see interleave_cutsets for the rationale)
            train_cuts = interleave_cutsets([train_cuts, msa_cuts])
            logging.info("MSA cuts interleaved into training set")

    min_dur = getattr(args, "min_cut_duration", 1.0)
    max_dur = getattr(args, "max_cut_duration", 50.0)
    def remove_short_and_long_utt(c: Cut):
        if c.duration < min_dur or c.duration > max_dur:
            # Filter short or over long sentences
            return False

        if c.supervisions[0].text.strip() == "":
            # For pseudo labeled sentences, we filter out those are empty
            return False

        # if isinstance(c.supervisions[0].language, float):
        #     lang_score = c.supervisions[0].language
        #     if lang_score <= args.pseudo_label_score_min:
        #         return False
        #     if lang_score >= args.pseudo_label_score_max:
        #         return False

        T = ((c.num_frames - 7) // 2 + 1) // 2
        if args.multi_bpe:
            tokens = _encode_text_multi_bpe(
                c.supervisions[0].text, _dialect_of_cut(c)
            )
        else:
            tokens = sp.encode(c.supervisions[0].text, out_type=str)

        if T < len(tokens):
            # logging.warning(
            #     f"Exclude cut with ID {c.id} from training. "
            #     f"Number of frames (before subsampling): {c.num_frames}. "
            #     f"Number of frames (after subsampling): {T}. "
            #     f"Text: {c.supervisions[0].text}. "
            #     f"Tokens: {tokens}. "
            #     f"Number of tokens: {len(tokens)}"
            # )
            return False

        return True

    # train_cuts = train_cuts.filter(remove_short_and_long_utt)

    # Final train-cuts describe, gated by --log-describe (LOG_DESCRIBE=1).
    # This is the only describe point for the --cuts-name path and covers the
    # full assembled training set (incl. MSA).  Skipped for the pseudo-label
    # path: the datamodule already logs per-dialect describes, and iterating
    # 15M+ cuts on every rank would hang startup.
    if getattr(args, "log_describe", False) and not args.train_with_pseudo_label:
        _log_describe(train_cuts, "Final train cuts describe:")

    if params.start_batch > 0 and checkpoints and "sampler" in checkpoints:
        # We only load the sampler's state dict when it loads a checkpoint
        # saved in the middle of an epoch
        sampler_state_dict = checkpoints["sampler"]
    else:
        sampler_state_dict = None

    train_dl = dataoceanai_arabic.train_dataloaders(
        train_cuts,
        sampler_state_dict=sampler_state_dict,
        world_size=world_size,
        rank=rank,
    )

    valid_cuts = dataoceanai_arabic.dev_cuts()
    valid_cuts = valid_cuts.filter(remove_short_and_long_utt)
    valid_dl = dataoceanai_arabic.valid_dataloaders(
        valid_cuts,
        world_size=world_size,
        rank=rank,
    )

    if False and not params.print_diagnostics:
        scan_pessimistic_batches_for_oom(
            model=model,
            train_dl=train_dl,
            optimizer=optimizer,
            sp=sp,
            params=params,
            spec_augment=spec_augment,
        )

    scaler = GradScaler(enabled=params.use_autocast, init_scale=1.0)
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        scheduler.step_epoch(epoch - 1)
        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)

        if writer is not None:
            writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        params.cur_epoch = epoch

        train_one_epoch(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sp=sp,
            train_dl=train_dl,
            valid_dl=valid_dl,
            scaler=scaler,
            spec_augment=spec_augment,
            writer=writer,
            world_size=world_size,
            rank=rank,
        )

        if params.print_diagnostics:
            diagnostic.print_diagnostics()
            break

        save_checkpoint(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=rank,
        )

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def display_and_save_batch(
    batch: dict,
    params: AttributeDict,
    sp: spm.SentencePieceProcessor,
) -> None:
    """Display the batch statistics and save the batch into disk.

    Args:
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      params:
        Parameters for training. See :func:`get_params`.
      sp:
        The SentencePiece tokenizer.
    """
    from lhotse.utils import uuid4

    filename = f"{params.exp_dir}/batch-{uuid4()}.pt"
    logging.info(f"Saving batch to {filename}")
    torch.save(batch, filename)

    supervisions = batch["supervisions"]
    features = batch["inputs"]

    logging.info(f"features shape: {features.shape}")

    if getattr(params, "multi_bpe", False):
        y = [
            _encode_text_multi_bpe(t, _dialect_of_cut(c))
            for t, c in zip(supervisions["text"], supervisions["cut"])
        ]
    else:
        y = sp.encode(supervisions["text"], out_type=int)
    num_tokens = sum(len(i) for i in y)
    logging.info(f"num tokens: {num_tokens}")


def scan_pessimistic_batches_for_oom(
    model: Union[nn.Module, DDP],
    train_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    sp: spm.SentencePieceProcessor,
    params: AttributeDict,
    spec_augment: Optional[SpecAugment] = None,
):
    from lhotse.dataset import find_pessimistic_batches

    logging.info(
        "Sanity check -- see if any of the batches in epoch 1 would cause OOM."
    )
    batches, crit_values = find_pessimistic_batches(train_dl.sampler)
    for criterion, cuts in batches.items():
        batch = train_dl.dataset[cuts]
        try:
            with torch.cuda.amp.autocast(
                enabled=params.use_autocast, dtype=params.dtype
            ):
                loss, _ = compute_loss(
                    params=params,
                    model=model,
                    sp=sp,
                    batch=batch,
                    is_training=True,
                    spec_augment=spec_augment,
                )
            loss.backward()
            optimizer.zero_grad()
        except Exception as e:
            if "CUDA out of memory" in str(e):
                logging.error(
                    "Your GPU ran out of memory with the current "
                    "max_duration setting. We recommend decreasing "
                    "max_duration and trying again.\n"
                    f"Failing criterion: {criterion} "
                    f"(={crit_values[criterion]}) ..."
                )
            display_and_save_batch(batch, params=params, sp=sp)
            raise
        logging.info(
            f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
        )


def main():
    parser = get_parser()
    parser.add_argument("--train-with-pseudo-label", type=str2bool, default=False)
    parser.add_argument("--train-with-msa", type=str2bool, default=False,
                        help="Include MSA training data in ARAB multi-dialect training.")
    parser.add_argument("--pseudo-label-score-min", type=float, default=7.0,
                        help="Min language score to keep pseudo-labeled cuts.")
    parser.add_argument("--pseudo-label-score-max", type=float, default=16.0,
                        help="Max language score to keep pseudo-labeled cuts.")
    parser.add_argument("--pseudo-label-score-dialects", type=str, default=None,
                        help="Per-dialect score thresholds as JSON, "
                             "e.g. '{\"egypt\":[10,16],\"saudi\":[8,18]}'")
    parser.add_argument("--min-cut-duration", type=float, default=1.0,
                        help="Minimum cut duration in seconds (default: 1.0).")
    parser.add_argument("--max-cut-duration", type=float, default=50.0,
                        help="Maximum cut duration in seconds (default: 50.0).")
    parser.add_argument("--transfer-learning", type=str2bool, default=False,
                        help="Use transfer learning data from other labeled datasets.")
    parser.add_argument("--train-with-tts-data", type=str2bool, default=False,
                        help="Use TTS data for training.")
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument(
        "--cuts-name",
        type=str,
        default=None,
        help="Custom train cuts filename (without .jsonl.gz). Loaded from manifest-dir.",
    )
    ArabicAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    main()

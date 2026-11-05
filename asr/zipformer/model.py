# Copyright    2021-2024  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Zengwei Yao,
#                                                       Yifan Yang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
#
# MERGED from:
#   - vietasr-saudi/ASR/zipformer/model.py     (base: encoder_embed + encoder, SpecAugment, CR-CTC, AED)
#   - vietasr-saudi/SSL/zipformer_fbank/model.py (HuBERT: single encoder, encoder_feature_layer, raw audio input)
#
# Supports two forward-encoder paths:
#   Stage 1 (supervised ASR):     encoder_embed + encoder
#   Stage 2 (SSL/HuBERT finetune): single encoder (HuBERT wrapper), raw audio input

from typing import List, Optional, Tuple

import k2
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder_interface import EncoderInterface
from lhotse.dataset import SpecAugment
from scaling import ScaledLinear
from utils import GradMultiply

from icefall.utils import add_sos, make_pad_mask, time_warp


class AsrModel(nn.Module):
    def __init__(
        self,
        encoder_embed: Optional[nn.Module] = None,
        encoder: Optional[EncoderInterface] = None,
        decoder: Optional[nn.Module] = None,
        joiner: Optional[nn.Module] = None,
        attention_decoder: Optional[nn.Module] = None,
        encoder_dim: int = 384,
        decoder_dim: int = 512,
        vocab_size: int = 500,
        use_transducer: bool = True,
        use_ctc: bool = False,
        use_attention_decoder: bool = False,
        encoder_feature_layer: int = -1,
        num_dialect_tokens: int = 0,
        lid_class_weight: Optional[List[float]] = None,
    ):
        """A joint CTC & Transducer ASR model.

        Two encoder modes:
          - Stage 1 (supervised): provide both ``encoder_embed`` and ``encoder``.
            ``encoder_embed`` is a Conv2d subsampling module converting
            (N, T, idim) → (N, T', odim). ``encoder`` receives (T, N, C)
            and returns (T, N, C) + lens.
          - Stage 2 (SSL finetune): provide ``encoder`` only (a HuBERT wrapper
            with ``extract_features(source, padding_mask, ...)``).  Raw audio
            (N, T) goes directly into ``encoder.extract_features()``.

        Args:
          encoder_embed:
            Conv2d subsampling module (Stage 1 only).
          encoder:
            Transcription network (Zipformer2 or HuBERT wrapper).
          decoder:
            Prediction network. Must have ``blank_id`` attribute.
          joiner:
            Joiner network.
          attention_decoder:
            Attention decoder head (optional).
          encoder_dim:
            Encoder output dimension.
          decoder_dim:
            Decoder output dimension.
          vocab_size:
            Vocabulary size.
          use_transducer:
            Enable transducer loss.
          use_ctc:
            Enable CTC loss.
          use_attention_decoder:
            Enable attention-decoder head.
          encoder_feature_layer:
            HuBERT feature layer index (Stage 2 only, default -1 = last).
        """
        super().__init__()

        assert (
            use_transducer or use_ctc
        ), f"At least one of them should be True, but got use_transducer={use_transducer}, use_ctc={use_ctc}"

        assert encoder is not None

        # Stage 1 has encoder_embed; Stage 2 (HuBERT) does not
        self.encoder_embed = encoder_embed
        self.encoder = encoder
        self.encoder_feature_layer = encoder_feature_layer

        self.use_transducer = use_transducer
        if use_transducer:
            assert decoder is not None
            assert hasattr(decoder, "blank_id")
            assert joiner is not None

            self.decoder = decoder
            self.joiner = joiner

            self.simple_am_proj = ScaledLinear(
                encoder_dim, vocab_size, initial_scale=0.25
            )
            self.simple_lm_proj = ScaledLinear(
                decoder_dim, vocab_size, initial_scale=0.25
            )
        else:
            assert decoder is None
            assert joiner is None

        self.use_ctc = use_ctc
        if use_ctc:
            self.ctc_output = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(encoder_dim, vocab_size),
                nn.LogSoftmax(dim=-1),
            )

        self.use_attention_decoder = use_attention_decoder
        if use_attention_decoder:
            self.attention_decoder = attention_decoder
        else:
            assert attention_decoder is None

        # Dialect conditioning (simple variant): a per-dialect embedding
        # added to the encoder output, plus an auxiliary LID classification
        # head.  num_dialect_tokens includes one NEUTRAL slot at index
        # num_dialect_tokens-1 (used for unknown dialects and un-conditioned
        # decoding).  0 disables both modules (backward compatible).
        self.num_dialect_tokens = num_dialect_tokens
        if num_dialect_tokens > 0:
            self.dialect_emb = nn.Embedding(num_dialect_tokens, encoder_dim)
            self.lid_head = nn.Linear(encoder_dim, num_dialect_tokens - 1)
        else:
            self.dialect_emb = None
            self.lid_head = None

        # Optional per-class weights for the LID cross-entropy, in dialect
        # index order (len == num_dialect_tokens-1).  Registered as a
        # NON-persistent buffer: it follows .to(device) but is deliberately
        # left out of state_dict, because
        #   * decode never uses it (it only scales the training CE), so
        #     shipping it makes checkpoints undecodable by builds that do not
        #     construct the model with this kwarg, and
        #   * keeping it out means resuming from a checkpoint cannot silently
        #     override the weights given on the command line.
        if num_dialect_tokens > 0 and lid_class_weight is not None:
            w = torch.tensor(lid_class_weight, dtype=torch.float32)
            if w.numel() != num_dialect_tokens - 1:
                raise ValueError(
                    f"lid_class_weight has {w.numel()} entries but the head has "
                    f"{num_dialect_tokens - 1} classes"
                )
            self.register_buffer("lid_class_weight", w, persistent=False)
        else:
            self.register_buffer("lid_class_weight", None, persistent=False)
        self.last_lid_logits = None

    def forward_encoder(
        self,
        x: torch.Tensor,
        x_lens: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        do_final_down_sample: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute encoder outputs.

        Two paths:
          - ASR (Stage 1): encoder_embed (Conv2d subsampling) + encoder.forward()
          - SSL finetune (Stage 2): encoder.extract_features() (HuBERT wrapper)

        Args:
          x:
            ASR mode: (N, T, C) fbank features.
            SSL mode: (N, T) raw audio or fbank.
          x_lens:
            Frame lengths (N,).  Required for ASR mode.
          padding_mask:
            Optional padding mask (N, T).  Used in SSL mode.
          do_final_down_sample:
            Whether to apply final downsampling (SSL mode).

        Returns:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
        """
        # ── Stage 1: ASR path (encoder_embed + encoder) ──
        if self.encoder_embed is not None:
            assert x_lens is not None, "x_lens required for ASR forward_encoder"
            x, x_lens = self.encoder_embed(x, x_lens)
            # Zipformer2 expects (T, N, C) format
            src_key_padding_mask = make_pad_mask(x_lens, max_len=x.size(1))
            x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
            encoder_out, encoder_out_lens, _ = self.encoder(x, x_lens, src_key_padding_mask)
            # encoder_out is (T, N, C) -> (N, T, C)
            encoder_out = encoder_out.permute(1, 0, 2)
            return encoder_out, encoder_out_lens

        # ── Stage 2: SSL finetune path (HuBERT extract_features) ──
        if padding_mask is not None and padding_mask.dim() == 1:
            padding_mask = make_pad_mask(padding_mask)
        if padding_mask is None and x_lens is not None:
            # Caller passed feature lengths; convert to 2-D padding mask
            padding_mask = make_pad_mask(x_lens, max_len=x.size(1))

        encoder_out, padding_mask, encoder_out_lens = self.encoder.extract_features(
            source=x,
            padding_mask=padding_mask,
            mask=self.encoder.training,
            output_layer=self.encoder_feature_layer,
            do_final_down_sample=do_final_down_sample,
        )
        assert torch.all(encoder_out_lens > 0), encoder_out_lens

        return encoder_out, encoder_out_lens

    def forward_ctc(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTC loss."""
        ctc_output = self.ctc_output(encoder_out)  # (N, T, C)

        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=ctc_output.permute(1, 0, 2),  # (T, N, C)
            targets=targets.cpu(),
            input_lengths=encoder_out_lens.cpu(),
            target_lengths=target_lengths.cpu(),
            reduction="sum",
        )
        return ctc_loss

    def forward_cr_ctc(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute CTC loss with consistency regularization loss."""
        ctc_output = self.ctc_output(encoder_out)  # (2*N, T, C)
        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=ctc_output.permute(1, 0, 2),
            targets=targets.cpu(),
            input_lengths=encoder_out_lens.cpu(),
            target_lengths=target_lengths.cpu(),
            reduction="sum",
        )

        exchanged_targets = ctc_output.detach().chunk(2, dim=0)
        exchanged_targets = torch.cat(
            [exchanged_targets[1], exchanged_targets[0]], dim=0
        )
        cr_loss = nn.functional.kl_div(
            input=ctc_output,
            target=exchanged_targets,
            reduction="none",
            log_target=True,
        )
        length_mask = make_pad_mask(encoder_out_lens).unsqueeze(-1)
        cr_loss = cr_loss.masked_fill(length_mask, 0.0).sum()

        return ctc_loss, cr_loss

    def forward_transducer(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        y: k2.RaggedTensor,
        y_lens: torch.Tensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Transducer loss."""
        blank_id = self.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)
        decoder_out = self.decoder(sos_y_padded)

        y_padded = y.pad(mode="constant", padding_value=0)
        y_padded = y_padded.to(torch.int64)
        boundary = torch.zeros(
            (encoder_out.size(0), 4),
            dtype=torch.int64,
            device=encoder_out.device,
        )
        boundary[:, 2] = y_lens
        boundary[:, 3] = encoder_out_lens

        lm = self.simple_lm_proj(decoder_out)
        am = self.simple_am_proj(encoder_out)

        with torch.cuda.amp.autocast(enabled=False):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=lm.float(),
                am=am.float(),
                symbols=y_padded,
                termination_symbol=blank_id,
                lm_only_scale=lm_scale,
                am_only_scale=am_scale,
                boundary=boundary,
                reduction="sum",
                return_grad=True,
            )

        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=prune_range,
        )

        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joiner.encoder_proj(encoder_out),
            lm=self.joiner.decoder_proj(decoder_out),
            ranges=ranges,
        )

        logits = self.joiner(am_pruned, lm_pruned, project_input=False)

        with torch.cuda.amp.autocast(enabled=False):
            pruned_loss = k2.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=y_padded,
                ranges=ranges,
                termination_symbol=blank_id,
                boundary=boundary,
                reduction="sum",
            )

        return simple_loss, pruned_loss

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        use_cr_ctc: bool = False,
        use_spec_aug: bool = False,
        spec_augment: Optional[SpecAugment] = None,
        supervision_segments: Optional[torch.Tensor] = None,
        time_warp_factor: Optional[int] = 80,
        freeze_encoder: bool = False,
        encoder_grad_scale: float = 1,
        dialect_ids: Optional[torch.Tensor] = None,
        lid_warm_frames: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            Stage 1: (N, T, C) fbank.  Stage 2: (N, T) raw audio.
          x_lens:
            (N,) lengths before padding.
          y:
            Ragged tensor [utt][label].
          prune_range:
            Prune range for RNN-T loss.
          am_scale / lm_scale:
            Smoothing scales for transducer loss.
          use_cr_ctc:
            Enable consistency-regularized CTC.
          use_spec_aug:
            Enable SpecAugment manually (CR-CTC mode).
          spec_augment:
            SpecAugment instance (CR-CTC mode).
          supervision_segments:
            (S, 3) tensor for time warping (CR-CTC mode).
          time_warp_factor:
            Time warping factor (CR-CTC mode).
          freeze_encoder:
            Freeze encoder weights during forward.
          encoder_grad_scale:
            Gradient scale factor on encoder output.

        Returns:
          (simple_loss, pruned_loss, ctc_loss, attention_decoder_loss,
           cr_loss, lid_ce) — lid_ce is the auxiliary LID classification
          loss (sum, empty tensor when dialect conditioning is disabled).
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert torch.all(x_lens > 0), x_lens
        assert y.num_axes == 2, y.num_axes
        assert x.size(0) == x_lens.size(0) == y.dim0, (x.shape, x_lens.shape, y.dim0)

        device = x.device

        if use_cr_ctc:
            assert self.use_ctc
            if use_spec_aug:
                assert spec_augment is not None and spec_augment.time_warp_factor < 1
                assert supervision_segments is not None
                x = time_warp(
                    x,
                    time_warp_factor=time_warp_factor,
                    supervision_segments=supervision_segments,
                )
                x = spec_augment(x.repeat(2, 1, 1))
            else:
                x = x.repeat(2, 1, 1)
            x_lens = x_lens.repeat(2)
            y = k2.ragged.cat([y, y], axis=0)
            if dialect_ids is not None:
                dialect_ids = dialect_ids.repeat(2)

        if freeze_encoder:
            with torch.no_grad():
                encoder_out, encoder_out_lens = self.forward_encoder(x, x_lens)
        else:
            encoder_out, encoder_out_lens = self.forward_encoder(x, x_lens)
            if encoder_grad_scale != 1:
                GradMultiply.apply(encoder_out, encoder_grad_scale)

        # Dialect conditioning (simple variant): the LID head runs on the
        # pure acoustic encoder output, and its CE is computed HERE (inside
        # the DDP forward graph) and returned, so the reducer sees the
        # lid_head parameters.  The dialect embedding is added for frames
        # >= lid_warm_frames so the first-token LID prediction stays
        # acoustic-only (no trivial "echo" shortcut).
        lid_ce = torch.empty(0, device=device)
        self.last_lid_logits = None
        if self.num_dialect_tokens > 0:
            # L2-normalize the pooled features so the head's logits stay
            # bounded (prevents weight blow-up on trained encoder features);
            # compute in fp32 — fp16 logits saturate and underflow the CE.
            with torch.cuda.amp.autocast(enabled=False):
                h = encoder_out.mean(dim=1).float()
                h = F.normalize(h, dim=-1, eps=1e-5)
                lid_logits = self.lid_head(h).float()
            self.last_lid_logits = lid_logits
            if dialect_ids is not None:
                mask = dialect_ids < self.num_dialect_tokens - 1
                if mask.any():
                    lid_ce = F.cross_entropy(
                        lid_logits[mask],
                        dialect_ids[mask],
                        weight=self.lid_class_weight,
                        reduction="sum",
                    )
                B, T, _ = encoder_out.shape
                neutral = self.num_dialect_tokens - 1
                emb_ids = torch.full(
                    (B, T), neutral, dtype=torch.long, device=device
                )
                if lid_warm_frames > 0 and T > lid_warm_frames:
                    emb_ids[:, lid_warm_frames:] = dialect_ids.unsqueeze(1)
                else:
                    emb_ids[:] = dialect_ids.unsqueeze(1)
                encoder_out = encoder_out + self.dialect_emb(emb_ids)

        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        if self.use_transducer:
            simple_loss, pruned_loss = self.forward_transducer(
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                y=y.to(device),
                y_lens=y_lens,
                prune_range=prune_range,
                am_scale=am_scale,
                lm_scale=lm_scale,
            )
            if use_cr_ctc:
                simple_loss = simple_loss * 0.5
                pruned_loss = pruned_loss * 0.5
        else:
            simple_loss = torch.empty(0)
            pruned_loss = torch.empty(0)

        if self.use_ctc:
            targets = y.values
            if not use_cr_ctc:
                ctc_loss = self.forward_ctc(
                    encoder_out=encoder_out,
                    encoder_out_lens=encoder_out_lens,
                    targets=targets,
                    target_lengths=y_lens,
                )
                cr_loss = torch.empty(0)
            else:
                ctc_loss, cr_loss = self.forward_cr_ctc(
                    encoder_out=encoder_out,
                    encoder_out_lens=encoder_out_lens,
                    targets=targets,
                    target_lengths=y_lens,
                )
                ctc_loss = ctc_loss * 0.5
                cr_loss = cr_loss * 0.5
        else:
            ctc_loss = torch.empty(0)
            cr_loss = torch.empty(0)

        if self.use_attention_decoder:
            attention_decoder_loss = self.attention_decoder.calc_att_loss(
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                ys=y.to(device),
                ys_lens=y_lens.to(device),
            )
            if use_cr_ctc:
                attention_decoder_loss = attention_decoder_loss * 0.5
        else:
            attention_decoder_loss = torch.empty(0)

        return (
            simple_loss,
            pruned_loss,
            ctc_loss,
            attention_decoder_loss,
            cr_loss,
            lid_ce,
        )

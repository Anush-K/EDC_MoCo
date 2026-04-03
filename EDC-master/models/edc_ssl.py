"""
models/edc_ssl.py

SSL-EDC model: MoCo-pretrained encoder + randomly-initialised decoder.
Drop-in replacement for models.edc.R50_R50.

CHANGE vs original: decoder warmup with gradual encoder unfreezing.
------------------------------------------------------------------
Problem (decoder cold-start):
    When a MoCo-pretrained encoder is paired with a randomly-initialised
    decoder, the decoder's gradients are very large in early iterations
    because its weights are random. In fine-tune mode, these large gradients
    flow back into the encoder and can corrupt the pretrained features before
    the decoder has had a chance to stabilise.

Solution (warmup_iters):
    For the first `warmup_iters` iterations the encoder is treated as fully
    frozen regardless of the `freeze_encoder` flag. The decoder trains freely
    and stabilises its weights against the fixed encoder features.
    After `warmup_iters`, the encoder unfreezes and begins fine-tuning at
    the low learning rate already configured in the runner (lr_encoder=1e-5).

    If freeze_encoder=True (linear-probe / SSL-Frozen condition), the encoder
    stays frozen for the entire run — warmup_iters has no effect in that case.

How to use:
    The runner creates the model normally. The training loop (edc1.py) calls
    model.set_iter(current_iteration) once per iteration before the forward
    pass. That is the only change needed in edc1.py.

    model = MoCo_R50_R50(
        moco_weights_path=args.moco_weights_path,
        freeze_encoder=args.freeze_encoder,
        warmup_iters=args.warmup_iters,   # new arg, default 200
    )

    # In training loop:
    model.set_iter(self.it)    # ← one new line in edc1.py
    result = model(x)

Forward output dict is IDENTICAL to R50_R50 and the original MoCo_R50_R50:
    loss, p_all, p1, p2, p3, e1_std, e2_std, e3_std
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet_moco    import MoCoResNet50Encoder
from models.resnet_decoder import resnet50_decoder


class MoCo_R50_R50(nn.Module):
    """
    SSL-EDC: MoCo-pretrained ResNet-50 encoder + randomly-initialised
    ResNet-50 decoder.

    Parameters
    ----------
    moco_weights_path : str
        Path to MoCo .pth file  {"encoder_q": state_dict}.
    img_size : int
        Input spatial size (default 256).
    train_encoder : bool
        If False, encoder is kept frozen throughout training.
        Default True → fine-tune mode.
    stop_grad : bool
        Detach encoder feature maps before cosine-loss computation.
    reshape : bool
        Use flattened cosine similarity for loss.
    bn_pretrain : bool
        Force encoder BatchNorm into eval() mode.
    anomap_layer : list[int]
        Which decoder levels contribute to the anomaly map [1,2,3].
    freeze_encoder : bool
        Hard-freeze encoder parameters (requires_grad=False) for the whole
        run. Used for SSL-Frozen / linear-probe condition.
        Overrides warmup behaviour — if True, encoder stays frozen always.
    warmup_iters : int
        Number of iterations to keep the encoder frozen at the start of
        fine-tuning, giving the decoder time to stabilise before the encoder
        begins updating. Default 200.
        Has no effect when freeze_encoder=True.
    """

    def __init__(
        self,
        moco_weights_path: str,
        img_size: int        = 256,
        train_encoder: bool  = True,
        stop_grad: bool      = True,
        reshape: bool        = True,
        bn_pretrain: bool    = False,
        anomap_layer         = None,
        freeze_encoder: bool = False,
        warmup_iters: int    = 200,
    ):
        super().__init__()

        if anomap_layer is None:
            anomap_layer = [1, 2, 3]

        # ── Encoder (MoCo SSL pretrained) ─────────────────────────────────
        # Pass freeze_encoder=True only for the permanent frozen condition.
        # Warmup-based freezing is handled separately below via _encoder_frozen.
        self.edc_encoder = MoCoResNet50Encoder(
            moco_weights_path=moco_weights_path,
            freeze_encoder=freeze_encoder,
        )

        # ── Decoder (randomly initialised) ────────────────────────────────
        self.edc_decoder = resnet50_decoder(pretrained=False, inplanes=[2048])

        # ── Hyperparameters ───────────────────────────────────────────────
        self.train_encoder   = train_encoder
        self.stop_grad       = stop_grad
        self.reshape         = reshape
        self.bn_pretrain     = bn_pretrain
        self.anomap_layer    = anomap_layer
        self.freeze_encoder  = freeze_encoder   # permanent freeze flag
        self.warmup_iters    = warmup_iters

        # ── Internal state ────────────────────────────────────────────────
        self._current_iter   = 0
        self._encoder_frozen = False            # tracks warmup freeze state

        # If fine-tune mode (not permanently frozen), start with encoder
        # frozen for warmup. The set_iter() call will unfreeze it later.
        if not freeze_encoder and train_encoder and warmup_iters > 0:
            self._apply_encoder_freeze(frozen=True)
            print(f"[MoCo_R50_R50] Decoder warmup active: encoder frozen for "
                  f"first {warmup_iters} iterations.")
        elif freeze_encoder:
            print("[MoCo_R50_R50] Encoder permanently frozen (SSL-Frozen mode).")
        else:
            print("[MoCo_R50_R50] No warmup — encoder trainable from iter 0.")

    # ------------------------------------------------------------------
    # Warmup management
    # ------------------------------------------------------------------

    def _apply_encoder_freeze(self, frozen: bool):
        """Toggle requires_grad on all encoder parameters."""
        for param in self.edc_encoder.parameters():
            param.requires_grad = not frozen
        self._encoder_frozen = frozen

    def set_iter(self, current_iter: int):
        """
        Called by the training loop at the start of each iteration.
        Handles the warmup → unfreeze transition.

        Only acts when:
        - freeze_encoder is False  (not permanent freeze mode)
        - train_encoder is True    (fine-tune mode, not eval-only)
        - warmup_iters > 0
        """
        self._current_iter = current_iter

        if self.freeze_encoder or not self.train_encoder or self.warmup_iters <= 0:
            return

        if current_iter >= self.warmup_iters and self._encoder_frozen:
            # Warmup complete — unfreeze encoder
            self._apply_encoder_freeze(frozen=False)
            print(f"\n[MoCo_R50_R50] Warmup complete at iter {current_iter}. "
                  f"Encoder unfrozen — fine-tuning begins.\n")

    # ------------------------------------------------------------------
    # Forward — identical output dict to R50_R50 and original MoCo_R50_R50
    # ------------------------------------------------------------------

    def forward(self, x):
        # Encoder mode handling
        # During warmup: encoder is frozen (requires_grad=False), keep in train
        # mode for BN stats unless bn_pretrain overrides.
        if not self.train_encoder and self.edc_encoder.training:
            self.edc_encoder.eval()
        if self.bn_pretrain and self.edc_encoder.training:
            self.edc_encoder.eval()

        B = x.shape[0]

        # ── Encoder forward ───────────────────────────────────────────────
        e1, e2, e3, e4 = self.edc_encoder(x)

        # Detach e4 if encoder is not being trained (permanent frozen mode)
        if not self.train_encoder:
            e4 = e4.detach()

        # ── Decoder forward ───────────────────────────────────────────────
        d1, d2, d3 = self.edc_decoder(e4)

        # ── Detach encoder maps for loss ──────────────────────────────────
        # stop_grad detaches e1/e2/e3 so they don't receive decoder gradients.
        # This is the normal EDC behaviour (stop_grad=True by default).
        # During warmup, encoder is frozen via requires_grad anyway, so
        # this is belt-and-braces.
        if (not self.train_encoder) or self.stop_grad:
            e1 = e1.detach()
            e2 = e2.detach()
            e3 = e3.detach()

        # ── Cosine-distance loss ──────────────────────────────────────────
        if self.reshape:
            l1 = 1. - torch.cosine_similarity(
                d1.reshape(B, -1), e1.reshape(B, -1), dim=1).mean()
            l2 = 1. - torch.cosine_similarity(
                d2.reshape(B, -1), e2.reshape(B, -1), dim=1).mean()
            l3 = 1. - torch.cosine_similarity(
                d3.reshape(B, -1), e3.reshape(B, -1), dim=1).mean()
        else:
            l1 = 1. - torch.cosine_similarity(d1, e1, dim=1).mean()
            l2 = 1. - torch.cosine_similarity(d2, e2, dim=1).mean()
            l3 = 1. - torch.cosine_similarity(d3, e3, dim=1).mean()

        # ── Pixel-wise anomaly maps (no grad) ─────────────────────────────
        with torch.no_grad():
            p1 = 1. - torch.cosine_similarity(d1, e1, dim=1).unsqueeze(1)
            p2 = 1. - torch.cosine_similarity(d2, e2, dim=1).unsqueeze(1)
            p3 = 1. - torch.cosine_similarity(d3, e3, dim=1).unsqueeze(1)

        loss = l1 + l2 + l3

        p2 = F.interpolate(p2, scale_factor=2,  mode='bilinear', align_corners=False)
        p3 = F.interpolate(p3, scale_factor=4,  mode='bilinear', align_corners=False)

        p_all = [[p1, p2, p3][l - 1] for l in self.anomap_layer]
        p_all = torch.cat(p_all, dim=1).mean(dim=1, keepdim=True)

        # ── Feature diversity stats (for tensorboard) ─────────────────────
        with torch.no_grad():
            e1_std = F.normalize(
                e1.permute(1, 0, 2, 3).flatten(1), dim=0).std(dim=1).mean()
            e2_std = F.normalize(
                e2.permute(1, 0, 2, 3).flatten(1), dim=0).std(dim=1).mean()
            e3_std = F.normalize(
                e3.permute(1, 0, 2, 3).flatten(1), dim=0).std(dim=1).mean()

        return {
            'loss'  : loss,
            'p_all' : p_all,
            'p1'    : p1,
            'p2'    : p2,
            'p3'    : p3,
            'e1_std': e1_std,
            'e2_std': e2_std,
            'e3_std': e3_std,
        }
"""
models/edc_ssl.py

SSL-EDC model: identical architecture and loss to the baseline R50_R50,
but the encoder is initialised from MoCo-pretrained weights instead of
ImageNet weights.

The forward() output dictionary is 100% identical to R50_R50 so the
existing EDC runner (methods/edc1.py) requires zero changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet_moco    import MoCoResNet50Encoder
from models.resnet_decoder import resnet50_decoder


class MoCo_R50_R50(nn.Module):
    """
    SSL-EDC: MoCo-pretrained ResNet-50 encoder + randomly-initialised
    ResNet-50 decoder.  Drop-in replacement for models.edc.R50_R50.

    Parameters
    ----------
    moco_weights_path : str
        Path to the MoCo .pth file that contains {"encoder_q": state_dict}.
    img_size : int
        Input spatial size (default 256).
    train_encoder : bool
        If False the encoder is kept frozen throughout training.
        Default True → fine-tune the encoder with a lower lr.
    stop_grad : bool
        If True, encoder feature maps e1/e2/e3 are detached before the
        cosine-loss computation (same behaviour as baseline stop_grad=True).
    reshape : bool
        Use flattened cosine similarity for loss (same as baseline).
    bn_pretrain : bool
        If True, encoder BatchNorm layers are forced into eval() mode
        (useful when fine-tuning on small batches).
    anomap_layer : list[int]
        Which decoder levels contribute to the anomaly map [1,2,3].
    freeze_encoder : bool
        Hard-freeze all encoder parameters (requires_grad=False).
        Overrides train_encoder when True.
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
    ):
        super().__init__()

        if anomap_layer is None:
            anomap_layer = [1, 2, 3]

        # ── Encoder (MoCo SSL pretrained) ────────────────────────────────────
        self.edc_encoder = MoCoResNet50Encoder(
            moco_weights_path=moco_weights_path,
            freeze_encoder=freeze_encoder,
        )

        # ── Decoder (randomly initialised, identical to baseline) ────────────
        self.edc_decoder = resnet50_decoder(pretrained=False, inplanes=[2048])

        # ── Hyper-parameters ─────────────────────────────────────────────────
        self.train_encoder = train_encoder
        self.stop_grad     = stop_grad
        self.reshape       = reshape
        self.bn_pretrain   = bn_pretrain
        self.anomap_layer  = anomap_layer

    # ------------------------------------------------------------------
    # forward  — identical logic to R50_R50.forward()
    # ------------------------------------------------------------------
    def forward(self, x):
        # Put encoder in eval mode when requested
        if not self.train_encoder and self.edc_encoder.training:
            self.edc_encoder.eval()
        if self.bn_pretrain and self.edc_encoder.training:
            self.edc_encoder.eval()

        B = x.shape[0]

        # ── Encoder forward ───────────────────────────────────────────────────
        e1, e2, e3, e4 = self.edc_encoder(x)

        if not self.train_encoder:
            e4 = e4.detach()

        # ── Decoder forward ───────────────────────────────────────────────────
        # resnet50_decoder.forward() returns (f1, f2, f3)
        # which correspond to d1, d2, d3 in the paper
        d1, d2, d3 = self.edc_decoder(e4)

        # ── Detach encoder maps for loss if stop_grad ─────────────────────────
        if (not self.train_encoder) or self.stop_grad:
            e1 = e1.detach()
            e2 = e2.detach()
            e3 = e3.detach()

        # ── Cosine-distance loss ──────────────────────────────────────────────
        if self.reshape:
            l1 = 1. - torch.cosine_similarity(d1.reshape(B, -1), e1.reshape(B, -1), dim=1).mean()
            l2 = 1. - torch.cosine_similarity(d2.reshape(B, -1), e2.reshape(B, -1), dim=1).mean()
            l3 = 1. - torch.cosine_similarity(d3.reshape(B, -1), e3.reshape(B, -1), dim=1).mean()
        else:
            l1 = 1. - torch.cosine_similarity(d1, e1, dim=1).mean()
            l2 = 1. - torch.cosine_similarity(d2, e2, dim=1).mean()
            l3 = 1. - torch.cosine_similarity(d3, e3, dim=1).mean()

        # ── Pixel-wise anomaly maps (no grad) ─────────────────────────────────
        with torch.no_grad():
            p1 = 1. - torch.cosine_similarity(d1, e1, dim=1).unsqueeze(1)
            p2 = 1. - torch.cosine_similarity(d2, e2, dim=1).unsqueeze(1)
            p3 = 1. - torch.cosine_similarity(d3, e3, dim=1).unsqueeze(1)

        loss = l1 + l2 + l3

        # Upsample p2, p3 to match p1 spatial size
        p2 = F.interpolate(p2, scale_factor=2, mode='bilinear', align_corners=False)
        p3 = F.interpolate(p3, scale_factor=4, mode='bilinear', align_corners=False)

        p_all = [[p1, p2, p3][l - 1] for l in self.anomap_layer]
        p_all = torch.cat(p_all, dim=1).mean(dim=1, keepdim=True)

        # ── Feature diversity statistics (for tensorboard) ───────────────────
        with torch.no_grad():
            e1_std = F.normalize(e1.permute(1, 0, 2, 3).flatten(1), dim=0).std(dim=1).mean()
            e2_std = F.normalize(e2.permute(1, 0, 2, 3).flatten(1), dim=0).std(dim=1).mean()
            e3_std = F.normalize(e3.permute(1, 0, 2, 3).flatten(1), dim=0).std(dim=1).mean()

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

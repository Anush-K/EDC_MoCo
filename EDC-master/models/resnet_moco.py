"""
models/resnet_moco.py

Wraps a MoCo-pretrained ResNet-50 encoder_q backbone so it behaves
identically to the custom ResNet in models/resnet.py.

The baseline ResNet.forward() returns (f1, f2, f3, f4):
    f1  — layer1 output  : [B, 256,  64, 64]   (for 256x256 input)
    f2  — layer2 output  : [B, 512,  32, 32]
    f3  — layer3 output  : [B,1024,  16, 16]   → e3 used in EDC loss
    f4  — layer4 output  : [B,2048,   8,  8]   → e4 latent, fed to decoder

The MoCo checkpoint saves encoder_q which is a full ResNet-50 **with** a
3-layer MLP projection head appended to the fc layer.  We strip the MLP
head and keep only the convolutional backbone (conv1 → layer4), making
the output signature identical to the baseline encoder.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50


class MoCoResNet50Encoder(nn.Module):
    """
    Drop-in replacement for models.resnet.resnet50(pretrained=True).

    Usage
    -----
        encoder = MoCoResNet50Encoder(
            moco_weights_path="/path/to/moco_all5datasets_allN_200ep.pth"
        )
        # Returns exactly the same tuple as the baseline encoder:
        f1, f2, f3, f4 = encoder(x)
    """

    def __init__(self, moco_weights_path: str, freeze_encoder: bool = False):
        super().__init__()

        # ── 1. Build a stock ResNet-50 (no pretrained weights yet) ──────────
        backbone = resnet50(pretrained=False)

        # MoCo encoder_q has a custom MLP head instead of the default fc.
        # We must replicate the same head structure used during MoCo training
        # so that state_dict keys match, then we discard it afterwards.
        dim_mlp = backbone.fc.weight.shape[1]          # 2048
        backbone.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, 128)
        )

        # ── 2. Load MoCo pretrained weights ─────────────────────────────────
        checkpoint = torch.load(moco_weights_path, map_location="cpu")

        # Checkpoint may be saved as {"encoder_q": state_dict} or directly
        if "encoder_q" in checkpoint:
            state_dict = checkpoint["encoder_q"]
        else:
            state_dict = checkpoint

        missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[MoCoEncoder] Missing keys  ({len(missing)}): {missing[:5]} ...")
        if unexpected:
            print(f"[MoCoEncoder] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")
        print(f"[MoCoEncoder] Loaded MoCo weights from: {moco_weights_path}")

        # ── 3. Strip the MLP head — keep only the convolutional backbone ────
        # backbone children order (standard ResNet-50):
        #   0: conv1, 1: bn1, 2: relu, 3: maxpool,
        #   4: layer1, 5: layer2, 6: layer3, 7: layer4,
        #   8: avgpool, 9: fc  ← drop avgpool + fc
        self.conv1   = backbone.conv1
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1  = backbone.layer1   # → f1  [B, 256, 64, 64]
        self.layer2  = backbone.layer2   # → f2  [B, 512, 32, 32]
        self.layer3  = backbone.layer3   # → f3  [B,1024, 16, 16]
        self.layer4  = backbone.layer4   # → f4  [B,2048,  8,  8]

        # ── 4. Optional: freeze the whole encoder ───────────────────────────
        if freeze_encoder:
            for param in self.parameters():
                param.requires_grad = False
            print("[MoCoEncoder] Encoder weights FROZEN.")
        else:
            print("[MoCoEncoder] Encoder weights are TRAINABLE (fine-tune mode).")

    def forward(self, x):
        x  = self.conv1(x)
        x  = self.bn1(x)
        x  = self.relu(x)
        x  = self.maxpool(x)

        f1 = self.layer1(x)    # [B,  256, 64, 64]
        f2 = self.layer2(f1)   # [B,  512, 32, 32]
        f3 = self.layer3(f2)   # [B, 1024, 16, 16]
        f4 = self.layer4(f3)   # [B, 2048,  8,  8]

        return f1, f2, f3, f4

"""
MoCo Pretraining Script  —  Improved for Medical Anomaly Detection
===================================================================

Key improvements over the original script:

1. ADAPTIVE AUGMENTATION
   - At dataset load time, a small sample of images is inspected to
     measure colour variance and image sharpness per dataset.
   - ColorJitter is gated: near-grayscale datasets (CT, OCT) receive
     reduced or zero colour jitter because their channels are nearly
     identical and jitter adds noise without useful augmentation signal.
   - GaussianBlur sigma is capped relative to measured image sharpness
     so we never destroy the fine texture that EDC relies on for anomaly
     scoring.
   - Crop scale lower bound is raised from 0.5 → 0.7 so MoCo never sees
     a crop so small it loses lesion-scale context.
   - Rotation is capped at 10° (was 20°) because structured anatomy
     (e.g. lung fields, retinal vasculature) has orientation meaning.
   - All decisions are made from image statistics, not dataset names, so
     the script generalises to any new dataset without code changes.

2. DENSE LOCAL CONTRASTIVE LOSS (auxiliary heads on layer2 + layer3)
   - Standard MoCo pools everything to a 128-d global vector and
     optimises only that. The spatial feature maps at layer2/layer3
     (which EDC uses directly for anomaly scoring) are bystanders.
   - We add small GAP + MLP projection heads on layer2 and layer3 of
     encoder_q and encoder_k. These compute an additional InfoNCE loss
     at each intermediate scale with its own queue.
   - Weight: global loss (layer4) × 1.0 + layer3 loss × 0.5 +
             layer2 loss × 0.3.  Layer2/3 losses are smaller because
             those feature maps are coarser and less reliable early on.
   - AT SAVE TIME: only encoder_q.state_dict() is saved, exactly as
     before. The auxiliary heads are not saved. The checkpoint format
     is 100% identical to the original script. MoCoResNet50Encoder and
     edc_ssl.py load it without any changes.

3. DATASET-SIZE-AWARE QUEUE AND MOMENTUM
   - Original K=4096, m=0.999 are ImageNet defaults (1.2M images).
   - For smaller medical datasets the queue can easily exceed the total
     dataset size, causing stale negatives to dominate.
   - K is automatically set to min(4096, dataset_size // 4) and clamped
     to a multiple of the batch size.
   - m is set to 0.996 for datasets < 5000 images and 0.999 otherwise.
     Slightly lower momentum means the key encoder adapts faster, which
     helps when the dataset is small and diversity is limited.

4. WARMUP LR SCHEDULE
   - Original: cosine annealing starts from epoch 0.
   - Improved: 5-epoch linear warmup then cosine annealing. This stops
     the large early gradients from corrupting the random initialisation
     before the contrastive objective has stabilised.

5. CHECKPOINT SAVES FULL MODEL STATE
   - Original checkpoint only saved encoder_q weights, so resuming
     required reloading without the full MoCo state (queue, key encoder).
   - Improved checkpoint saves the full model state_dict (includes queue,
     queue_ptr, key encoder) plus optimizer and scheduler. The final
     save for EDC use still exports only encoder_q, as before.

6. t-SNE VISUALISATION IMPROVEMENTS
   - Plots both layer4 (global) features and layer3 (spatial, pre-pool)
     features side by side. If the SSL is working well, layer3 clusters
     should be as tight as layer4 clusters.
   - Colours auto-generated so the script handles any number of datasets.
   - Intra-cluster compactness score (mean pairwise cosine distance within
     each dataset cluster) printed to stdout so you have a scalar metric
     to track across runs without needing to eyeball the plot.

HOW TO VERIFY THE IMPROVEMENTS HELPED
--------------------------------------
A. During pretraining:
   - Watch "Loss_l4", "Loss_l3", "Loss_l2" in the tqdm bar.
     All three should decrease. If l2/l3 plateau while l4 drops, the
     local heads are not learning — check the weight scale.
   - If total loss is much higher than the original (~3× or more at
     epoch 1), the local loss weights may need reducing.

B. t-SNE after pretraining:
   - Clusters should be tighter and more separated than the original
     script's t-SNE.
   - The "Intra-cluster cosine distance" printout should be lower than
     a run with the original script on the same data.

C. During EDC fine-tuning (checked in edc1.py logs):
   - e1_std, e2_std, e3_std should be noticeably higher from iteration 1
     with the improved MoCo weights compared to baseline ImageNet weights.
     Higher std = more diverse feature maps = better anomaly discriminability.
   - AUROC at convergence should exceed baseline. If it does not, the
     most likely culprit is augmentation still being too strong — try
     reducing BLUR_SIGMA_MAX and JITTER_STRENGTH below.

USAGE
-----
python3 MoCo_Pretrain_tSNE.py \
    --data_roots \
        /home/cs24d0008/EDC_SSL/LungCT/train \
        /home/cs24d0008/EDC_SSL/APTOS/train \
        /home/cs24d0008/EDC_SSL/BUSI/train \
    --save_path /home/cs24d0008/EDC_SSL/EDC_Improved_Weights \
    --epochs 200 \
    --batch_size 128 \
    --run_name multi_v1

SANITY CHECK
python3 MoCo_Pretrain_tSNE.py \
    --data_roots /home/cs24d0008/EDC_SSL/LungCT/train \
    --save_path ./weights/test \
    --epochs 1 \
    --batch_size 128 \
    --skip_tsne

  # Single-dataset pretrain (approach 2):
  python3 MoCo_Pretrain_tSNE.py \
      --data_roots /path/LungCT/train \
      --save_path  /path/to/weights \
      --epochs     200 \
      --batch_size 64
"""
# tmux new -s moco

# chmod +x run_all_ssl_finetune.sh
# ./run_all_baseline.sh
# ./run_all_ssl_finetune.sh
# ./run_all_ssl_frozen.sh



# ==============================================================
# IMPORTS
# ==============================================================
import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import Counter
from PIL import Image, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings("ignore")


# ==============================================================
# 1. ARGUMENT PARSING
# ==============================================================
def get_args():
    parser = argparse.ArgumentParser(description="Improved MoCo Pretraining for Medical UAD")

    # Data
    parser.add_argument(
        "--data_roots", nargs="+", required=True,
        help=(
            "One or more dataset root folders. Each folder must be an "
            "ImageFolder-compatible directory (subdirectory = class name). "
            "If a NORMAL subfolder exists, only NORMAL images are used. "
            "Example: --data_roots /data/LungCT/train /data/ISIC/train"
        )
    )
    parser.add_argument(
        "--save_path", type=str, default="./MoCo_Weights",
        help="Directory to save checkpoints and final weights."
    )

    # Training
    parser.add_argument("--epochs",     type=int,   default=200)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=0.06)
    parser.add_argument("--momentum",   type=float, default=0.9,   help="SGD momentum")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Linear LR warmup epochs before cosine annealing.")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--num_workers", type=int,  default=4)

    # MoCo hyperparams (auto-tuned if not provided)
    parser.add_argument("--moco_dim",  type=int,   default=128,  help="Projection head output dim.")
    parser.add_argument("--moco_k",    type=int,   default=None,
                        help="Queue size. Auto-computed from dataset size if not given.")
    parser.add_argument("--moco_m",    type=float, default=None,
                        help="Momentum for key encoder. Auto-set from dataset size if not given.")
    parser.add_argument("--moco_t",    type=float, default=0.07, help="Temperature.")

    # Local loss weights
    parser.add_argument("--w_local_l3", type=float, default=0.5,
                        help="Weight for layer3 local contrastive loss.")
    parser.add_argument("--w_local_l2", type=float, default=0.3,
                        help="Weight for layer2 local contrastive loss.")

    # Augmentation overrides (optional — normally auto-detected)
    parser.add_argument("--jitter_strength", type=float, default=None,
                        help="Override colour jitter strength [0,1]. Auto-detected if not set.")
    parser.add_argument("--blur_sigma_max",  type=float, default=None,
                        help="Override max GaussianBlur sigma. Auto-detected if not set.")
    parser.add_argument("--crop_scale_min",  type=float, default=0.7,
                        help="Minimum scale for RandomResizedCrop. Default 0.7 (was 0.5).")

    # t-SNE
    parser.add_argument("--tsne_max_per_dataset", type=int,   default=2000)
    parser.add_argument("--tsne_perplexity",       type=int,   default=50)
    parser.add_argument("--tsne_iter",             type=int,   default=1000)
    parser.add_argument("--skip_tsne",             action="store_true",
                        help="Skip t-SNE visualisation (saves time on large runs).")

    # Save name suffix
    parser.add_argument("--run_name", type=str, default=None,
                        help="Optional suffix for saved weight filename.")

    return parser.parse_args()


# ==============================================================
# 2. REPRODUCIBILITY
# ==============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==============================================================
# 3. ADAPTIVE AUGMENTATION
# ==============================================================

def _sample_images(root, n=64):
    """
    Sample up to n PIL images from an ImageFolder root for inspection.
    Works on filtered (NORMAL-only) roots too.
    """
    images = []
    for dirpath, _, fnames in os.walk(root):
        for fname in fnames:
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                images.append(os.path.join(dirpath, fname))
    if len(images) == 0:
        return []
    chosen = random.sample(images, min(n, len(images)))
    pil_imgs = []
    for p in chosen:
        try:
            pil_imgs.append(Image.open(p).convert("RGB"))
        except Exception:
            pass
    return pil_imgs


def measure_dataset_properties(root, n_sample=64):
    """
    Measure two properties of a dataset from a small image sample:

    colour_variance : float
        Mean per-channel standard deviation across sampled images.
        High → the dataset is truly coloured (e.g. dermoscopy).
        Low  → near-grayscale (e.g. CT, OCT, X-ray).
        Threshold: < 0.08 (on [0,1] scale) = treat as grayscale.

    sharpness : float
        Mean gradient magnitude (Sobel-like) normalised to [0,1].
        High → fine texture present (e.g. retinal vessels, lung texture).
        Low  → smooth images (e.g. some fundus backgrounds).
        Threshold: > 0.15 = texture-rich, be conservative with blur.

    Returns a dict with keys: colour_variance, sharpness, n_images.
    """
    imgs = _sample_images(root, n=n_sample)
    if not imgs:
        return {"colour_variance": 0.5, "sharpness": 0.5, "n_images": 0}

    colour_vars = []
    sharpnesses = []

    for img in imgs:
        arr = np.array(img).astype(np.float32) / 255.0  # H x W x 3

        # Colour variance: std of per-channel means across channels
        # If R≈G≈B, this is near 0 (grayscale-like)
        ch_means = arr.reshape(-1, 3).mean(axis=0)       # (3,)
        colour_vars.append(ch_means.std())

        # Sharpness: mean absolute gradient magnitude (Laplacian proxy)
        gray = arr.mean(axis=2)                           # H x W
        gy = np.abs(np.diff(gray, axis=0)).mean()
        gx = np.abs(np.diff(gray, axis=1)).mean()
        sharpnesses.append((gx + gy) / 2.0)

    return {
        "colour_variance": float(np.mean(colour_vars)),
        "sharpness":       float(np.mean(sharpnesses)),
        "n_images":        len(imgs),
    }


class GaussianBlur:
    """PIL GaussianBlur with configurable sigma range."""
    def __init__(self, sigma_min=0.1, sigma_max=1.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, x):
        sigma = random.uniform(self.sigma_min, self.sigma_max)
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))


def build_adaptive_transform(props, args, crop_size=224):
    """
    Build a MoCo augmentation pipeline adapted to the measured image
    properties of a dataset.

    Decisions made here (all data-driven, no dataset name used):

    - jitter_strength : scaled by colour_variance.
      If colour_variance < 0.05 (essentially grayscale), strength → 0
      (no colour jitter — it adds no useful invariance and can corrupt
      channel-encoded HU values in CT).
      Between 0.05 and 0.15 (weakly coloured), strength is interpolated.
      Above 0.15 (truly coloured), strength = args.jitter_strength or 0.4.

    - blur_sigma_max : scaled by sharpness.
      High sharpness (> 0.15) → cap blur at 1.0 to preserve texture.
      Low sharpness (< 0.05)  → allow up to 2.0 (image is smooth anyway).
      In between → linear interpolation.

    - rotation : fixed at 10° maximum (was 20°). Structured anatomy has
      orientation meaning; 10° is enough to provide rotational invariance
      without corrupting organ geometry.

    - crop_scale_min : args.crop_scale_min (default 0.7, was 0.5).
      Ensures every crop retains enough tissue context.
    """

    # --- Colour jitter strength ---
    cv = props["colour_variance"]
    if args.jitter_strength is not None:
        jitter_s = args.jitter_strength
    else:
        if cv < 0.05:
            jitter_s = 0.0          # grayscale: no jitter
        elif cv < 0.15:
            # linearly interpolate from 0 to 0.4 across [0.05, 0.15]
            jitter_s = 0.4 * (cv - 0.05) / 0.10
        else:
            jitter_s = 0.4          # fully coloured

    # --- Blur sigma max ---
    sh = props["sharpness"]
    if args.blur_sigma_max is not None:
        blur_max = args.blur_sigma_max
    else:
        if sh > 0.15:
            blur_max = 1.0          # texture-rich: limit blur
        elif sh < 0.05:
            blur_max = 2.0          # smooth: allow more blur
        else:
            # linear interp: from 2.0 (sharpness=0.05) to 1.0 (sharpness=0.15)
            blur_max = 2.0 - 1.0 * (sh - 0.05) / 0.10

    # --- Build transform ---
    aug_list = [
        transforms.Resize(256),
        transforms.RandomResizedCrop(crop_size, scale=(args.crop_scale_min, 1.0)),
        transforms.RandomHorizontalFlip(),
        # Rotation: always conservative for anatomy
        transforms.RandomApply([transforms.RandomRotation(degrees=10)], p=0.5),
        # Blur: always applied but sigma capped
        transforms.RandomApply([GaussianBlur(sigma_min=0.1, sigma_max=blur_max)], p=0.5),
    ]

    # Colour jitter: only if dataset is sufficiently coloured
    if jitter_s > 0.01:
        aug_list.append(
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=jitter_s,
                    contrast=jitter_s,
                    saturation=jitter_s * 0.5,
                    hue=jitter_s * 0.1,
                )
            ], p=0.5)
        )

    aug_list += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]

    return transforms.Compose(aug_list), jitter_s, blur_max


class TwoCropsTransform:
    """Apply the same transform twice independently to produce two views."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]


# ==============================================================
# 4. DATASET LOADING
# ==============================================================

def load_dataset(root, transform):
    """
    Load an ImageFolder dataset from root.
    If a NORMAL class exists, filter to NORMAL images only
    (consistent with EDC training which is normal-only).
    Falls back to all images if no NORMAL class is found
    (handles per-dataset pretrain where root may already point to
    the NORMAL folder directly).
    """
    dataset = ImageFolder(root, transform=TwoCropsTransform(transform))

    if "NORMAL" in dataset.class_to_idx:
        normal_idx = dataset.class_to_idx["NORMAL"]
        dataset.samples = [s for s in dataset.samples if s[1] == normal_idx]
        dataset.targets = [normal_idx] * len(dataset.samples)
        print(f"    Filtered to NORMAL class only.")

    return dataset


# ==============================================================
# 5. AUXILIARY PROJECTION HEADS
# ==============================================================

class LocalProjectionHead(nn.Module):
    """
    Small GAP + 2-layer MLP projection head attached to an intermediate
    feature map (layer2 or layer3).

    Input : [B, C, H, W]  — spatial feature map
    Output: [B, dim]      — L2-normalised projection vector

    This head is used ONLY during pretraining. It is never saved to the
    checkpoint and has no effect on the downstream EDC model structure.

    Why GAP before the MLP?
    The contrastive loss operates on vectors, not maps. GAP collapses
    spatial dimensions. The loss then forces the GAP vector (= average
    spatial feature) of two augmented crops to be similar. This means
    layer2/layer3 spatial features must encode crop-invariant content
    rather than just being optimised as a pass-through for layer4.
    """
    def __init__(self, in_channels, hidden_dim=512, out_dim=128):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        x = self.gap(x).flatten(1)   # [B, C]
        x = self.mlp(x)              # [B, out_dim]
        return F.normalize(x, dim=1)


# ==============================================================
# 6. IMPROVED MoCo MODEL
# ==============================================================

class MoCoImproved(nn.Module):
    """
    MoCo v2 with auxiliary dense contrastive loss on layer2 and layer3.

    Architecture
    ------------
    encoder_q / encoder_k : ResNet-50 backbone + 3-layer MLP head on fc
        (identical to original — checkpoint compatible)
    head_q_l3 / head_k_l3 : LocalProjectionHead on layer3 output [B,1024,H,W]
    head_q_l2 / head_k_l2 : LocalProjectionHead on layer2 output [B, 512,H,W]

    Three queues
    ------------
    queue    (dim × K)  — global (layer4) negatives
    queue_l3 (dim × K)  — layer3 negatives
    queue_l2 (dim × K)  — layer2 negatives

    All three queues have the same K and are managed identically.

    Forward returns a dict with keys: loss, loss_l4, loss_l3, loss_l2
    so the training loop can log each component separately.

    Save behaviour
    --------------
    Only encoder_q.state_dict() is saved. The auxiliary heads are
    discarded. The checkpoint format matches the original script exactly.
    """

    def __init__(self, dim=128, K=4096, m=0.999, T=0.07,
                 w_l3=0.5, w_l2=0.3):
        super().__init__()

        self.K   = K
        self.m   = m
        self.T   = T
        self.w_l3 = w_l3
        self.w_l2 = w_l2

        # ── Backbone builders ─────────────────────────────────────────────
        def _make_encoder():
            enc = models.resnet50(pretrained=False)
            dim_mlp = enc.fc.weight.shape[1]          # 2048
            enc.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.ReLU(),
                nn.Linear(dim_mlp, dim),
            )
            return enc

        self.encoder_q = _make_encoder()
        self.encoder_k = _make_encoder()

        # Initialise key encoder from query encoder, freeze key encoder
        for pq, pk in zip(self.encoder_q.parameters(),
                          self.encoder_k.parameters()):
            pk.data.copy_(pq.data)
            pk.requires_grad = False

        # ── Auxiliary projection heads ────────────────────────────────────
        # layer3 output: [B, 1024, H, W]
        self.head_q_l3 = LocalProjectionHead(1024, hidden_dim=512, out_dim=dim)
        self.head_k_l3 = LocalProjectionHead(1024, hidden_dim=512, out_dim=dim)
        # layer2 output: [B,  512, H, W]
        self.head_q_l2 = LocalProjectionHead(512,  hidden_dim=256, out_dim=dim)
        self.head_k_l2 = LocalProjectionHead(512,  hidden_dim=256, out_dim=dim)

        # Initialise key auxiliary heads from query heads, freeze
        for pq, pk in zip(self.head_q_l3.parameters(),
                          self.head_k_l3.parameters()):
            pk.data.copy_(pq.data)
            pk.requires_grad = False

        for pq, pk in zip(self.head_q_l2.parameters(),
                          self.head_k_l2.parameters()):
            pk.data.copy_(pq.data)
            pk.requires_grad = False

        # ── Queues ─────────────────────────────────────────────────────────
        # Global (layer4)
        self.register_buffer("queue",     torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Layer3
        self.register_buffer("queue_l3",     torch.randn(dim, K))
        self.queue_l3 = F.normalize(self.queue_l3, dim=0)
        self.register_buffer("queue_l3_ptr", torch.zeros(1, dtype=torch.long))

        # Layer2
        self.register_buffer("queue_l2",     torch.randn(dim, K))
        self.queue_l2 = F.normalize(self.queue_l2, dim=0)
        self.register_buffer("queue_l2_ptr", torch.zeros(1, dtype=torch.long))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _momentum_update(self):
        """EMA update for key encoder and all key auxiliary heads."""
        for pq, pk in zip(self.encoder_q.parameters(),
                          self.encoder_k.parameters()):
            pk.data = pk.data * self.m + pq.data * (1.0 - self.m)

        for pq, pk in zip(self.head_q_l3.parameters(),
                          self.head_k_l3.parameters()):
            pk.data = pk.data * self.m + pq.data * (1.0 - self.m)

        for pq, pk in zip(self.head_q_l2.parameters(),
                          self.head_k_l2.parameters()):
            pk.data = pk.data * self.m + pq.data * (1.0 - self.m)

    @torch.no_grad()
    def _enqueue(self, queue_buf, ptr_buf, keys):
        """Enqueue a batch of keys into the given queue buffer."""
        B = keys.shape[0]
        ptr = int(ptr_buf)
        # Wrap-around write (drop_last=True guarantees B divides K evenly)
        queue_buf[:, ptr:ptr + B] = keys.T
        ptr_buf[0] = (ptr + B) % self.K

    def _infoNCE(self, q, k, queue):
        """
        Compute InfoNCE (MoCo) loss given query q, key k, and negative queue.
        q, k : [B, dim] L2-normalised
        queue : [dim, K]
        Returns scalar loss.
        """
        l_pos = torch.einsum("nc,nc->n", q, k).unsqueeze(-1)      # [B, 1]
        l_neg = torch.einsum("nc,ck->nk", q, queue.clone().detach())  # [B, K]
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T        # [B, 1+K]
        labels = torch.zeros(logits.shape[0], dtype=torch.long,
                             device=logits.device)
        return F.cross_entropy(logits, labels)

    # ------------------------------------------------------------------
    # Intermediate feature extraction
    # ------------------------------------------------------------------

    def _encode_q(self, x):
        """
        Forward pass through encoder_q, returning intermediate maps.
        Returns: (l2_feat, l3_feat, global_proj)
            l2_feat  : layer2 output  [B,  512, H2, W2]
            l3_feat  : layer3 output  [B, 1024, H3, W3]
            global_q : MLP projection [B, dim]  (L2-normalised)
        """
        x = self.encoder_q.conv1(x)
        x = self.encoder_q.bn1(x)
        x = self.encoder_q.relu(x)
        x = self.encoder_q.maxpool(x)
        x = self.encoder_q.layer1(x)
        l2 = self.encoder_q.layer2(x)      # [B, 512, ...]
        l3 = self.encoder_q.layer3(l2)     # [B, 1024, ...]
        l4 = self.encoder_q.layer4(l3)     # [B, 2048, ...]
        # avgpool + fc (the MLP head)
        out = self.encoder_q.avgpool(l4).flatten(1)
        out = self.encoder_q.fc(out)
        return l2, l3, F.normalize(out, dim=1)

    @torch.no_grad()
    def _encode_k(self, x):
        """
        Same as _encode_q but for the key encoder (no_grad context assumed
        by caller).
        """
        x = self.encoder_k.conv1(x)
        x = self.encoder_k.bn1(x)
        x = self.encoder_k.relu(x)
        x = self.encoder_k.maxpool(x)
        x = self.encoder_k.layer1(x)
        l2 = self.encoder_k.layer2(x)
        l3 = self.encoder_k.layer3(l2)
        l4 = self.encoder_k.layer4(l3)
        out = self.encoder_k.avgpool(l4).flatten(1)
        out = self.encoder_k.fc(out)
        return l2, l3, F.normalize(out, dim=1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, im_q, im_k):
        # Query forward
        l2_q, l3_q, q = self._encode_q(im_q)

        with torch.no_grad():
            self._momentum_update()
            l2_k, l3_k, k = self._encode_k(im_k)

        # Global (layer4) InfoNCE loss
        loss_l4 = self._infoNCE(q, k, self.queue)

        # Layer3 local InfoNCE loss
        q_l3 = self.head_q_l3(l3_q)
        with torch.no_grad():
            k_l3 = self.head_k_l3(l3_k)
        loss_l3 = self._infoNCE(q_l3, k_l3, self.queue_l3)

        # Layer2 local InfoNCE loss
        q_l2 = self.head_q_l2(l2_q)
        with torch.no_grad():
            k_l2 = self.head_k_l2(l2_k)
        loss_l2 = self._infoNCE(q_l2, k_l2, self.queue_l2)

        # Total loss
        loss = loss_l4 + self.w_l3 * loss_l3 + self.w_l2 * loss_l2

        # Enqueue
        self._enqueue(self.queue,    self.queue_ptr,    k)
        self._enqueue(self.queue_l3, self.queue_l3_ptr, k_l3)
        self._enqueue(self.queue_l2, self.queue_l2_ptr, k_l2)

        return {
            "loss"    : loss,
            "loss_l4" : loss_l4,
            "loss_l3" : loss_l3,
            "loss_l2" : loss_l2,
        }


# ==============================================================
# 7. LR SCHEDULE WITH WARMUP
# ==============================================================

class WarmupCosineScheduler:
    """
    Linear warmup for warmup_epochs, then cosine annealing.
    Wraps an optimizer directly (does not use torch.optim.lr_scheduler
    so it works cleanly with resume).
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.base_lr       = base_lr
        self.min_lr        = min_lr
        self._epoch        = 0

    def step(self):
        self._epoch += 1
        lr = self._get_lr(self._epoch)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _get_lr(self, epoch):
        if epoch <= self.warmup_epochs:
            return self.base_lr * epoch / self.warmup_epochs
        progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + np.cos(np.pi * progress))

    def get_last_lr(self):
        return [self._get_lr(self._epoch)]

    def state_dict(self):
        return {"_epoch": self._epoch}

    def load_state_dict(self, d):
        self._epoch = d["_epoch"]


# ==============================================================
# 8. t-SNE VISUALISATION
# ==============================================================

def run_tsne(save_path, dataset_roots, final_weights_path,
             max_per_dataset, perplexity, n_iter, seed, device):
    """
    Run t-SNE on both layer4 (global) and layer3 (spatial, pre-pool)
    features from the pretrained encoder_q.

    Prints intra-cluster cosine distance for each dataset as a scalar
    metric to track improvements across runs.
    """

    print("\n" + "=" * 60)
    print(" t-SNE VISUALISATION")
    print("=" * 60)

    # ── Build backbone (layer4 features) ─────────────────────────────
    def build_backbone_l4(weights_path):
        enc = resnet50(pretrained=False)
        dim_mlp = enc.fc.weight.shape[1]
        enc.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, 128)
        )
        ckpt = torch.load(weights_path, map_location="cpu")
        enc.load_state_dict(ckpt["encoder_q"])
        # Strip MLP head → avgpool output = 2048-d global feature
        backbone = nn.Sequential(*list(enc.children())[:-1])
        backbone.eval()
        return backbone

    # ── Build backbone (layer3 features) ─────────────────────────────
    class Layer3Backbone(nn.Module):
        def __init__(self, weights_path):
            super().__init__()
            enc = resnet50(pretrained=False)
            dim_mlp = enc.fc.weight.shape[1]
            enc.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, 128)
            )
            ckpt = torch.load(weights_path, map_location="cpu")
            enc.load_state_dict(ckpt["encoder_q"])
            self.stem    = nn.Sequential(enc.conv1, enc.bn1, enc.relu, enc.maxpool)
            self.layer1  = enc.layer1
            self.layer2  = enc.layer2
            self.layer3  = enc.layer3
            self.gap      = nn.AdaptiveAvgPool2d(1)

        def forward(self, x):
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            return self.gap(x).flatten(1)   # [B, 1024]

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def extract_features(backbone, roots, device, max_per_ds, seed):
        all_feats  = []
        all_labels = []
        all_names  = []
        for ds_idx, (name, root) in enumerate(roots.items()):
            if not os.path.exists(root):
                print(f"  Skipping {name} — path not found")
                continue
            ds = ImageFolder(root, transform=eval_transform)
            dl = DataLoader(ds, batch_size=64, shuffle=False,
                            num_workers=4, pin_memory=True)
            feats = []
            with torch.no_grad():
                for imgs, _ in tqdm(dl, desc=f"  Extracting [{name}]", leave=False):
                    feats.append(backbone(imgs.to(device)).cpu().numpy())
            feats = np.concatenate(feats, axis=0)
            if max_per_ds and len(feats) > max_per_ds:
                np.random.seed(seed)
                idx = np.random.choice(len(feats), max_per_ds, replace=False)
                feats = feats[idx]
            print(f"  {name}: {len(feats)} samples")
            all_feats.append(feats)
            all_labels.extend([ds_idx] * len(feats))
            all_names.append(name)
        return np.concatenate(all_feats, axis=0), np.array(all_labels), all_names

    print("\nLoading weights:", final_weights_path)
    backbone_l4 = build_backbone_l4(final_weights_path).to(device)
    backbone_l3 = Layer3Backbone(final_weights_path).to(device)
    backbone_l3.eval()

    print("\nExtracting layer4 features...")
    feats_l4, labels, names = extract_features(
        backbone_l4, dataset_roots, device, max_per_dataset, seed)

    print("\nExtracting layer3 features...")
    feats_l3, _, _ = extract_features(
        backbone_l3, dataset_roots, device, max_per_dataset, seed)

    # Auto-generate colours for any number of datasets
    cmap   = cm.get_cmap("tab10", len(names))
    colors = [cmap(i) for i in range(len(names))]

    def _intra_cluster_cosine(feats, labels, names):
        """Lower = tighter cluster = better representation."""
        scores = {}
        for i, name in enumerate(names):
            mask = labels == i
            f = feats[mask]
            if len(f) < 2:
                continue
            f_norm = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
            # Mean pairwise cosine similarity (sub-sample for speed)
            idx = np.random.choice(len(f_norm), min(500, len(f_norm)), replace=False)
            f_sub = f_norm[idx]
            sim = f_sub @ f_sub.T
            # Off-diagonal mean
            n = len(f_sub)
            off_diag = (sim.sum() - np.trace(sim)) / (n * (n - 1))
            scores[name] = float(off_diag)
        return scores

    def _plot_tsne(feats, labels, names, colors, title, out_path):
        print(f"\nRunning t-SNE for: {title}")
        tsne = TSNE(n_components=2, perplexity=perplexity,
                    n_iter=n_iter, random_state=seed, verbose=0)
        emb = tsne.fit_transform(feats)
        plt.figure(figsize=(10, 8))
        for i, (name, color) in enumerate(zip(names, colors)):
            mask = labels == i
            plt.scatter(emb[mask, 0], emb[mask, 1],
                        label=name, color=color, alpha=0.6, s=8)
        plt.title(title)
        plt.xlabel("t-SNE dim 1")
        plt.ylabel("t-SNE dim 2")
        plt.legend(markerscale=3, fontsize=9)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Saved → {out_path}")

    # Layer4 t-SNE
    scores_l4 = _intra_cluster_cosine(feats_l4, labels, names)
    _plot_tsne(feats_l4, labels, names, colors,
               "t-SNE — layer4 (global) features",
               os.path.join(save_path, "tsne_layer4.png"))

    # Layer3 t-SNE
    scores_l3 = _intra_cluster_cosine(feats_l3, labels, names)
    _plot_tsne(feats_l3, labels, names, colors,
               "t-SNE — layer3 (spatial, pre-pool) features",
               os.path.join(save_path, "tsne_layer3.png"))

    # Print scalar metrics
    print("\n── Intra-cluster cosine similarity (higher = tighter clusters) ──")
    print(f"{'Dataset':<20} {'Layer4':>10} {'Layer3':>10}")
    print("-" * 44)
    for name in names:
        l4 = scores_l4.get(name, float("nan"))
        l3 = scores_l3.get(name, float("nan"))
        print(f"{name:<20} {l4:>10.4f} {l3:>10.4f}")
    print()


# ==============================================================
# 9. MAIN
# ==============================================================

def main():
    args = get_args()
    set_seed(args.seed)

    os.makedirs(args.save_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Build dataset names from roots (used for logging and t-SNE only)
    # ------------------------------------------------------------------
    dataset_roots = {}
    for root in args.data_roots:
        root = root.rstrip("/")
        name = os.path.basename(root)
        dataset_roots[name] = root

    # ------------------------------------------------------------------
    # Measure image properties and build adaptive augmentation per dataset
    # ------------------------------------------------------------------
    print("\n── Dataset properties & augmentation settings ──")
    individual_datasets = {}
    aug_summary = {}

    for name, root in dataset_roots.items():
        print(f"\n[{name}]  root: {root}")
        props = measure_dataset_properties(root, n_sample=64)
        print(f"    colour_variance : {props['colour_variance']:.4f}"
              f"  ({'near-grayscale' if props['colour_variance'] < 0.05 else 'coloured'})")
        print(f"    sharpness       : {props['sharpness']:.4f}"
              f"  ({'texture-rich' if props['sharpness'] > 0.15 else 'smooth'})")

        transform, jitter_s, blur_max = build_adaptive_transform(props, args)
        print(f"    → jitter_strength : {jitter_s:.3f}")
        print(f"    → blur_sigma_max  : {blur_max:.2f}")
        print(f"    → crop_scale_min  : {args.crop_scale_min:.2f}")

        aug_summary[name] = {"jitter": jitter_s, "blur_max": blur_max}

        ds = load_dataset(root, transform)
        print(f"    images loaded   : {len(ds)}")
        individual_datasets[name] = ds

    # ------------------------------------------------------------------
    # Combined dataloader
    # ------------------------------------------------------------------
    combined_dataset = ConcatDataset(list(individual_datasets.values()))
    total_images = len(combined_dataset)
    print(f"\nTotal combined images: {total_images}")

    loader = DataLoader(
        combined_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Batches per epoch   : {len(loader)}")

    # ------------------------------------------------------------------
    # Auto-tune MoCo K and m
    # ------------------------------------------------------------------
    if args.moco_k is None:
        # Queue should be large enough to hold diverse negatives but not
        # exceed the dataset size. Rule: total_images // 4, clamped to
        # [256, 4096], then rounded down to nearest multiple of batch_size.
        raw_k = max(256, min(4096, total_images // 4))
        K = (raw_k // args.batch_size) * args.batch_size
        K = max(K, args.batch_size)   # at minimum one batch
    else:
        K = args.moco_k

    if args.moco_m is None:
        # Smaller datasets → faster adapting key encoder (lower momentum)
        m = 0.996 if total_images < 5000 else 0.999
    else:
        m = args.moco_m

    print(f"\nMoCo K (queue size) : {K}  (auto: {args.moco_k is None})")
    print(f"MoCo m (momentum)   : {m}  (auto: {args.moco_m is None})")
    print(f"MoCo T (temperature): {args.moco_t}")

    # ------------------------------------------------------------------
    # Model, optimiser, scheduler
    # ------------------------------------------------------------------
    model = MoCoImproved(
        dim  = args.moco_dim,
        K    = K,
        m    = m,
        T    = args.moco_t,
        w_l3 = args.w_local_l3,
        w_l2 = args.w_local_l2,
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr           = args.lr,
        momentum     = args.momentum,
        weight_decay = args.weight_decay,
    )

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs = args.warmup_epochs,
        total_epochs  = args.epochs,
        base_lr       = args.lr,
        min_lr        = 1e-6,
    )

    # AMP
    use_amp = (device.type == "cuda")
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None
    print(f"AMP enabled         : {use_amp}")

    # ------------------------------------------------------------------
    # Resume from checkpoint if available
    # ------------------------------------------------------------------
    start_epoch = 0
    resume_path = os.path.join(args.save_path, "moco_latest.pth")

    if os.path.exists(resume_path):
        print(f"\nResuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        # Full model state (includes queue and key encoder)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        print(f"Resuming from epoch {start_epoch}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f" MoCo PRETRAINING  ({start_epoch} → {args.epochs} epochs)")
    print("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        model.train()

        total_loss    = 0.0
        total_loss_l4 = 0.0
        total_loss_l3 = 0.0
        total_loss_l2 = 0.0

        pbar = tqdm(loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")

        for step, (images, _) in enumerate(pbar):
            im_q = images[0].to(device, non_blocking=True)
            im_k = images[1].to(device, non_blocking=True)

            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    out = model(im_q, im_k)
                scaler.scale(out["loss"]).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(im_q, im_k)
                out["loss"].backward()
                optimizer.step()

            total_loss    += out["loss"].item()
            total_loss_l4 += out["loss_l4"].item()
            total_loss_l3 += out["loss_l3"].item()
            total_loss_l2 += out["loss_l2"].item()

            n = step + 1
            pbar.set_postfix({
                "Total" : f"{total_loss / n:.4f}",
                "L4"    : f"{total_loss_l4 / n:.4f}",
                "L3"    : f"{total_loss_l3 / n:.4f}",
                "L2"    : f"{total_loss_l2 / n:.4f}",
                "LR"    : f"{scheduler.get_last_lr()[0]:.6f}",
            })

        scheduler.step()

        n_batches = len(loader)
        print(
            f"\nEpoch {epoch+1:4d} | "
            f"Total: {total_loss/n_batches:.4f} | "
            f"L4: {total_loss_l4/n_batches:.4f} | "
            f"L3: {total_loss_l3/n_batches:.4f} | "
            f"L2: {total_loss_l2/n_batches:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Full checkpoint (for resuming) — saves full model state
        torch.save({
            "epoch"    : epoch + 1,
            "model"    : model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, resume_path)

        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Save final weights — encoder_q only, identical format to original
    # ------------------------------------------------------------------
    suffix    = f"_{args.run_name}" if args.run_name else ""
    ds_tag    = "_".join(dataset_roots.keys())
    save_name = f"moco_{ds_tag}{suffix}_{args.epochs}ep.pth"
    save_file = os.path.join(args.save_path, save_name)

    torch.save({"encoder_q": model.encoder_q.state_dict()}, save_file)
    print(f"\nFinal encoder_q weights saved to:\n  {save_file}")
    print("Checkpoint format: {{\"encoder_q\": state_dict}}")
    print("Compatible with MoCoResNet50Encoder and edc_ssl.py — no changes needed.")

    # ------------------------------------------------------------------
    # t-SNE
    # ------------------------------------------------------------------
    if not args.skip_tsne:
        run_tsne(
            save_path         = args.save_path,
            dataset_roots     = dataset_roots,
            final_weights_path= save_file,
            max_per_dataset   = args.tsne_max_per_dataset,
            perplexity        = args.tsne_perplexity,
            n_iter            = args.tsne_iter,
            seed              = args.seed,
            device            = device,
        )
    else:
        print("\nt-SNE skipped (--skip_tsne).")

    print("\nDone.")


if __name__ == "__main__":
    main()

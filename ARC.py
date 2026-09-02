"""Aberrated Reflection-matrix corrector: covariance-aware, dual-head SwinV2-UNet for reflection matrix
aberration correction.

Architecture only -- no dataset loading, loss functions, or training loop here
(see the separate training script for those). Inputs are R, RR^dagger, and
R^dagger R, each represented as [input_channels, 40, 40] (real/imag stacked
per illumination channel, matching the base papers' "1600x40x40" convention --
see the H5 adapter dataset in the training script for how these are built
losslessly from the true 1600x1600 reflection matrix, no resizing involved).

Outputs (return_dict=True):
    {
        "output_aberration": [B, 2, 40, 40],  # unit phasor (cos, sin) of phi_o
        "input_aberration":  [B, 2, 40, 40],  # unit phasor (cos, sin) of phi_i
    }
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# =============================================================================
# Small utility functions used inside the model
# =============================================================================


def center_crop_like(x: Tensor, ref: Tensor) -> Tensor:
    """Center-crop tensor x to the spatial size of ref.

    Args:
        x:   Tensor [B, C, H, W]
        ref: Tensor [B, C_ref, H_ref, W_ref]
    """
    _, _, h, w = x.shape
    _, _, rh, rw = ref.shape
    if h == rh and w == rw:
        return x
    if rh > h or rw > w:
        raise ValueError(f"Cannot crop tensor of size {(h, w)} to larger size {(rh, rw)}")
    top = (h - rh) // 2
    left = (w - rw) // 2
    return x[:, :, top : top + rh, left : left + rw]


def make_na_mask(size: int = 40, nasz: int = 20, device: Optional[torch.device] = None) -> Tensor:
    """Return NA support mask with shape [1, 1, size, size].

    This follows the MATLAB generation logic:
        kc = 20.5
        NA_mask = (kx2-kc).^2 + (ky2-kc).^2 < NAsz^2
    """
    coords = torch.arange(1, size + 1, dtype=torch.float32, device=device)
    ky, kx = torch.meshgrid(coords, coords, indexing="ij")
    kc = size / 2.0 + 0.5
    mask = ((kx - kc) ** 2 + (ky - kc) ** 2) < float(nasz**2)
    return mask.float().unsqueeze(0).unsqueeze(0)


def normalize_phasor(x: Tensor, mask: Optional[Tensor] = None, eps: float = 1e-8) -> Tensor:
    """Normalize a two-channel phasor to unit magnitude.

    Args:
        x: Tensor [B, 2, H, W]. Channel 0 is real, channel 1 is imaginary.
        mask: Optional mask [1, 1, H, W] or [B, 1, H, W]. If supplied,
              normalization is applied inside the mask and output is zeroed
              outside the mask. This matches the generated truth support.
    """
    if x.shape[1] != 2:
        raise ValueError("normalize_phasor expects exactly 2 channels")
    mag = torch.sqrt(torch.sum(x * x, dim=1, keepdim=True) + eps)
    y = x / mag
    if mask is not None:
        y = y * mask.to(dtype=y.dtype, device=y.device)
    return y



# =============================================================================
# Window attention helpers, SwinV2 encoder, U-Net decoder, heads, full model
# =============================================================================


def window_partition(x: Tensor, window_size: int) -> Tensor:
    """Partition NHWC tensor into non-overlapping windows.

    Args:
        x: Tensor [B, H, W, C]

    Returns:
        Tensor [num_windows * B, window_size * window_size, C]
    """
    b, h, w, c = x.shape
    if h % window_size != 0 or w % window_size != 0:
        raise ValueError(f"H={h}, W={w} must be divisible by window_size={window_size}")
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(-1, window_size * window_size, c)


def window_reverse(windows: Tensor, window_size: int, h: int, w: int) -> Tensor:
    """Reverse window partition back to NHWC tensor."""
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(b, h, w, -1)


def build_shifted_window_mask(
    input_resolution: Tuple[int, int],
    window_size: int,
    shift_size: int,
    device: torch.device,
) -> Optional[Tensor]:
    """Build attention mask for shifted-window attention.

    Returns a tensor broadcastable to [num_windows, num_heads, N, N].
    """
    if shift_size == 0:
        return None

    h, w = input_resolution
    img_mask = torch.zeros((1, h, w, 1), device=device)
    h_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    w_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    cnt = 0
    for hs in h_slices:
        for ws in w_slices:
            img_mask[:, hs, ws, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
    return attn_mask.unsqueeze(1)


# =============================================================================
# SwinV2-style encoder
# =============================================================================


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class WindowAttentionV2(nn.Module):
    """Window attention with SwinV2-style cosine attention and CPB."""

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)
        self.logit_scale = nn.Parameter(torch.log(10.0 * torch.ones(num_heads, 1, 1)))

        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous().float()

        relative_coords[:, :, 0] /= max(window_size - 1, 1)
        relative_coords[:, :, 1] /= max(window_size - 1, 1)
        relative_coords *= 8.0
        relative_coords = torch.sign(relative_coords) * torch.log2(torch.abs(relative_coords) + 1.0) / math.log2(8.0)
        self.register_buffer("relative_coords_table", relative_coords, persistent=False)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        b_windows, n_tokens, channels = x.shape

        qkv = self.qkv(x).reshape(b_windows, n_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        logit_scale = torch.clamp(self.logit_scale, max=math.log(100.0)).exp()
        attn = (q @ k.transpose(-2, -1)) * logit_scale

        relative_position_bias = self.cpb_mlp(self.relative_coords_table)
        relative_position_bias = 16.0 * torch.sigmoid(relative_position_bias)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).unsqueeze(0)
        attn = attn + relative_position_bias

        if attn_mask is not None:
            n_windows = attn_mask.shape[0]
            attn = attn.view(b_windows // n_windows, n_windows, self.num_heads, n_tokens, n_tokens)
            attn = attn + attn_mask.unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n_tokens, n_tokens)

        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_windows, n_tokens, channels)
        x = self.proj(x)
        return self.proj_drop(x)


class SwinV2Block(nn.Module):
    """SwinV2-style block with optional shifted-window attention."""

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 5,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        h, w = input_resolution
        self.input_resolution = input_resolution
        self.window_size = min(window_size, h, w)
        self.shift_size = shift_size
        if self.window_size == min(h, w) and (h <= self.window_size or w <= self.window_size):
            self.shift_size = 0
        if self.shift_size >= self.window_size:
            self.shift_size = 0

        if h % self.window_size != 0 or w % self.window_size != 0:
            raise ValueError(
                f"input_resolution={input_resolution} must be divisible by window_size={self.window_size}"
            )

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttentionV2(
            dim,
            self.window_size,
            num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        if (h, w) != self.input_resolution:
            raise ValueError(f"Expected spatial size {self.input_resolution}, got {(h, w)}")

        shortcut = x
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        x_norm = self.norm1(x_nhwc)

        if self.shift_size > 0:
            shifted_x = torch.roll(x_norm, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = build_shifted_window_mask(self.input_resolution, self.window_size, self.shift_size, x.device)
        else:
            shifted_x = x_norm
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        attn_windows = self.attn(x_windows, attn_mask=attn_mask)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)

        if self.shift_size > 0:
            x_attn = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_attn = shifted_x

        x_res = shortcut.permute(0, 2, 3, 1).contiguous() + x_attn
        x_res = x_res + self.mlp(self.norm2(x_res))
        return x_res.permute(0, 3, 1, 2).contiguous()


class SwinV2Stage(nn.Module):
    """Projection/downsampling followed by alternating regular/shifted SwinV2 blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        depth: int = 2,
        stride: int = 1,
        window_size: int = 5,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                SwinV2Block(
                    dim=out_channels,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if i % 2 == 0 else window_size // 2,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        return x


class SwinV2Encoder(nn.Module):
    """Four-stage hierarchical SwinV2-style encoder for 40x40 pupil maps."""

    def __init__(
        self,
        input_channels: int = 3200,
        base_filters: int = 64,
        depths: Sequence[int] = (2, 2, 2, 2),
        window_size: int = 5,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        if len(depths) != 4:
            raise ValueError("depths must contain four integers")

        nf = base_filters
        self.stage1 = SwinV2Stage(input_channels, nf, (40, 40), 4, depths[0], 1, window_size, dropout, attn_dropout)
        self.stage2 = SwinV2Stage(nf, nf * 2, (20, 20), 4, depths[1], 2, window_size, dropout, attn_dropout)
        self.stage3 = SwinV2Stage(nf * 2, nf * 4, (10, 10), 8, depths[2], 2, window_size, dropout, attn_dropout)
        self.stage4 = SwinV2Stage(nf * 4, nf * 8, (5, 5), 8, depths[3], 2, window_size, dropout, attn_dropout)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        layer1 = self.stage1(x)           # [B, nf, 40, 40]
        layer2 = self.stage2(layer1)      # [B, 2nf, 20, 20]
        layer3 = self.stage3(layer2)      # [B, 4nf, 10, 10]
        bottleneck = self.stage4(layer3)  # [B, 8nf, 5, 5]
        return layer1, layer2, layer3, bottleneck


# =============================================================================
# U-Net decoder and model heads
# =============================================================================


class ConvBNDropGELU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class UNetDecoder(nn.Module):
    """U-Net-style decoder with skip connections."""

    def __init__(self, base_filters: int = 64, dropout: float = 0.1):
        super().__init__()
        nf = base_filters
        self.upconv1 = nn.ConvTranspose2d(nf * 8, nf * 4, kernel_size=2, stride=2)
        self.ex_conv1 = nn.Sequential(
            ConvBNDropGELU(nf * 8, nf * 4, dropout),
            ConvBNDropGELU(nf * 4, nf * 4, dropout),
        )

        self.upconv2 = nn.ConvTranspose2d(nf * 4, nf * 2, kernel_size=2, stride=2)
        self.ex_conv2 = nn.Sequential(
            ConvBNDropGELU(nf * 4, nf * 2, dropout),
            ConvBNDropGELU(nf * 2, nf * 2, dropout),
        )

        self.upconv3 = nn.ConvTranspose2d(nf * 2, nf, kernel_size=2, stride=2)
        self.ex_conv3 = nn.Sequential(
            ConvBNDropGELU(nf * 2, nf, dropout),
            ConvBNDropGELU(nf, nf, dropout),
        )

    def forward(self, layer1: Tensor, layer2: Tensor, layer3: Tensor, bottleneck: Tensor) -> Tensor:
        up1 = self.upconv1(bottleneck)
        cat1 = torch.cat((center_crop_like(layer3, up1), up1), dim=1)
        x = self.ex_conv1(cat1)

        up2 = self.upconv2(x)
        cat2 = torch.cat((center_crop_like(layer2, up2), up2), dim=1)
        x = self.ex_conv2(cat2)

        up3 = self.upconv3(x)
        cat3 = torch.cat((center_crop_like(layer1, up3), up3), dim=1)
        return self.ex_conv3(cat3)


class PhasorHead(nn.Module):
    """Predict a two-channel aberration phasor map."""

    def __init__(
        self,
        in_channels: int,
        output_channels: int = 2,
        normalize_output: bool = True,
        use_na_mask_for_normalization: bool = True,
        size: int = 40,
        nasz: int = 20,
    ):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, output_channels, kernel_size=1, stride=1)
        self.normalize_output = normalize_output
        self.use_na_mask_for_normalization = use_na_mask_for_normalization

        if use_na_mask_for_normalization:
            self.register_buffer("na_mask", make_na_mask(size=size, nasz=nasz), persistent=False)
        else:
            self.na_mask = None

    def forward(self, x: Tensor) -> Tensor:
        y = self.proj(x)
        if self.normalize_output and y.shape[1] == 2:
            mask = self.na_mask if self.use_na_mask_for_normalization else None
            y = normalize_phasor(y, mask=mask)
        return y


# =============================================================================
# Aberrated Reflection-matrix Corrector (ARC) model
# =============================================================================


class FeatureFusion(nn.Module):
    """Fuse corresponding R, RR^dagger, and R^dagger R encoder features."""

    def __init__(self, channels: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, r_feat: Tensor, rrh_feat: Tensor, rhr_feat: Tensor) -> Tensor:
        return self.fuse(torch.cat((r_feat, rrh_feat, rhr_feat), dim=1))


class ARC(nn.Module):
    """Covariance-aware dual-head SwinV2-UNet.

    Inputs:
        r:   R_input tensor,        [B, 3200, 40, 40]
        rrh: RRt_input tensor,      [B, 3200, 40, 40]
        rhr: RtR_input tensor,      [B, 3200, 40, 40]

    Outputs when return_dict=True:
        {
            "output_aberration": [B, 2, 40, 40],
            "input_aberration":  [B, 2, 40, 40],
        }
    """

    def __init__(
        self,
        input_channels: int = 3200,
        output_channels_per_head: int = 2,
        base_filters: int = 64,
        depths: Sequence[int] = (2, 2, 2, 2),
        window_size: int = 5,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        normalize_output: bool = True,
        share_covariance_encoder: bool = True,
        return_dict: bool = True,
    ):
        super().__init__()
        nf = base_filters
        self.return_dict = return_dict
        self.share_covariance_encoder = share_covariance_encoder

        self.r_encoder = SwinV2Encoder(
            input_channels=input_channels,
            base_filters=nf,
            depths=depths,
            window_size=window_size,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )

        self.rrh_encoder = SwinV2Encoder(
            input_channels=input_channels,
            base_filters=nf,
            depths=depths,
            window_size=window_size,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )

        if share_covariance_encoder:
            self.rhr_encoder = self.rrh_encoder
        else:
            self.rhr_encoder = SwinV2Encoder(
                input_channels=input_channels,
                base_filters=nf,
                depths=depths,
                window_size=window_size,
                dropout=dropout,
                attn_dropout=attn_dropout,
            )

        self.fuse1 = FeatureFusion(nf)
        self.fuse2 = FeatureFusion(nf * 2)
        self.fuse3 = FeatureFusion(nf * 4)
        self.fuse4 = FeatureFusion(nf * 8)

        self.decoder = UNetDecoder(base_filters=nf, dropout=dropout)
        self.output_head = PhasorHead(nf, output_channels_per_head, normalize_output=normalize_output)
        self.input_head = PhasorHead(nf, output_channels_per_head, normalize_output=normalize_output)

    def forward(self, r: Tensor, rrh: Tensor, rhr: Tensor) -> Union[Dict[str, Tensor], Tuple[Tensor, Tensor]]:
        r_feats = self.r_encoder(r)
        rrh_feats = self.rrh_encoder(rrh)
        rhr_feats = self.rhr_encoder(rhr)

        f1 = self.fuse1(r_feats[0], rrh_feats[0], rhr_feats[0])
        f2 = self.fuse2(r_feats[1], rrh_feats[1], rhr_feats[1])
        f3 = self.fuse3(r_feats[2], rrh_feats[2], rhr_feats[2])
        f4 = self.fuse4(r_feats[3], rrh_feats[3], rhr_feats[3])

        decoded = self.decoder(f1, f2, f3, f4)
        output_aberration = self.output_head(decoded)
        input_aberration = self.input_head(decoded)

        if self.return_dict:
            return {
                "output_aberration": output_aberration,
                "input_aberration": input_aberration,
            }
        return output_aberration, input_aberration



# =============================================================================
# Shape smoke test -- run directly (`python ARC.py`) to sanity
# check the architecture wires together correctly, with no dataset required.
# =============================================================================


def smoke_test_model(device: torch.device) -> None:
    print("Running model shape smoke test...")
    input_channels = 16
    x = torch.randn(1, input_channels, 40, 40, device=device)
    model = ARC(
        input_channels=input_channels,
        output_channels_per_head=2,
        base_filters=16,
        depths=(1, 1, 1, 1),
        normalize_output=False,
        return_dict=True,
    ).to(device)
    out = model(x, x, x)
    print("  R input:", tuple(x.shape))
    print("  output_aberration:", tuple(out["output_aberration"].shape))
    print("  input_aberration :", tuple(out["input_aberration"].shape))
    print("  params:", f"{sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smoke_test_model(device)
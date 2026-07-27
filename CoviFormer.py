from __future__ import annotations
import math
from typing import Sequence, Optional, Tuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F

def make_na_mask(size: int = 40, nasz: int = 20, device: Optional[torch.device] = None) -> Tensor:
    """Return NA support mask with shape [1, 1, size, size]."""
    coords = torch.arange(1, size + 1, dtype=torch.float32, device=device)
    ky, kx = torch.meshgrid(coords, coords, indexing="ij")
    kc = size / 2.0 + 0.5
    mask = ((kx - kc) ** 2 + (ky - kc) ** 2) < float(nasz**2)
    return mask.float().unsqueeze(0).unsqueeze(0)

def normalize_phasor(x: Tensor, mask: Optional[Tensor] = None, eps: float = 1e-8) -> Tensor:
    """Normalize a two-channel phasor to unit magnitude."""
    if x.shape[1] != 2:
        raise ValueError("normalize_phasor expects exactly 2 channels")
    mag = torch.sqrt(torch.sum(x * x, dim=1, keepdim=True) + eps)
    y = x / mag
    if mask is not None:
        y = y * mask.to(dtype=y.dtype, device=y.device)
    return y

def window_partition(x: Tensor, window_size: int) -> Tensor:
    b, h, w, c = x.shape
    if h % window_size != 0 or w % window_size != 0:
        raise ValueError(f"H={h}, W={w} must be divisible by window_size={window_size}")
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(-1, window_size * window_size, c)

def window_reverse(windows: Tensor, window_size: int, h: int, w: int) -> Tensor:
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
    if shift_size == 0:
        return None
    h, w = input_resolution
    img_mask = torch.zeros((1, h, w, 1), device=device)
    h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
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

# =============================================================================
# NEW: SwinBlock / SwinStage — wires the previously-dead WindowAttentionV2,
# window_partition/reverse, and build_shifted_window_mask into an actual
# forward path (fixes critique #3 "no transformer", #5 "no positional
# encoding", and #6 "Swin blocks are dead code" — all three were really one
# fix, since the position bias lives inside WindowAttentionV2 itself).
# =============================================================================
class SwinBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 5,
        shift_size: int = 0,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttentionV2(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, dim * 4)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, H, W, C]
        b, h, w, c = x.shape
        shortcut = x
        x = self.norm1(x)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        attn_mask = build_shifted_window_mask((h, w), self.window_size, self.shift_size, x.device)
        windows = window_partition(x, self.window_size)
        attn_windows = self.attn(windows, attn_mask=attn_mask)
        x = window_reverse(attn_windows, self.window_size, h, w)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

class SwinStage(nn.Module):
    """One regular + one shifted window-attention block, operating on the
    encoder's [B, C, H, W] feature map (handles the NCHW <-> NHWC conversion
    internally so callers don't need to know about the Swin layout)."""

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int = 8,
        window_size: int = 5,
    ):
        super().__init__()
        self.block1 = SwinBlock(dim, input_resolution, num_heads, window_size=window_size, shift_size=0)
        self.block2 = SwinBlock(dim, input_resolution, num_heads, window_size=window_size, shift_size=window_size // 2)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, H, W] -> [B, H, W, C] for windowed attention, then back
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.block1(x)
        x = self.block2(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class CrossAttentionFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.query_conv = nn.Conv2d(channels, channels // 8, 1)
        self.key_conv = nn.Conv2d(channels, channels // 8, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x_main: Tensor, x_aux: Tensor) -> Tensor:
        B, C, H, W = x_main.shape
        q = self.query_conv(x_main).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key_conv(x_aux).view(B, -1, H * W)
        v = self.value_conv(x_aux).view(B, -1, H * W)

        # FIX (critique #4): q/k have C // 8 channels (from the 1x1 convs
        # above), so the attention scale must be sqrt(C // 8), not sqrt(C).
        # The old sqrt(C) scale over-divided by ~2.83x, over-softening attn.
        attn_dim = q.shape[-1]
        score = torch.bmm(q, k) / (attn_dim ** 0.5)
        attn = F.softmax(score, dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return x_main + self.gamma * out

class CoviFormerEncoder(nn.Module):
    def __init__(self, in_channels: int = 2):
        super().__init__()
        self.stream_r = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        )
        self.stream_rr = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        )
        self.stream_rt = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        )
        self.fusion1 = CrossAttentionFusion(128)
        self.fusion2 = CrossAttentionFusion(128)

    def forward(self, r: Tensor, rr_t: Tensor, r_t_r: Tensor) -> Tensor:
        feat_r = self.stream_r(r)
        feat_rr = self.stream_rr(rr_t)
        feat_rt = self.stream_rt(r_t_r)
        
        fused_1 = self.fusion1(feat_r, feat_rr)
        fused_2 = self.fusion2(fused_1, feat_rt)
        return fused_2

class DecoupledDecoder(nn.Module):
    def __init__(self, channels: int = 128):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoder_o = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )
        self.decoder_i = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def forward(self, x: Tensor) -> dict:
        x_up = self.up(x)
        out_raw = self.decoder_o(x_up)
        in_raw = self.decoder_i(x_up)
        return {"output_raw": out_raw, "input_raw": in_raw}

class PhasorHead(nn.Module):
    """FIX (critique #1): no longer calls atan2 internally. Always projects
    the predicted (cos, sin) onto the unit circle and returns the 2-channel
    phasor directly. Converting to an angle is now the caller's job, done
    only where needed (loss computation / visualization), never inside the
    model. `normalize_output` is no longer an optional flag — normalization
    always happens, since skipping it was the direct path to atan2 blowing
    up near (0, 0), especially under AMP fp16."""

    def __init__(self, size: int = 40, nasz: int = 20):
        super().__init__()
        self.register_buffer("na_mask", make_na_mask(size=size, nasz=nasz), persistent=False)

    def forward(self, y: Tensor) -> dict:
        phasor = y[:, :2, :, :]
        uncertainty = F.softplus(y[:, 2:, :, :])
        phasor = normalize_phasor(phasor, mask=self.na_mask)
        return {"phasor": phasor, "sigma": uncertainty}

class PhysicsPropagationLayer(nn.Module):
    """Phase-gradient smoothness regularizer (critique #2: this was never a
    real physics/propagation loss — it's Total Variation on the phase, and
    is renamed accordingly at the point of use in CoviFormer.forward).

    Rewritten to consume the 2-channel phasor directly instead of a scalar
    angle map, since PhasorHead no longer produces an angle (fix #1). Uses
    the identity cos(a - b) = cos(a)cos(b) + sin(a)sin(b) to measure
    neighbor-to-neighbor phase consistency without ever calling atan2 —
    this keeps the "no atan2 inside the model" rule intact end-to-end and
    is naturally wrap-safe (no branch-cut discontinuity, no ±pi jump)."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _phasor_smoothness(phasor: Tensor) -> Tensor:
        # phasor: [B, 2, H, W], channel 0 = cos(phi), channel 1 = sin(phi)
        cos_p, sin_p = phasor[:, 0:1, :, :], phasor[:, 1:2, :, :]

        cos_x1, sin_x1 = cos_p[..., :, :-1], sin_p[..., :, :-1]
        cos_x2, sin_x2 = cos_p[..., :, 1:], sin_p[..., :, 1:]
        cos_delta_x = cos_x1 * cos_x2 + sin_x1 * sin_x2
        smoothness_x = torch.mean(1.0 - cos_delta_x)

        cos_y1, sin_y1 = cos_p[..., :-1, :], sin_p[..., :-1, :]
        cos_y2, sin_y2 = cos_p[..., 1:, :], sin_p[..., 1:, :]
        cos_delta_y = cos_y1 * cos_y2 + sin_y1 * sin_y2
        smoothness_y = torch.mean(1.0 - cos_delta_y)

        return smoothness_x + smoothness_y

    def forward(self, phasor_o: Tensor, phasor_i: Tensor) -> Tensor:
        return self._phasor_smoothness(phasor_o) + self._phasor_smoothness(phasor_i)

class CoviFormer(nn.Module):
    def __init__(
        self,
        input_channels: int = 2,
    ):
        super().__init__()
        self.encoder = CoviFormerEncoder(in_channels=input_channels)
        # NEW: Swin stage inserted between the conv encoder and the
        # bottleneck (critiques #3, #5, #6). Encoder output is [B, 128, 20, 20]
        # (40x40 input, stride-2 conv halves it) — window_size=5 divides
        # 20 evenly (20 = 4 x 5).
        self.swin_stage = SwinStage(dim=128, input_resolution=(20, 20), num_heads=8, window_size=5)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1)
        )
        self.decoder = DecoupledDecoder(channels=128)
        self.output_head = PhasorHead()
        self.input_head = PhasorHead()
        self.physics_layer = PhysicsPropagationLayer()

    def forward(self, r: Tensor, rrh: Tensor, rhr: Tensor) -> dict:
        x_encoded = self.encoder(r, rrh, rhr)
        x_swin = self.swin_stage(x_encoded)
        x_bottleneck = self.bottleneck(x_swin)
        decoded = self.decoder(x_bottleneck)
        
        output_aberration = self.output_head(decoded["output_raw"])
        input_aberration = self.input_head(decoded["input_raw"])

        # NOTE: "output_aberration" / "input_aberration" below are now the
        # 2-channel (cos, sin) phasors, not scalar angle maps (fix #1). The
        # training script's loss function will need updating to consume
        # phasors instead of angles before this model can be trained again —
        # flagged for our next step, not changed here.
        return {
            "output_aberration": output_aberration["phasor"],
            "output_sigma": output_aberration["sigma"],
            "input_aberration": input_aberration["phasor"],
            "input_sigma": input_aberration["sigma"],
            "smoothness_regularizer": self.physics_layer(output_aberration["phasor"], input_aberration["phasor"])
        }
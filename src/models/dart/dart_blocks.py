import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, to_2tuple, trunc_normal_

from src.models.generic.generic_blocks import PatchEmbed, PatchUnEmbed


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply feature-wise affine modulation using condition-derived shift/scale."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Mlp(nn.Module):
    """Feed-forward network with two linear layers, activation, and dropout.

    Args:
        in_features: Input feature dimension.
        hidden_features: Hidden layer dimension. Defaults to ``in_features``.
        out_features: Output feature dimension. Defaults to ``in_features``.
        act_layer: Activation layer class.
        drop: Dropout rate.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, N, C)``.

        Returns:
            Output tensor of shape ``(B, N, C_out)``.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention (W-MSA) with relative position bias.

    Supports both shifted and non-shifted window configurations.

    Args:
        dim: Number of input channels.
        window_size: Height and width of the attention window.
        num_heads: Number of attention heads.
        qkv_bias: Whether to add learnable bias to query, key, value projections.
        qk_scale: Override default qk scale of ``head_dim ** -0.5``.
        attn_drop: Dropout ratio for attention weights.
        proj_drop: Dropout ratio for output projection.
    """

    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # Define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute window attention with optional mask.

        Args:
            x: Input features of shape ``(num_windows * B, N, C)``.
            mask: Attention mask of shape ``(num_windows, Wh*Ww, Wh*Ww)`` or ``None``.

        Returns:
            Output tensor of shape ``(num_windows * B, N, C)``.
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer block with (shifted) window multi-head self-attention.

    Args:
        dim: Number of input channels.
        input_resolution: Input spatial resolution ``(H, W)``.
        num_heads: Number of attention heads.
        window_size: Window size for local attention.
        shift_size: Shift size for SW-MSA. Set to 0 for W-MSA.
        mlp_ratio: Ratio of MLP hidden dim to embedding dim.
        qkv_bias: Whether to add learnable bias to query, key, value.
        qk_scale: Override default qk scale.
        drop: Dropout rate.
        attn_drop: Attention dropout rate.
        drop_path: Stochastic depth rate.
        act_layer: Activation layer class.
        norm_layer: Normalization layer class.
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        use_adaln_zero: bool = False,
        cond_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_adaln_zero = use_adaln_zero

        if min(self.input_resolution) <= self.window_size:
            # If window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        if not (0 <= self.shift_size < self.window_size):
            raise ValueError("shift_size must be in the range of 0 to window_size")

        if self.use_adaln_zero and cond_dim is None:
            raise ValueError("cond_dim must be provided when use_adaln_zero=True")

        if self.use_adaln_zero and norm_layer is nn.LayerNorm:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        else:
            self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if self.use_adaln_zero and norm_layer is nn.LayerNorm:
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        else:
            self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if self.use_adaln_zero:
            assert cond_dim is not None
            cond_dim_int = int(cond_dim)
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_dim_int, 6 * dim, bias=True),
            )
            adaln_linear = self.adaLN_modulation[-1]
            if isinstance(adaln_linear, nn.Linear):
                adaln_linear._skip_global_init = True  # type: ignore[attr-defined]
                nn.init.constant_(adaln_linear.weight, 0)
                nn.init.constant_(adaln_linear.bias, 0)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size: tuple[int, int]) -> torch.Tensor:
        """Calculate attention mask for shifted window MSA.

        Regions that should not attend to each other (across window boundaries
        after cyclic shift) are masked with ``-100.0``.

        Args:
            x_size: Spatial dimensions ``(H, W)``.

        Returns:
            Attention mask of shape ``(num_windows, Wh*Ww, Wh*Ww)``.
        """
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = self.window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
            attn_mask == 0, 0.0
        )
        return attn_mask

    def window_partition(self, x: torch.Tensor, window_size: int) -> torch.Tensor:
        """Partition feature map into non-overlapping windows.

        Args:
            x: Input tensor of shape ``(B, H, W, C)``.
            window_size: Size of each square window.

        Returns:
            Windows tensor of shape ``(num_windows * B, window_size, window_size, C)``.
        """
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = (
            x.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, window_size, window_size, C)
        )
        return windows

    def window_reverse(
        self, windows: torch.Tensor, window_size: int, H: int, W: int
    ) -> torch.Tensor:
        """Reverse window partition back to a feature map.

        Args:
            windows: Windows tensor of shape
                ``(num_windows * B, window_size, window_size, C)``.
            window_size: Size of each square window.
            H: Height of the feature map.
            W: Width of the feature map.

        Returns:
            Feature map tensor of shape ``(B, H, W, C)``.
        """
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(
            B, H // window_size, W // window_size, window_size, window_size, -1
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def forward(
        self,
        x: torch.Tensor,
        x_size: tuple[int, int],
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with (shifted) window attention and residual MLP.

        Args:
            x: Input tensor of shape ``(B, H*W, C)``.
            x_size: Spatial dimensions ``(H, W)``.
            cond: Optional conditioning tensor of shape ``(B, cond_dim)``.

        Returns:
            Output tensor of shape ``(B, H*W, C)``.
        """
        H, W = x_size
        B, L, C = x.shape

        shortcut = x

        shift_mlp = scale_mlp = gate_msa = gate_mlp = None
        if self.use_adaln_zero:
            if cond is None:
                raise ValueError(
                    "cond must be provided when use_adaln_zero=True in SwinTransformerBlock"
                )
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = self.adaLN_modulation(cond).chunk(6, dim=1)
            x = modulate(self.norm1(x), shift_msa, scale_msa)
        else:
            x = self.norm1(x)

        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        # Partition windows
        x_windows = self.window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn(
                x_windows, mask=self.calculate_mask(x_size).to(x.device)
            )

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = self.window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        if self.use_adaln_zero:
            assert gate_msa is not None and gate_mlp is not None
            assert shift_mlp is not None and scale_mlp is not None
            x = shortcut + gate_msa.unsqueeze(1) * self.drop_path(x)
            x = x + gate_mlp.unsqueeze(1) * self.drop_path(
                self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
            )
        else:
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """Basic Swin Transformer layer consisting of multiple transformer blocks.

    Alternates between regular window MSA (W-MSA) and shifted window MSA
    (SW-MSA) across consecutive blocks.

    Args:
        dim: Number of input channels.
        input_resolution: Input spatial resolution ``(H, W)``.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        window_size: Local window size.
        mlp_ratio: Ratio of MLP hidden dim to embedding dim.
        qkv_bias: Whether to add learnable bias to query, key, value.
        qk_scale: Override default qk scale.
        drop: Dropout rate.
        attn_drop: Attention dropout rate.
        drop_path: Stochastic depth rate (single value or per-block list).
        norm_layer: Normalization layer class.
        downsample: Downsample layer class applied at the end of the stage.
        use_checkpoint: Whether to use gradient checkpointing.
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        downsample: type[nn.Module] | None = None,
        use_checkpoint: bool = False,
        use_adaln_zero: bool = False,
        cond_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    use_adaln_zero=use_adaln_zero,
                    cond_dim=cond_dim,
                )
                for i in range(depth)
            ]
        )

        # Patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer
            )
        else:
            self.downsample = None

    def forward(
        self,
        x: torch.Tensor,
        x_size: tuple[int, int],
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through all transformer blocks.

        Args:
            x: Input tensor of shape ``(B, H*W, C)``.
            x_size: Spatial dimensions ``(H, W)``.
            cond: Optional conditioning tensor of shape ``(B, cond_dim)``.

        Returns:
            Output tensor of shape ``(B, H*W, C)``.
        """
        for blk in self.blocks:
            if self.use_checkpoint:

                def _blk_forward(inp: torch.Tensor) -> torch.Tensor:
                    return blk(inp, x_size, cond)

                checkpointed = checkpoint.checkpoint(_blk_forward, x)
                if not isinstance(checkpointed, torch.Tensor):
                    raise TypeError("Checkpointed block output must be a Tensor")
                x = checkpointed
            else:
                x = blk(x, x_size, cond)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class ResidualSwinTransformerBlock(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Wraps a :class:`BasicLayer` with a residual connection through a
    convolutional block and patch embedding / unembedding.

    Args:
        dim: Number of input channels.
        input_resolution: Input spatial resolution ``(H, W)``.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        window_size: Window size.
        mlp_ratio: Ratio of MLP hidden dim to embedding dim.
        qkv_bias: Whether to add learnable bias to query, key, value.
        qk_scale: Override default qk scale.
        drop: Dropout rate.
        attn_drop: Attention dropout rate.
        drop_path: Stochastic depth rate (single value or per-block list).
        norm_layer: Normalization layer class.
        downsample: Downsample layer class.
        use_checkpoint: Whether to use gradient checkpointing.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: Residual convolution type (``'1conv'`` or ``'3conv'``).
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        downsample: type[nn.Module] | None = None,
        use_checkpoint: bool = False,
        img_size: int = 256,
        patch_size: int = 4,
        resi_connection: str = "1conv",
        use_adaln_zero: bool = False,
        cond_dim: int | None = None,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            use_adaln_zero=use_adaln_zero,
            cond_dim=cond_dim,
        )

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,
            embed_dim=dim,
            norm_layer=None,
        )

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,
            embed_dim=dim,
            norm_layer=None,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_size: tuple[int, int],
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor of shape ``(B, H*W, C)``.
            x_size: Spatial dimensions ``(H, W)``.
            cond: Optional conditioning tensor of shape ``(B, cond_dim)``.

        Returns:
            Output tensor of shape ``(B, H*W, C)``.
        """
        residual_map = self.patch_unembed(self.residual_group(x, x_size, cond), x_size)
        return self.patch_embed(self.conv(residual_map)) + x


class SwinSpatial(nn.Module):
    """
    Swin restoration block of DART, adapted from SwinIR (Liang et al., ICCVW 2021,
    https://github.com/JingyunLiang/SwinIR) with optional AdaLN-Zero conditioning
    of every Swin Transformer block.

    Args:
        img_size: Input image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Patch embedding dimension.
        depths: Number of transformer blocks in each RSTB stage.
        num_heads: Number of attention heads in each RSTB stage.
        window_size: Window size for local attention.
        mlp_ratio: Ratio of MLP hidden dim to embedding dim.
        qkv_bias: Whether to add learnable bias to query, key, value.
        qk_scale: Override default qk scale.
        drop_rate: Dropout rate.
        attn_drop_rate: Attention dropout rate.
        drop_path_rate: Stochastic depth rate.
        norm_layer: Normalization layer class.
        ape: Whether to use absolute position embedding.
        patch_norm: Whether to add normalization after patch embedding.
        use_checkpoint: Whether to use gradient checkpointing.
        upscale: Upscale factor for reconstruction.
        upsampler: Reconstruction module type (``'nearest+conv'`` or ``None``).
        resi_connection: Residual convolution type (``'1conv'`` or ``'3conv'``).
        use_adaln_zero: Whether to modulate every Swin block with AdaLN-Zero.
        cond_dim: Size of the AdaLN condition vector.
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 1,
        in_chans: int = 3 + 16,
        embed_dim: int = 64,
        depths: tuple[int, ...] = (2, 2),
        num_heads: tuple[int, ...] = (4, 4),
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        ape: bool = False,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        upscale: int = 2,
        upsampler: str | None = "nearest+conv",
        resi_connection: str = "1conv",
        use_adaln_zero: bool = False,
        cond_dim: int | None = None,
    ) -> None:
        super().__init__()
        num_in_ch = in_chans

        self.upsampler = upsampler
        self.upscale = upscale
        self.use_adaln_zero = use_adaln_zero
        self.cond_dim = cond_dim

        # ------------------------- 1, shallow feature extraction ------------------------- #
        # Downsample input spatial size into original 1/2
        self.conv_first = nn.Conv2d(num_in_ch, int(embed_dim / 2), 3, 1, 1)
        self.conv_second = nn.Conv2d(int(embed_dim / 2), embed_dim, 4, 2, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # Split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=int(img_size / 2),
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # Merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=int(img_size / 2),
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        # Absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = ResidualSwinTransformerBlock(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=int(img_size / 2),
                patch_size=patch_size,
                resi_connection=resi_connection,
                use_adaln_zero=use_adaln_zero,
                cond_dim=cond_dim,
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        # Build the last conv layer in deep feature extraction
        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1),
            )

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == "nearest+conv":
            # For real-world SR (less artifacts)
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
            )
            self.conv_up1 = nn.Conv2d(embed_dim, int(embed_dim / 2), 3, 1, 1)
            self.conv_last = nn.Conv2d(int(embed_dim / 2), int(embed_dim / 4), 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights for Linear and LayerNorm modules.

        Args:
            m: Module to initialize.
        """
        if getattr(m, "_skip_global_init", False):
            return

        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        """Parameters excluded from weight decay."""
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self) -> set[str]:
        """Parameter name keywords excluded from weight decay."""
        return {"relative_position_bias_table"}

    def forward_features(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract deep features through patch embedding and RSTB layers.

        Args:
            x: Input feature map of shape ``(B, C, H, W)``.
            cond: Optional conditioning tensor of shape ``(B, cond_dim)``.

        Returns:
            Output feature map of shape ``(B, C, H, W)``.
        """
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)

        if self.ape:
            x = x + self.absolute_pos_embed

        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size, cond)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)

        return x

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the full SwinSpatial pipeline.

        Args:
            x: Input feature tensor of shape ``(B, C, H, W)``.
            cond: Optional conditioning tensor of shape ``(B, cond_dim)``.

        Returns:
            Restored feature tensor.
        """
        if self.upsampler == "nearest+conv":
            # For real-world SR
            x1 = self.lrelu(self.conv_first(x))
            x = self.conv_second(x1)
            x = self.conv_after_body(self.forward_features(x, cond)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(
                self.conv_up1(
                    F.interpolate(x, scale_factor=self.upscale, mode="nearest")
                )
                + x1
            )
            x = self.conv_last(x)

        else:
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first, cond)) + x_first
            x = x + self.conv_last(res)

        return x

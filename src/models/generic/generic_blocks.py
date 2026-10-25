import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import to_2tuple


class PatchEmbed(nn.Module):
    """
    Transfer 2D feature map into 1D token sequence.

    Args:
        img_size: Image size
        patch_size: Patch size
        in_chans: Input channels
        embed_dim: Embedding dimension
        norm_layer: Normalization layer (optional)
    """

    def __init__(
        self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    """
    Return 2D feature map from 1D token sequence.

    Args:
        img_size: Image size
        patch_size: Patch size
        in_chans: Input channels
        embed_dim: Embedding dimension
        norm_layer: Normalization layer (optional)
    """

    def __init__(
        self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])
        return x


class PSUpsample(nn.Module):
    """
    Upsampling with Pixel-Shuffle.

    Args:
        in_feat: Input features
        out_feat: Output features
        scale_factor: Upscaling factor
    """

    def __init__(self, in_feat: int, out_feat: int, scale_factor: int):
        super(PSUpsample, self).__init__()

        self.scale_factor = scale_factor
        self.up_conv = nn.Conv2d(
            in_feat,
            out_feat * scale_factor * scale_factor,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_conv(x)
        return F.pixel_shuffle(x, upscale_factor=self.scale_factor)

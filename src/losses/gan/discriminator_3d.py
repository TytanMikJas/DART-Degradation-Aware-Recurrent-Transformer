import torch
import torch.nn as nn


def spectral_norm(module: nn.Module, mode: bool = True) -> nn.Module:
    """
    Conditionally apply spectral normalization.

    Args:
        module: Module to normalize
        mode: Whether to apply spectral norm

    Returns:
        Normalized module or original module
    """
    if mode:
        return nn.utils.spectral_norm(module)
    return module


class BaseNetwork(nn.Module):
    """Base network class with initialization utilities."""

    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type: str = "normal", gain: float = 0.02):
        """
        Initialize network weights.

        Based on https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39

        Args:
            init_type: Initialization type ('normal', 'xavier', 'kaiming', 'orthogonal', 'none')
            gain: Gain for initialization
        """

        def init_func(m):
            classname = m.__class__.__name__

            if classname.find("InstanceNorm2d") != -1:
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, "weight") and (
                classname.find("Conv") != -1 or classname.find("Linear") != -1
            ):
                if init_type == "normal":
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == "xavier":
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == "kaiming":
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == "none":
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        f"Initialization method [{init_type}] is not implemented"
                    )

                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # Propagate to children
        for m in self.children():
            if hasattr(m, "init_weights"):
                m.init_weights(init_type, gain)


class Discriminator3D(BaseNetwork):
    """
    3D Convolutional Discriminator for video sequences.

    Uses 3D convolutions to process temporal information directly, making it
    suitable for video-based adversarial training.

    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        base_channels: Base number of channels (default: 64)
        use_spectral_norm: Whether to use spectral normalization (default: True)
        learning_rate: Learning rate for discriminator optimizer (default: 0.0002)
        init_weights: Whether to initialize weights on construction (default: True)
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        use_spectral_norm: bool = True,
        use_sigmoid: bool = False,
        learning_rate: float = 0.0002,
        init_weights: bool = True,
    ):
        super(Discriminator3D, self).__init__()

        self.use_sigmoid = use_sigmoid
        self.learning_rate = learning_rate
        nf = base_channels

        # 3D Convolutional layers
        # Temporal kernel size of 3, spatial kernel of 5x5
        # Stride (1, 2, 2) to downsample spatially but maintain temporal resolution
        self.conv = nn.Sequential(
            spectral_norm(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=nf * 1,
                    kernel_size=(3, 5, 5),
                    stride=(1, 2, 2),
                    padding=1,
                    bias=not use_spectral_norm,
                ),
                use_spectral_norm,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(
                    nf * 1,
                    nf * 2,
                    kernel_size=(3, 5, 5),
                    stride=(1, 2, 2),
                    padding=(1, 2, 2),
                    bias=not use_spectral_norm,
                ),
                use_spectral_norm,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(
                    nf * 2,
                    nf * 2,
                    kernel_size=(3, 5, 5),
                    stride=(1, 2, 2),
                    padding=(1, 2, 2),
                    bias=not use_spectral_norm,
                ),
                use_spectral_norm,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(
                    nf * 2,
                    nf * 2,
                    kernel_size=(3, 5, 5),
                    stride=(1, 2, 2),
                    padding=(1, 2, 2),
                    bias=not use_spectral_norm,
                ),
                use_spectral_norm,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(
                nf * 2,
                nf * 2,
                kernel_size=(3, 5, 5),
                stride=(1, 2, 2),
                padding=(1, 2, 2),
            ),
        )

        if init_weights:
            self.init_weights()

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through discriminator.

        Args:
            xs: Input video tensor (B, T, C, H, W)

        Returns:
            Discriminator logits (B, T, C', H', W')
        """
        # Transpose from (B, T, C, H, W) to (B, C, T, H, W) for Conv3d
        xs_t = torch.transpose(xs, 1, 2)

        # Apply 3D convolutions
        feat = self.conv(xs_t)
        if self.use_sigmoid:
            feat = torch.sigmoid(feat)

        # Transpose back to (B, T, C', H', W')
        out = torch.transpose(feat, 1, 2)
        return out

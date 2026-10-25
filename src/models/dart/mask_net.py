import torch
import torch.nn as nn


class DilationPyramidMaskNet(nn.Module):
    """Dilation Pyramid MaskNet: a shared stem, three parallel branches with
    dilation rates 1, 2 and 3, and a fused head that predicts mask logits."""

    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels, hidden_channels, kernel_size=3, padding=1, bias=True
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.branch_d1 = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                dilation=1,
                bias=True,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.branch_d2 = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=True,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.branch_d3 = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=3,
                dilation=3,
                bias=True,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(
                hidden_channels * 3,
                hidden_channels,
                kernel_size=1,
                padding=0,
                bias=True,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        p1 = self.branch_d1(feat)
        p2 = self.branch_d2(feat)
        p3 = self.branch_d3(feat)
        return self.fuse(torch.cat([p1, p2, p3], dim=1))


class MaskNet(nn.Module):
    """Predicts the soft defect mask logits used by the gated temporal fusion.

    The input is the concatenation of the aligned hidden state, the encoded
    current frame, the residual indicator and (optionally) the warped mask of the
    previous step. Two heads are available:

    * ``dilation_pyramid=True``: the multi-scale Dilation Pyramid MaskNet of DART.
    * ``dilation_pyramid=False``: the shallow two-layer predictor of RTN, used for
      the "- MaskNet" ablation.
    """

    def __init__(
        self, hidden_channels: int, use_pre_mask: bool, dilation_pyramid: bool = True
    ):
        """Creates a MaskNet.

        Args:
            hidden_channels (int): hidden channels of the Gated Aggregation module.
            use_pre_mask (bool): whether the previous (warped) mask is an extra input.
            dilation_pyramid (bool): use the Dilation Pyramid head (DART) instead of
                the shallow RTN head.
        """
        super(MaskNet, self).__init__()
        # hidden state, latent feature, residual indicator + optional previous mask
        self.hidden_channels = hidden_channels * 2 + 1 + (1 if use_pre_mask else 0)
        self.use_pre_mask = use_pre_mask
        self.net = (
            self._build_dilation_pyramid_mask_net()
            if dilation_pyramid
            else self._build_rtn_mask_net()
        )

    def _build_rtn_mask_net(self) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(
                in_channels=self.hidden_channels,
                out_channels=int(self.hidden_channels / 2),
                kernel_size=3,
                padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=int(self.hidden_channels / 2),
                out_channels=1,
                kernel_size=3,
                padding=1,
            ),
        )

    def _build_dilation_pyramid_mask_net(self) -> nn.Module:
        hidden = max(16, int(self.hidden_channels / 2))
        return DilationPyramidMaskNet(self.hidden_channels, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MaskNet.

        Args:
            x: concatenated hidden state, latent feature, residual indicator and
                optionally the previous mask, shape (B, in_channels, H, W).

        Returns:
            Mask logits of shape (B, 1, H, W).
        """
        return self.net(x)

import torch
from torch import nn


class Gated_Aggregation(nn.Module):
    """
    Mask-gated temporal fusion (Eq. 2 of the paper).

    Encodes the current frame, predicts the soft defect mask with ``masknet`` and
    blends the aligned hidden state with the encoded frame:
    ``F_t = E(x_t) * (1 - M_t) + s_{t-1}^align * M_t``.

    Args:
        masknet: MaskNet predicting the mask logits
        hidden_channels: Number of hidden channels
        kernel_size: Convolution kernel size of the frame encoder
        in_channels: Number of input image channels
        use_pre_mask: Whether the warped mask of the previous step is a MaskNet input
        stride: Convolution stride
        padding: Convolution padding
        dilation: Convolution dilation
        bias: Whether to use bias
        activation: Activation function (default: LeakyReLU)
    """

    def __init__(
        self,
        masknet: nn.Module,
        hidden_channels: int,
        kernel_size: int,
        in_channels: int = 3,
        use_pre_mask: bool = False,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
        activation: nn.Module = nn.LeakyReLU(0.2, inplace=True),
    ):
        super(Gated_Aggregation, self).__init__()

        self.activation = activation
        self.use_pre_mask = use_pre_mask
        self.conv2d_projection_head = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.masknet = masknet
        self.init_weights()

    def init_weights(self):
        """Initialize weights using Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(
        self,
        hidden_state: torch.Tensor,
        curr_lr: torch.Tensor,
        residual_indicator: torch.Tensor,
        pre_mask: torch.Tensor | None = None,
        head: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Forward pass for gated aggregation.

        Args:
            hidden_state: Aligned hidden state from propagation
            curr_lr: Current degraded frame
            residual_indicator: Indicator of temporal inconsistency
            pre_mask: Warped mask of the previous step (used only if use_pre_mask=True)
            head: Whether this is the first frame of the propagation direction

        Returns:
            3-tuple ``(output, mask, mask_logits)``. ``mask`` is the soft mask
            ``sigmoid(mask_logits)``. ``mask_logits`` is the raw pre-sigmoid MaskNet
            output ``(B,1,H,W)`` for non-head frames, or ``None`` for head frames
            (no temporal gate exists yet, so they are not supervised).
        """
        latent_feature = self.conv2d_projection_head(curr_lr)

        if self.use_pre_mask:
            if pre_mask is None:
                raise ValueError("use_pre_mask is True but pre_mask is None")
            gate_input = torch.cat(
                [hidden_state, latent_feature, residual_indicator, pre_mask], dim=1
            )
        else:
            gate_input = torch.cat(
                [hidden_state, latent_feature, residual_indicator], dim=1
            )

        mask_logits = self.masknet(gate_input)
        x = torch.sigmoid(mask_logits)

        if head:
            return latent_feature, x, None
        aggregated = x * hidden_state + (1 - x) * latent_feature
        return aggregated, x, mask_logits

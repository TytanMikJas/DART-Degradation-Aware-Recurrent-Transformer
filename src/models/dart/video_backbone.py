import torch
import torch.nn as nn

from src.models.dart.dart_blocks import SwinSpatial
from src.models.dart.temporal_modules import Gated_Aggregation
from src.models.generic.generic_blocks import PSUpsample
from src.models.generic.utils import flow_warp
from src.models.optical_flow.sequence_optical_flow import SequenceOpticalFlow


class AdaLNConditionEncoder(nn.Module):
    """Condition Encoder A(M_t, R_t) of the AdaLN-Zero degradation conditioning.

    Three alternating Conv/SiLU layers over the concatenated mask and residual
    indicator, a global average pool and a two-layer MLP produce the condition
    vector that modulates every AdaLN Swin block.
    """

    def __init__(self, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )
        self.to_cond = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        learned_mask: torch.Tensor,
        residual_indicator: torch.Tensor,
    ) -> torch.Tensor:
        cond_input = torch.cat([learned_mask, residual_indicator], dim=1)
        return self.to_cond(self.encoder(cond_input).mean(dim=(2, 3)))


class VideoBackbone(nn.Module):
    """
    DART restoration network: bidirectional recurrent propagation with mask-gated
    temporal fusion and AdaLN-Zero conditioned Swin restoration blocks.

    Args:
        num_feat: Base number of feature channels (default: 16)
        out_channels: Number of output channels (default: 3)
        depths: Depth of each Transformer-based SwinSpatial layer (default: [2, 2, 2])
        upscale: Upscale factor for output (default: 1)
        use_pre_mask: Recurrently propagate the previous soft mask into MaskNet
        use_adaln_zero: Condition the Swin blocks on A(M_t, R_t) via AdaLN-Zero
        expose_gate_mask_logits: Keep the raw MaskNet logits of the last forward
            pass (``last_gate_mask_logits_{forward,backward}``) for mask supervision
        cond_dim: AdaLN condition vector size
        cond_hidden_dim: Hidden width of the condition encoder
        optical_flow: Optical flow network handler
        forward_resblocks: SwinSpatial blocks for forward propagation
        backward_resblocks: SwinSpatial blocks for backward propagation
        forward_aggregation: Gated aggregation for forward propagation
        backward_aggregation: Gated aggregation for backward propagation
    """

    def __init__(
        self,
        num_feat: int = 16,
        out_channels: int = 3,
        depths: list[int] | None = None,
        upscale: int = 1,
        use_pre_mask: bool = False,
        use_adaln_zero: bool = False,
        expose_gate_mask_logits: bool = False,
        cond_dim: int = 64,
        cond_hidden_dim: int = 64,
        optical_flow: SequenceOpticalFlow | None = None,
        forward_resblocks: SwinSpatial | None = None,
        backward_resblocks: SwinSpatial | None = None,
        forward_aggregation: Gated_Aggregation | None = None,
        backward_aggregation: Gated_Aggregation | None = None,
    ):
        super(VideoBackbone, self).__init__()

        if depths is None:
            raise ValueError("Missing VideoBackbone configuration.")
        if forward_resblocks is None or backward_resblocks is None:
            raise ValueError("Missing VideoBackbone Swin blocks.")
        if forward_aggregation is None or backward_aggregation is None:
            raise ValueError("Missing VideoBackbone aggregation blocks.")

        self.num_feat = num_feat
        self.optical_flow = optical_flow
        self.use_pre_mask = use_pre_mask
        self.use_adaln_zero = use_adaln_zero
        self.expose_gate_mask_logits = expose_gate_mask_logits
        self.cond_dim = cond_dim

        self.condition_encoder: AdaLNConditionEncoder | None = (
            AdaLNConditionEncoder(output_dim=cond_dim, hidden_dim=cond_hidden_dim)
            if use_adaln_zero
            else None
        )

        # Bidirectional Propagation with swin transformer blocks
        self.forward_resblocks: SwinSpatial = forward_resblocks
        self.backward_resblocks: SwinSpatial = backward_resblocks

        # Concatenate Aggregation
        self.concate = nn.Conv2d(
            num_feat * 2, num_feat, kernel_size=3, stride=1, padding=1, bias=True
        )

        # Hidden State Aggregation (Gated)
        self.Forward_Aggregation: Gated_Aggregation = forward_aggregation
        self.Backward_Aggregation: Gated_Aggregation = backward_aggregation

        # Upsampling
        self.up1 = PSUpsample(num_feat, num_feat, scale_factor=upscale)
        self.up2 = PSUpsample(num_feat, num_feat, scale_factor=1)

        # Output layers
        self.conv_hr = nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1)
        self.conv_last = nn.Conv2d(
            num_feat, out_channels, kernel_size=3, stride=1, padding=1
        )

        # Global Residual Learning
        self.img_up = nn.Upsample(
            scale_factor=upscale, mode="bilinear", align_corners=False
        )

        # Activation Function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.last_gate_mask_logits_forward: torch.Tensor | None = None
        self.last_gate_mask_logits_backward: torch.Tensor | None = None

    def _build_condition(
        self,
        learned_mask: torch.Tensor,
        residual_indicator: torch.Tensor,
    ) -> torch.Tensor | None:
        """Global degradation condition A(M_t, R_t), or None without AdaLN-Zero."""
        if self.condition_encoder is None:
            return None
        return self.condition_encoder(
            learned_mask=learned_mask, residual_indicator=residual_indicator
        )

    def comp_flow(
        self, degraded_frames: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute optical flow using SequenceOpticalFlow for feature warping.

        Args:
            degraded_frames: LR/degraded frames with shape (n, t, c, h, w)

        Returns:
            Tuple of (forward_flow, backward_flow):
                - forward_flow: Flow from frame i+1 to frame i (n, t-1, 2, h, w)
                - backward_flow: Flow from frame i to frame i+1 (n, t-1, 2, h, w)
        """
        if self.optical_flow is None:
            # No optical flow - return zero flows
            n, t, _, h, w = degraded_frames.size()
            forward_flow = degraded_frames.new_zeros(n, t - 1, 2, h, w)
            backward_flow = degraded_frames.new_zeros(n, t - 1, 2, h, w)
            return forward_flow, backward_flow

        return self.optical_flow(degraded_frames)

    def forward(self, degraded_frames: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through video backbone.

        Args:
            degraded_frames: Degraded frames (n, t, c, h, w) in [-1, 1]

        Returns:
            Restored frames (n, t, c, h, w) in [-1, 1]
        """
        n, t, c, h, w = degraded_frames.size()

        if h < 64 or w < 64:
            raise ValueError(f"Height and width must be at least 64, got {h} and {w}.")

        forward_flow, backward_flow = self.comp_flow(degraded_frames)

        # Backward Propagation
        rlt = []
        backward_gate_logits: list[torch.Tensor] = []
        feat_prop = degraded_frames.new_zeros(n, self.num_feat, h, w)
        pre_mask = degraded_frames.new_full((n, 1, h, w), 0.5)

        for i in range(t - 1, -1, -1):
            curr_lr = degraded_frames[:, i, :, :, :]

            if i < t - 1:
                flow = backward_flow[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

                pixel_prop = flow_warp(
                    degraded_frames[:, i + 1, :, :, :], flow.permute(0, 2, 3, 1)
                )
                residual_indicator = torch.abs(
                    pixel_prop[:, :1, :, :] - curr_lr[:, :1, :, :]
                )

                if self.use_pre_mask:
                    pre_mask = flow_warp(
                        pre_mask,
                        flow.permute(0, 2, 3, 1),
                        interp_mode="nearest",
                    )

                feat_prop, learned_mask, gate_logits = self.Backward_Aggregation(
                    feat_prop,
                    curr_lr,
                    residual_indicator,
                    pre_mask=pre_mask if self.use_pre_mask else None,
                    head=False,
                )
            else:
                # Last frame in time (backward head): no feat from future neighbor
                residual_indicator = degraded_frames.new_zeros(n, 1, h, w)
                feat_prop, learned_mask, gate_logits = self.Backward_Aggregation(
                    feat_prop,
                    curr_lr,
                    residual_indicator,
                    pre_mask=pre_mask if self.use_pre_mask else None,
                    head=True,
                )
            cond = self._build_condition(learned_mask, residual_indicator)
            feat_prop = self.backward_resblocks(feat_prop, cond=cond)

            rlt.append(feat_prop)
            if self.expose_gate_mask_logits and gate_logits is not None:
                backward_gate_logits.append(gate_logits)
            if self.use_pre_mask:
                pre_mask = learned_mask

        rlt = rlt[::-1]  # Reverse to chronological order
        backward_gate_logits = backward_gate_logits[::-1]

        # Forward Propagation
        feat_prop = degraded_frames.new_zeros(n, self.num_feat, h, w)
        forward_gate_logits: list[torch.Tensor] = []
        pre_mask = degraded_frames.new_full((n, 1, h, w), 0.5)

        for i in range(0, t):
            curr_lr = degraded_frames[:, i, :, :, :]

            if i > 0:
                flow = forward_flow[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

                pixel_prop = flow_warp(
                    degraded_frames[:, i - 1, :, :, :], flow.permute(0, 2, 3, 1)
                )
                residual_indicator = torch.abs(
                    pixel_prop[:, :1, :, :] - curr_lr[:, :1, :, :]
                )
                if self.use_pre_mask:
                    pre_mask = flow_warp(
                        pre_mask,
                        flow.permute(0, 2, 3, 1),
                        interp_mode="nearest",
                    )
                feat_prop, learned_mask, gate_logits = self.Forward_Aggregation(
                    feat_prop,
                    curr_lr,
                    residual_indicator,
                    pre_mask=pre_mask if self.use_pre_mask else None,
                    head=False,
                )
            else:
                # First frame in time (forward head): no feat from past neighbor
                residual_indicator = degraded_frames.new_zeros(n, 1, h, w)
                feat_prop, learned_mask, gate_logits = self.Forward_Aggregation(
                    feat_prop,
                    curr_lr,
                    residual_indicator,
                    pre_mask=pre_mask if self.use_pre_mask else None,
                    head=True,
                )
            cond = self._build_condition(learned_mask, residual_indicator)
            feat_prop = self.forward_resblocks(feat_prop, cond=cond)

            # Fusion and Upsampling
            cat_feat = torch.cat([rlt[i], feat_prop], dim=1)
            sr_rlt = self.lrelu(self.concate(cat_feat))
            sr_rlt = self.lrelu(self.up1(sr_rlt))
            sr_rlt = self.lrelu(self.up2(sr_rlt))
            sr_rlt = self.lrelu(self.conv_hr(sr_rlt))
            sr_rlt = self.conv_last(sr_rlt)

            # Global Residual Learning
            base = self.img_up(curr_lr)
            sr_rlt += base

            # Output activation (tanh to keep in [-1, 1] range)
            rlt[i] = torch.tanh(sr_rlt)
            if self.expose_gate_mask_logits and gate_logits is not None:
                forward_gate_logits.append(gate_logits)
            if self.use_pre_mask:
                pre_mask = learned_mask

        self.last_gate_mask_logits_backward = (
            torch.stack(backward_gate_logits, dim=1) if backward_gate_logits else None
        )
        self.last_gate_mask_logits_forward = (
            torch.stack(forward_gate_logits, dim=1) if forward_gate_logits else None
        )

        return torch.stack(rlt, dim=1)

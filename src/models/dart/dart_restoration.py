import warnings
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from src.models.generic.sequence_adversarial_restoration import (
    SequenceAdversarialRestorationModel,
)


@dataclass
class DegradationAwareLossConfig:
    """Loss weights of the DART training objective (Sec. 3.5, Eq. 10).

    The generator loss is
        L = g_adv_loss
          + lambda_rec * (L1 + perceptual)
          + lambda_mask_sup * (BCE + dice_weight * Dice)
    where the adversarial weight lives in ``AdversarialLoss.generator_weight``.
    ``lambda_mask_sup = 0`` disables mask supervision (the "- mask supervision"
    ablation).

    Attributes:
        lambda_rec: Weight of the reconstruction loss (pixel L1 + perceptual).
        lambda_mask_sup: Weight of the mask supervision loss (lambda_mask).
        dice_weight: Weight of the Dice term inside the mask loss (lambda_dice).
        pos_weight_clamp_min/max: Clamp (w_min, w_max) of the class-balancing
            positive weight N_neg / N_pos of the BCE term.
        epsilon: Numerical stability epsilon for divisions and clamps.
    """

    lambda_rec: float = 1.0
    lambda_mask_sup: float = 0.0
    dice_weight: float = 0.5
    pos_weight_clamp_min: float = 1.0
    pos_weight_clamp_max: float = 100.0
    epsilon: float = 1e-6


class DARTAdversarialRestorationModel(SequenceAdversarialRestorationModel):
    """
    DART restoration model with adversarial training and direct mask supervision.

    Args:
        optical_flow: SequenceOpticalFlow instance
        discriminator: Discriminator3D instance
        denormalizer: Denormalizer instance for perceptual loss
        net: VideoBackbone instance
        adversarial_loss: AdversarialLoss instance
        loss: WeightedMultiLoss instance
        generator_optimizer_config: OptimizerConfig instance
        discriminator_optimizer_config: OptimizerConfig instance
    """

    def __init__(
        self,
        *args,
        degradation_aware_loss_config: DegradationAwareLossConfig | None = None,
        **kwargs,
    ):
        """Initialize DART restoration model with optional degradation-aware loss weights."""
        super().__init__(*args, **kwargs)
        self.degradation_aware_loss_config = (
            degradation_aware_loss_config or DegradationAwareLossConfig()
        )
        self._last_aux_mask_logits: list[tuple[torch.Tensor, int]] = []
        self._last_degradation_mask: torch.Tensor | None = None
        self._warned_once: set[str] = set()

    def _warn_once(self, key: str, message: str) -> None:
        """Emit a Python warning once per process for a stable key."""
        if key not in self._warned_once:
            warnings.warn(message)
            self._warned_once.add(key)

    def _log_mask_warning_flag(
        self,
        step_name: Literal["train", "val"],
        key: str,
        value: float,
    ) -> None:
        """Log warning-state flags to experiment tracking for train/val."""
        self.log(
            f"{step_name}/warnings/{key}",
            value,
            prog_bar=False,
            on_step=step_name == "train",
            on_epoch=step_name == "val",
            sync_dist=True,
            rank_zero_only=True,
        )

    def _normalize_mask_shape(
        self,
        mask: torch.Tensor | None,
        reference: torch.Tensor,
        clamp_to_unit: bool = True,
    ) -> torch.Tensor | None:
        """Convert optional mask to shape (B, T, 1, H, W). Clamps to [0, 1] when ``clamp_to_unit``.

        When ``clamp_to_unit=False`` the tensor is returned in its native range
        so that auxiliary mask *logits* (unbounded) can also pass through this
        helper without being squashed.
        """
        if mask is None:
            return None
        if not torch.is_tensor(mask):
            return None

        if mask.ndim == 4:
            # Support (B, T, H, W) or (B, 1, H, W)
            if mask.shape[1] == reference.shape[1]:
                mask = mask.unsqueeze(2)
            elif mask.shape[1] == 1:
                mask = mask.unsqueeze(1)
            else:
                return None

        if mask.ndim != 5:
            return None

        if mask.shape[0] != reference.shape[0] or mask.shape[1] != reference.shape[1]:
            return None

        if mask.shape[2] != 1:
            if mask.shape[2] > 1:
                mask = mask[:, :, :1]
            else:
                return None

        if mask.shape[-2:] != reference.shape[-2:]:
            b, t, c, h, w = mask.shape
            flat = mask.view(b * t, c, h, w)
            if clamp_to_unit:
                resized = F.interpolate(flat, size=reference.shape[-2:], mode="nearest")
            else:
                resized = F.interpolate(
                    flat,
                    size=reference.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            mask = resized.view(b, t, c, reference.shape[-2], reference.shape[-1])

        mask = mask.float()
        if clamp_to_unit:
            mask = torch.clamp(mask, 0.0, 1.0)
        return mask

    def _balanced_mask_loss(
        self,
        pred_logits: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Class-balanced BCE-with-logits plus a small dice term.

        Both ``pred_logits`` and ``gt_mask`` are expected as ``(B, T, 1, H, W)``.
        ``pred_logits`` are raw logits (use BCEWithLogits for numerical stability).
        The positive-class weight is derived per-batch from mask density and
        clamped to a safe range so degenerate batches do not blow up the loss.
        """
        cfg = self.degradation_aware_loss_config
        pos = gt_mask.sum().clamp(min=1.0)
        neg = (1.0 - gt_mask).sum().clamp(min=1.0)
        pos_weight = (neg / pos).clamp(
            min=cfg.pos_weight_clamp_min, max=cfg.pos_weight_clamp_max
        )
        bce = F.binary_cross_entropy_with_logits(
            pred_logits,
            gt_mask,
            pos_weight=pos_weight,
            reduction="mean",
        )
        pred = torch.sigmoid(pred_logits)
        inter = (pred * gt_mask).sum()
        dice = 1.0 - (2.0 * inter + cfg.epsilon) / (
            pred.sum() + gt_mask.sum() + cfg.epsilon
        )
        return bce + cfg.dice_weight * dice

    def forward(self, degraded_frames: torch.Tensor) -> torch.Tensor:
        """Run DART backbone and cache the auxiliary mask logits used for supervision."""
        restored = self.net(degraded_frames)
        if not isinstance(restored, torch.Tensor):
            raise TypeError("VideoBackbone must return restored frames as a Tensor.")

        aux_logits: list[tuple[torch.Tensor, int]] = []
        net_gate_backward = getattr(self.net, "last_gate_mask_logits_backward", None)
        net_gate_forward = getattr(self.net, "last_gate_mask_logits_forward", None)
        if isinstance(net_gate_backward, torch.Tensor):
            aux_logits.append((net_gate_backward, 0))
        if isinstance(net_gate_forward, torch.Tensor):
            aux_logits.append((net_gate_forward, 1))

        self._last_aux_mask_logits = aux_logits
        return restored

    def _common_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Store optional degradation mask from batch and run shared prediction/target step."""
        self._last_degradation_mask = batch.get("degradation_mask")
        return super()._common_step(batch)

    def generator_loss(
        self,
        fake: torch.Tensor,
        ground_truth: torch.Tensor,
        step_name: Literal["train", "val"] = "train",
    ) -> torch.Tensor:
        """Compute DART generator loss.
        Composition:
            total = g_adv_loss
                  + lambda_rec * (L1 + perceptual)
                  + lambda_mask_sup * balanced_mask_loss
        """
        cfg = self.degradation_aware_loss_config

        gen_vid_feat = self.discriminator(fake)
        g_adv_loss = self.adversarial_loss(gen_vid_feat, is_real=True, is_disc=False)

        rec_base = self.criterion(
            self._reshape_time_into_batch(fake),
            self._reshape_time_into_batch(ground_truth),
        )
        rec_base = self.log_and_unwrap_loss(
            rec_base, step_name=step_name, include_total_loss=False
        )

        gt_mask = self._normalize_mask_shape(self._last_degradation_mask, fake)

        aux_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for logits, offset in self._last_aux_mask_logits:
            t_len = logits.shape[1]
            ref_slice = fake[:, offset : offset + t_len]
            norm_logits = self._normalize_mask_shape(
                logits, ref_slice, clamp_to_unit=False
            )
            if norm_logits is None:
                continue
            if gt_mask is not None:
                gt_slice = gt_mask[:, offset : offset + t_len]
                aux_pairs.append((norm_logits, gt_slice))

        missing_aux = float(len(aux_pairs) == 0)
        missing_gt_mask = float(gt_mask is None)

        self._log_mask_warning_flag(step_name, "missing_aux_mask_logits", missing_aux)
        self._log_mask_warning_flag(
            step_name, "missing_degradation_mask", missing_gt_mask
        )

        if len(aux_pairs) == 0:
            self._warn_once(
                f"{step_name}_missing_aux_mask",
                "DART degradation-aware loss: gate mask logits are missing "
                "(set expose_gate_mask_logits: true in the experiment config); "
                "mask supervision is set to zero.",
            )
        if gt_mask is None:
            self._warn_once(
                f"{step_name}_missing_gt_mask",
                "DART degradation-aware loss: degradation mask ground truth is "
                "missing; mask supervision is set to zero.",
            )

        zero = fake.new_tensor(0.0)

        rec_total = cfg.lambda_rec * rec_base

        if len(aux_pairs) > 0 and cfg.lambda_mask_sup > 0.0:
            mask_sup = torch.stack(
                [self._balanced_mask_loss(logits, gt) for logits, gt in aux_pairs]
            ).mean()
        else:
            mask_sup = zero

        total = g_adv_loss + rec_total + cfg.lambda_mask_sup * mask_sup

        for name, value in (
            ("gen_adversarial_loss", g_adv_loss),
            ("reconstruction_loss_base", rec_base),
            ("mask_supervision_loss", mask_sup),
        ):
            self.log(
                f"{step_name}/loss/{name}",
                value,
                prog_bar=False,
                on_step=step_name == "train",
                on_epoch=step_name == "val",
                sync_dist=True,
                rank_zero_only=True,
            )

        return total

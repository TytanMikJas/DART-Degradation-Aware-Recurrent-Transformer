from dataclasses import dataclass
from typing import Any, Literal, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from hydra.utils import get_class
from pytorch_lightning.loggers import WandbLogger

from src.losses.adversarial_loss import AdversarialLoss
from src.losses.weighted_multi_loss import MultiLossResult
from src.metrics.metrics_handler import MetricsHandler
from src.models.optical_flow.sequence_optical_flow import SequenceOpticalFlow
from src.utils.denormalizer import Denormalizer
from src.utils.frame_output_handler import FrameOutputHandler


@dataclass
class OptimizerConfig:
    optimizer_class: str
    learning_rate: float
    beta1: float
    beta2: float
    lr_decay_start_epoch: int
    lr_decay_epochs: int
    flow_lr_multiplier: Optional[float] = None


class SequenceAdversarialRestorationModel(pl.LightningModule):
    """
    Sequence video restoration with adversarial training.

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
        optical_flow: SequenceOpticalFlow,
        discriminator: nn.Module,
        denormalizer: Denormalizer,
        net: nn.Module,
        # Loss
        adversarial_loss: AdversarialLoss,
        loss: nn.Module,
        # Optimization
        generator_optimizer_config: OptimizerConfig,
        discriminator_optimizer_config: OptimizerConfig,
        test_chunk_size: int = 11,
        test_chunk_stride: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["optical_flow", "discriminator", "denormalizer"]
        )

        self.metrics_handler = MetricsHandler(self.device)

        # Models
        self.optical_flow = optical_flow
        self.discriminator = discriminator
        self.denormalizer = denormalizer
        self.net = net

        # Optimization
        self.generator_optimizer_config = generator_optimizer_config
        self.discriminator_optimizer_config = discriminator_optimizer_config

        # Initialize loss functions
        self.criterion = loss
        self.adversarial_loss = adversarial_loss

        # Unified test-time bookkeeping keyed by eval_type (e.g. "synthetic",
        # "absolutedegradation", "srwov").  Each eval_type maps video_name ->
        # {metric_name -> accumulated_value, "num_frames" -> count}.
        self.test_outputs: dict[str, dict[str, dict[str, float]]] = {}
        self.test_chunk_size = test_chunk_size
        self.test_chunk_stride = test_chunk_stride

        # Automatic optimization OFF - handle D and G steps manually
        self.automatic_optimization = False

    @staticmethod
    def _pad_to_divisor(
        tensor: torch.Tensor, divisor: int = 16
    ) -> tuple[torch.Tensor, dict[str, int]]:
        """
        Pads spatial dimensions to be divisible by divisor.
        Returns padded tensor and padding info for later unpadding.
        Pads symmetrically to preserve center alignment.
        """
        _, _, _, h, w = tensor.shape
        pad_h = (divisor - h % divisor) % divisor
        pad_w = (divisor - w % divisor) % divisor

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        if pad_h == 0 and pad_w == 0:
            return tensor, {
                "pad_top": 0,
                "pad_bottom": 0,
                "pad_left": 0,
                "pad_right": 0,
            }

        padded = torch.nn.functional.pad(
            tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0
        )
        pad_info = {
            "pad_top": pad_top,
            "pad_bottom": pad_bottom,
            "pad_left": pad_left,
            "pad_right": pad_right,
        }
        return padded, pad_info

    @staticmethod
    def _unpad(tensor: torch.Tensor, pad_info: dict[str, int]) -> torch.Tensor:
        """
        Removes padding added by _pad_to_divisor.
        """
        pad_top = pad_info["pad_top"]
        pad_bottom = pad_info["pad_bottom"]
        pad_left = pad_info["pad_left"]
        pad_right = pad_info["pad_right"]

        if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
            return tensor

        _, _, _, h, w = tensor.shape
        return tensor[:, :, :, pad_top : h - pad_bottom, pad_left : w - pad_right]

    def forward(self, degraded_frames: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through generator.

        Args:
            degraded_frames: Low-resolution/degraded frames (B, T, C, H, W) in [-1, 1] range

        Returns:
            Restored frames (B, T, C, H, W) in [-1, 1] range
        """
        return self.net(degraded_frames)

    def discriminator_loss(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the discriminator loss as the sum of the adversarial loss
        Args:
            real (torch.Tensor): Ground truth high-quality images.
            fake (torch.Tensor): Generated images from the model.
        Returns:
            torch.Tensor: Total discriminator loss.
        """
        # Real and fake discriminator outputs
        real_vid_feat = self.discriminator(real)
        fake_vid_feat = self.discriminator(fake.detach())

        # Discriminator losses
        dis_real_loss = self.adversarial_loss(real_vid_feat, is_real=True, is_disc=True)
        dis_fake_loss = self.adversarial_loss(
            fake_vid_feat, is_real=False, is_disc=True
        )
        d_loss = (dis_real_loss + dis_fake_loss) * 0.5

        return d_loss

    def _reshape_time_into_batch(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reshapes tensor from (B, T, C, H, W) to (B*T, C, H, W).
        It's needed as many metrics expect (N, C, H, W) shape.
        """
        _b, _t, c, h, w = tensor.shape
        return tensor.reshape(-1, c, h, w)

    def _get_video_name_id_map(
        self, batch: dict[str, torch.Tensor | str]
    ) -> dict[str, list[int]]:
        """Creates mapping from ``batch["video_name"]`` if present.

        When ``video_name`` is missing we treat all items as belonging to a
        single unnamed video (empty string).
        """
        video_name_id_map: dict[str, list[int]] = {}
        names = batch.get("video_name")
        if names is None:
            # fallback: everything same video
            video_name_id_map[""] = list(range(len(next(iter(batch.values())))))
            return video_name_id_map
        for i, video_name in enumerate(names):
            if video_name not in video_name_id_map:
                video_name_id_map[video_name] = []
            video_name_id_map[video_name].append(i)
        return video_name_id_map

    def _iter_chunk_plan(
        self, total_len: int, chunk_size: int, stride: int
    ) -> list[tuple[int, int, int]]:
        """Creates a chunking plan.

        Returns tuples of (start, end, commit_start).
        - start/end: slice of input clip to infer.
        - commit_start: first global frame index from this chunk to keep/save.
        """
        if total_len <= 0:
            return []
        if chunk_size <= 0:
            chunk_size = total_len
        if stride <= 0:
            stride = chunk_size

        chunk_size = min(chunk_size, total_len)

        plan: list[tuple[int, int, int]] = []
        i = 0
        while i < total_len:
            end = min(i + chunk_size, total_len)
            if i == 0:
                commit_start = 0
            else:
                overlap = max(0, chunk_size - stride)
                commit_start = min(end, i + overlap)

            plan.append((i, end, commit_start))
            if end >= total_len:
                break
            i += stride
        return plan

    def _extract_relpaths_for_sample(
        self, relpaths: Any, sample_idx: int, expected_len: int
    ) -> list[str]:
        """Normalizes dataloader-collated relpaths to list[str] for one sample."""

        def _flatten_singletons(x: Any) -> Any:
            if isinstance(x, (list, tuple)) and len(x) == 1:
                return _flatten_singletons(x[0])
            return x

        relpaths = _flatten_singletons(relpaths)

        # Normal case: _test_collate_fn preserves list[str] intact.
        if (
            isinstance(relpaths, list)
            and relpaths
            and all(isinstance(p, str) for p in relpaths)
        ):
            return relpaths

        # Fallback: default_collate transposes list[str] into list[tuple[str]].
        # e.g. batch_size=1 with ["p1","p2",...] becomes [("p1",),("p2",),...]
        if (
            isinstance(relpaths, list)
            and relpaths
            and isinstance(relpaths[0], (list, tuple))
        ):
            try:
                flat = [
                    p[0] if isinstance(p, (list, tuple)) and len(p) >= 1 else str(p)
                    for p in relpaths
                ]
                if all(isinstance(p, str) for p in flat):
                    return flat
            except (IndexError, TypeError):
                pass

        # batch_size > 1: list[list[str]] indexed by sample.
        if (
            isinstance(relpaths, list)
            and len(relpaths) > sample_idx
            and isinstance(relpaths[sample_idx], (list, tuple))
            and all(isinstance(p, str) for p in relpaths[sample_idx])
        ):
            return list(relpaths[sample_idx])

        return [f"{i:08d}.png" for i in range(expected_len)]

    def _extract_video_name_for_sample(self, names: Any, sample_idx: int) -> str:
        names = names if names is not None else ""
        if isinstance(names, str):
            return names
        if isinstance(names, (list, tuple)) and len(names) > sample_idx:
            n = names[sample_idx]
            return n[0] if isinstance(n, (list, tuple)) and n else str(n)
        return ""

    def _sum_metric_values(
        self, metric_value: torch.Tensor, expected_count: int
    ) -> float:
        """Converts metric tensor to sum over expected_count frames.

        Some metric backends return per-image vectors, others return reduced scalars.
        """
        mt = torch.atleast_1d(metric_value.detach()).float()
        if mt.numel() == expected_count:
            return float(mt.sum().cpu())
        return float((mt.mean() * expected_count).cpu())

    def generator_loss(
        self,
        fake: torch.Tensor,
        ground_truth: torch.Tensor,
        step_name: Literal["train", "val"] = "train",
    ) -> torch.Tensor:
        """
        Computes the generator loss as the sum of the reconstruction loss
        also logs loss components.
        Args:
            fake (torch.Tensor): Generated images from the model.
            ground_truth (torch.Tensor): Ground truth high-quality images.
            step_name (Literal["train", "val"]): Current step name for logging.
            Defaults to "train". used for logging purposes.
        Returns:
            torch.Tensor: Total generator loss.
        """
        gen_vid_feat = self.discriminator(fake)
        g_adv_loss = self.adversarial_loss(gen_vid_feat, is_real=True, is_disc=False)

        _b, _t, c, h, w = fake.shape

        # The perceptual loss receives frames in [-1, 1] without VGG19 input
        # normalization, as in RTN and MambaOFR.
        reconstruction_loss = self.criterion(
            self._reshape_time_into_batch(fake),
            self._reshape_time_into_batch(ground_truth),
        )
        reconstruction_loss = self.log_and_unwrap_loss(
            reconstruction_loss, step_name=step_name, include_total_loss=False
        )

        self.log(
            f"{step_name}/loss/gen_adversarial_loss",
            g_adv_loss,
            prog_bar=False,
            on_step=step_name == "train",
            on_epoch=step_name == "val",
            sync_dist=True,
            rank_zero_only=True,
        )
        return g_adv_loss + reconstruction_loss

    def _common_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Common step for training and validation.
        Args:
            batch: Dictionary with 'degraded' and 'clean' keys
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing predicted and ground truth tensors.
        """
        lq = batch["degraded"]  # (B, T, C, H, W) in [-1, 1]
        gt = batch["clean"]  # (B, T, C, H, W) in [-1, 1]
        predicted = self(lq)
        return predicted, gt

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """
        Training step with manual optimization for discriminator and generator.

        Args:
            batch: Dictionary with 'degraded' and 'clean' keys
            batch_idx: Batch index
        """
        opt_g, opt_d = self.optimizers()

        predicted, gt = self._common_step(batch)

        # DISCRIMINATOR STEP
        opt_d.zero_grad()
        d_loss = self.discriminator_loss(gt, predicted)
        self.manual_backward(d_loss)
        opt_d.step()
        self.log(
            "train/loss/disc_loss",
            d_loss,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            rank_zero_only=True,
        )

        # GENERATOR STEP
        opt_g.zero_grad()
        g_loss = self.generator_loss(predicted, gt)
        self.manual_backward(g_loss)
        opt_g.step()
        self.log(
            "train/loss/gen_total_loss",
            g_loss,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            rank_zero_only=True,
        )

    @torch.no_grad()
    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """
        Validation step.

        Args:
            batch: Dictionary with 'degraded' and 'clean' keys
            batch_idx: Batch index

        Returns:
            Validation loss
        """
        predicted, gt = self._common_step(batch)
        d_loss = self.discriminator_loss(gt, predicted)
        self.log(
            "val/loss/disc_loss",
            d_loss,
            prog_bar=False,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=True,
        )
        v_loss = self.generator_loss(predicted, gt, step_name="val")
        self.log(
            "val/loss/gen_total_loss",
            v_loss,
            prog_bar=False,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=True,
        )
        self.log_validation_metrics(batch, predicted, batch_idx)
        return v_loss

    def on_test_epoch_start(self) -> None:
        self.test_outputs = {}

    @torch.no_grad()
    def test_step(
        self,
        batch: dict[str, torch.Tensor | str],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Run test-time restoration with full-clip chunking and unique-frame commit.

        Expected primary batch schema (sequence mode):
            - ``degraded``: ``(B, T, C, H, W)`` normalized input clip(s)
            - ``clean``: optional ``(B, T, C, H, W)`` target clip(s) for synthetic data
            - ``relpath``: per-frame relative paths (used for deterministic saving)
            - ``video_name``: clip identifier(s) used for per-video aggregation

        Main behavior for sequence batches (``degraded.ndim == 5``):
            1. Build a sliding temporal plan using ``test_chunk_size`` and
               ``test_chunk_stride``.
            2. Run the model on each chunk.
            3. Commit outputs:
               - first chunk: keep all frames,
               - later chunks: keep only the newly advanced tail.
               This avoids duplicate saves/metrics and preserves full coverage
               of clip boundaries.
            4. Save committed frames immediately via ``FrameOutputHandler``.
            5. Compute metrics only on committed frames and accumulate running
               per-video sums plus ``num_frames``.

        Data type routing:
            - synthetic batch (``clean`` present): full-reference + no-reference metrics
            - real batch (``clean`` absent): no-reference metrics only

        Input contract:
            This implementation is sequence-only during testing and expects
            ``degraded`` with shape ``(B, T, C, H, W)``.
        """
        degraded = batch.get("degraded")
        clean = batch.get("clean")

        if degraded is None:
            print(
                f"[test_step batch={batch_idx} dl={dataloader_idx}] Missing degraded input"
            )
            return

        # Determine eval_type: use the DataModule's per-loader name when available
        # (supports arbitrary number of test sets with custom subdirectory names).
        # Falls back to "synthetic" / "real" based on clean-data presence so that
        # existing DART test configs that pre-date this change continue to work.
        dm = getattr(getattr(self, "trainer", None), "datamodule", None)
        if dm is not None and hasattr(dm, "eval_type_for_loader"):
            eval_type = dm.eval_type_for_loader(dataloader_idx)
        else:
            eval_type = "synthetic" if clean is not None else "real"
        outputs_dict = self.test_outputs.setdefault(eval_type, {})
        saver = FrameOutputHandler.get_instance()

        # Full-clip chunked inference:
        # first chunk -> keep all, next chunks -> keep only newly advanced tail.
        if degraded.ndim != 5:
            raise ValueError(
                f"Sequence test expects degraded shape (B,T,C,H,W), got {tuple(degraded.shape)}"
            )

        batch_size, total_len = degraded.shape[:2]
        for b in range(batch_size):
            video_name = self._extract_video_name_for_sample(batch.get("video_name"), b)
            clip_relpaths = self._extract_relpaths_for_sample(
                batch.get("relpath"), b, total_len
            )
            if len(clip_relpaths) != total_len:
                clip_relpaths = [f"{i:08d}.png" for i in range(total_len)]

            clip_degraded = degraded[b : b + 1]
            clip_clean = clean[b : b + 1] if clean is not None else None

            plan = self._iter_chunk_plan(
                total_len=total_len,
                chunk_size=self.test_chunk_size,
                stride=self.test_chunk_stride,
            )

            video_outputs = outputs_dict.setdefault(video_name, {})
            video_outputs.setdefault("num_frames", 0)

            for start, end, commit_start in plan:
                try:
                    part_degraded = clip_degraded[:, start:end]
                    part_degraded_padded, pad_info = self._pad_to_divisor(
                        part_degraded, divisor=16
                    )
                    part_pred_padded = self(part_degraded_padded)
                    part_pred = self._unpad(part_pred_padded, pad_info)
                except Exception as e:
                    print(
                        f"[test_step batch={batch_idx} dl={dataloader_idx}] "
                        f"Forward failed at chunk [{start}:{end}] for video '{video_name}': "
                        f"{type(e).__name__}: {e}"
                    )
                    return

                part_pred = torch.clamp(self.denormalizer(part_pred), 0.0, 1.0)
                local_commit_start = max(0, commit_start - start)
                committed_pred = part_pred[:, local_commit_start:]
                committed_count = committed_pred.shape[1]
                if committed_count == 0:
                    continue

                committed_relpaths = clip_relpaths[commit_start:end]
                try:
                    saver.save_batch(
                        committed_pred.squeeze(0),
                        committed_relpaths,
                        eval_type,
                    )
                except Exception as e:
                    print(
                        f"[test_step batch={batch_idx}] Saving chunk [{commit_start}:{end}] "
                        f"for video '{video_name}' failed: {e}"
                    )

                committed_pred_flat = self._reshape_time_into_batch(committed_pred)

                if clip_clean is not None:
                    part_clean = torch.clamp(
                        self.denormalizer(clip_clean[:, start:end]), 0.0, 1.0
                    )
                    committed_clean = part_clean[:, local_commit_start:]
                    committed_clean_flat = self._reshape_time_into_batch(
                        committed_clean
                    )
                    metrics = self.metrics_handler.eval_metrics(
                        committed_pred_flat, committed_clean_flat
                    )
                else:
                    metrics = self.metrics_handler.eval_nr_metrics(committed_pred_flat)

                for metric_name, metric_value in metrics.items():
                    video_outputs.setdefault(metric_name, 0.0)
                    video_outputs[metric_name] += self._sum_metric_values(
                        metric_value, committed_count
                    )

                video_outputs["num_frames"] += committed_count

    def on_test_epoch_end(self) -> None:
        all_metrics_rows = []
        for eval_type, videos_metrics_map in self.test_outputs.items():
            if len(videos_metrics_map) == 0:
                continue
            final_metrics: dict[str, float] = {}
            total_frames = 0

            for video_data in videos_metrics_map.values():
                num_frames = video_data["num_frames"]
                total_frames += num_frames

                for metric_name, metric_value in video_data.items():
                    if metric_name == "num_frames":
                        continue
                    final_metrics.setdefault(metric_name, 0.0)
                    final_metrics[metric_name] += metric_value

            for metric_name, total_sum in final_metrics.items():
                global_avg = total_sum / total_frames if total_frames > 0 else 0.0
                all_metrics_rows.append(
                    {"Type": eval_type, "Metric": metric_name, "Value": global_avg}
                )

        if all_metrics_rows:
            import pandas as pd
            import tabulate
            import wandb
            from pytorch_lightning.loggers import WandbLogger

            df = pd.DataFrame(all_metrics_rows)
            print("----- Test Metrics Summary -----")
            print(tabulate.tabulate(df, headers="keys", tablefmt="psql"))
            if self.logger and isinstance(self.logger, WandbLogger):
                metrics_table = wandb.Table(dataframe=df)
                self.logger.experiment.log({"test/metrics_summary": metrics_table})

    def log_validation_metrics(
        self,
        batch: dict[str, torch.Tensor],
        predicted: torch.Tensor,
        batch_idx: int,
    ) -> None:
        """
        Logs validation metrics and draws images to wandb.
        """
        predicted_t = torch.clamp(self.denormalizer(predicted.detach()), 0.0, 1.0)
        gt_t = torch.clamp(self.denormalizer(batch["clean"].detach()), 0.0, 1.0)
        degraded_t = torch.clamp(
            self.denormalizer(batch["degraded"].detach()), 0.0, 1.0
        )
        metrics = self.metrics_handler.eval_metrics(
            self._reshape_time_into_batch(predicted_t),
            self._reshape_time_into_batch(gt_t),
        )
        metrics = {
            f"val/metrics/{metric}": value.mean() for metric, value in metrics.items()
        }
        self.log_dict(
            metrics,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=True,
        )
        if batch_idx == 0 and self.global_rank == 0:
            self._log_images(degraded_t, gt_t, predicted_t, step_name="val")

    def _log_images(
        self,
        degraded: torch.Tensor,
        gt: torch.Tensor,
        predicted: torch.Tensor,
        step_name: str,
    ) -> None:
        """
        Logs degraded, gt, predicted images to wandb with corresponding step name.
        """
        degraded_uint_8 = (degraded * 255).to(torch.uint8)
        gt_uint_8 = (gt * 255).to(torch.uint8)
        predicted_uint_8 = (predicted * 255).to(torch.uint8)
        degraded_imgs = self._tensors_to_wandb_imgs(
            degraded=degraded_uint_8,
            ground_truth=gt_uint_8,
            restored=predicted_uint_8,
            caption="Degraded / Ground Truth / Predicted",
        )
        if self.logger and isinstance(self.logger, WandbLogger):
            self.logger.experiment.log(
                {
                    f"images/{step_name}/{self.current_epoch}": degraded_imgs,
                    "global_step": self.global_step,
                },
            )

    def _tensors_to_wandb_imgs(
        self,
        degraded: torch.Tensor,
        ground_truth: torch.Tensor,
        restored: torch.Tensor,
        caption: str = "",
    ) -> list[wandb.Image]:
        """
        Converts a tensor to a list of wandb.Image for logging.
        Args:
            degraded (torch.Tensor): Degraded images tensor of shape (B, T, C, H, W)
                with values in [0, 255]  uint8.
            ground_truth (torch.Tensor): Ground truth images tensor of shape (B, T, C, H, W)
                with values in [0, 255] - uint8.
            restored (torch.Tensor): Restored images tensor of shape (B, T, C, H, W)
                with values in [0, 255] - uint8.
            caption (str): Caption for the images. Will be appended with the image index. of batch
        Returns:
            list[wandb.Image]: List of wandb.Image objects.
        """
        batch_size, num_frames, C, H, W = degraded.shape

        wandb_images = []
        for b in range(batch_size):
            vertical_stacks = []
            for t in range(num_frames):
                img_degraded = degraded[b, t].permute(1, 2, 0)
                img_gt = ground_truth[b, t].permute(1, 2, 0)
                img_restored = restored[b, t].permute(1, 2, 0)
                stack_img = torch.cat([img_degraded, img_gt, img_restored], dim=0)
                vertical_stacks.append(stack_img)
            composed_img = torch.cat(vertical_stacks, dim=1)
            wandb_images.append(
                wandb.Image(composed_img.cpu().numpy(), caption=f"{caption} b={b}")
            )
        return wandb_images

    def _setup_optimizer(
        self, optimizer_config: OptimizerConfig, params: list[torch.nn.Parameter]
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        optimizer_class = get_class(optimizer_config.optimizer_class)
        optimizer = optimizer_class(
            params,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
        )

        def _lr_lambda(epoch):
            if epoch < optimizer_config.lr_decay_start_epoch:
                return 1.0
            else:
                # Linear decay over lr_decay_epochs
                decay_progress = (
                    epoch - optimizer_config.lr_decay_start_epoch
                ) / optimizer_config.lr_decay_epochs
                return max(0.0, 1.0 - decay_progress)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
        return optimizer, scheduler

    def configure_optimizers(self):
        """
        Configure optimizers with separate learning rates for flow network.

        Returns:
            Tuple of (optimizers, schedulers)
        """
        # Separate parameters for generator: backbone vs flow network
        if (
            self.generator_optimizer_config.flow_lr_multiplier is None
            or self.generator_optimizer_config.flow_lr_multiplier == 0.0
        ):
            # No flow network or zero LR for flow
            optim_params_g = self.net.parameters()
        else:
            normal_params = []
            flow_net_params = []
            for name, param in self.net.named_parameters():
                if "optical_flow" in name:
                    flow_net_params.append(param)
                else:
                    normal_params.append(param)

            optim_params_g = [
                {
                    "params": normal_params,
                    "lr": self.generator_optimizer_config.learning_rate,
                },
                {
                    "params": flow_net_params,
                    "lr": self.generator_optimizer_config.learning_rate
                    * self.generator_optimizer_config.flow_lr_multiplier,
                },
            ]

        optimizer_g, scheduler_g = self._setup_optimizer(
            self.generator_optimizer_config, optim_params_g
        )
        optimizer_d, scheduler_d = self._setup_optimizer(
            self.discriminator_optimizer_config, self.discriminator.parameters()
        )

        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]

    def on_train_epoch_end(self) -> None:
        """Called at the end of each training epoch."""
        # Update learning rate schedulers
        schedulers = self.lr_schedulers()
        if schedulers is not None:
            for scheduler in schedulers:
                scheduler.step()

    def log_loss_metrics(
        self,
        loss: MultiLossResult,
        step_name: Literal["train", "val"],
        include_total_loss: bool = True,
    ):
        """
        Logs loss metrics to wandb.
        """
        loss_val, metrics = loss
        metrics = {
            f"{step_name}/loss/{metric}": value for metric, value in metrics.items()
        }
        if include_total_loss:
            metrics[f"{step_name}/loss/total_loss"] = loss_val
        on_step = step_name == "train"
        on_epoch = step_name == "val"
        self.log_dict(
            metrics,
            prog_bar=False,
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=True,
            rank_zero_only=True,
        )

    def log_and_unwrap_loss(
        self,
        loss: MultiLossResult | torch.Tensor,
        step_name: Literal["train", "val"],
        include_total_loss: bool = True,
    ) -> torch.Tensor:
        """
        Logs loss or/and it's sublosses
        if loss is a MultiLossResult and unwraps it to return the actual loss tensor.
        if loss is a simple tensor, just logs it and returns.
        """
        if isinstance(loss, tuple):
            self.log_loss_metrics(
                loss, step_name=step_name, include_total_loss=include_total_loss
            )
            return loss[0]
        else:
            self.log(
                f"{step_name}/loss/total_loss",
                loss,
                prog_bar=True,
                on_step=step_name == "train",
                on_epoch=step_name == "val",
                sync_dist=True,
                rank_zero_only=True,
            )
            return loss

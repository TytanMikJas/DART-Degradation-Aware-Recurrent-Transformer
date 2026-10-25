from typing import Literal

import pyiqa
import torch
from torch import nn


class MetricsHandler(nn.Module):
    """
    Wrapper class for the pyiqa image quality metrics.
    """

    FR_METRICS: tuple[str, ...] = ("psnr", "ssim", "lpips")
    NR_METRICS: tuple[str, ...] = ("brisque", "niqe", "clipiqa+", "musiq", "maniqa")

    DEFAULT_METRICS: tuple[str, ...] = FR_METRICS + ("brisque", "niqe")

    def __init__(
        self,
        device: torch.device | str = "cpu",
        metrics: list[str] | None = DEFAULT_METRICS,
    ):
        super().__init__()

        self._metric_fns = nn.ModuleDict(
            {name: pyiqa.create_metric(name, device=device) for name in metrics}
        )

    @torch.inference_mode()
    def eval_fr_metrics(
        self,
        generated: torch.Tensor,
        high_quality: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Evals full reference metrics i.e. metrics for paired data
        """
        return {
            name: metric(generated, high_quality)
            for name, metric in self._metric_fns.items()
            if name in self.FR_METRICS
        }

    @torch.inference_mode()
    def eval_nr_metrics(self, generated: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Evaluates all no-reference metrics
        """
        return {
            name: metric(generated)
            for name, metric in self._metric_fns.items()
            if name in self.NR_METRICS
        }

    @torch.inference_mode()
    def eval_metrics(
        self,
        generated: torch.Tensor,
        high_quality: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        paired_metrics = self.eval_fr_metrics(generated, high_quality)
        unpaired_metrics = self.eval_nr_metrics(generated)
        return paired_metrics | unpaired_metrics

    def get_available_metrics(self, filter: Literal["fr", "nr"] | None = None):
        filter_subset: list[str]
        match filter:
            case "fr":
                filter_subset = self.FR_METRICS
            case "nr":
                filter_subset = self.NR_METRICS
            case None:
                filter_subset = self.FR_METRICS + self.NR_METRICS

        return [
            metric_name
            for metric_name in self._metric_fns.keys()
            if metric_name in filter_subset
        ]

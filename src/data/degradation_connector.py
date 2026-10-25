"""
Connector that adapts AbsoluteDegradation's DegradationPipeline for use as
the degradation_fn in VideoFrameSequenceDataset (online degradation mode).

Format conversion:
    Dataset side:  RGB uint8  [0-255]  (H, W, 3)  ← load_rgb()
    Pipeline side: grayscale float32 [0-1]   (H, W)  ← expected by every distortion step

The pipeline also crops frames internally (CropStep + motion padding trimming),
so when this connector is used, spatial crop transforms must be removed from
the albumentations pipeline in the DataModule.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from absolute_degradation.distortions.degradation_pipeline import (
        DegradationPipeline,
    )

logger = logging.getLogger(__name__)


class OnlineDegradationFn:
    """
    Wraps AbsoluteDegradation's DegradationPipeline as a degradation function
    compatible with VideoFrameSequenceDataset.

    Conversion flow:
        1. Input: RGB uint8 [0-255] (H, W, 3) — from AbsoluteCinema's load_rgb
        2. Convert to grayscale float32 [0-1] (H, W) — pipeline's expected format
        3. Run DegradationPipeline (crops, distorts, quantizes)
        4. Convert output back to RGB uint8 [0-255] (H, W, 3) — for dataset/model

    The pipeline internally handles:
        - Random cropping (CropStep to crop_h × crop_w)
        - Motion padding removal (Y/X AxisMotionStep)
        - Degradation degree selection (LIGHT / MEDIUM / HEAVY)
        - Film scratch generation
        - Quantization to uint8

    Args:
        pipeline: An instantiated DegradationPipeline from AbsoluteDegradation.
        include_mask: If True, connector returns degradation masks alongside
            degraded and gt frames.
    """

    def __init__(
        self,
        pipeline: "DegradationPipeline",
        include_mask: bool = False,
    ) -> None:
        self.pipeline = pipeline
        self.include_mask = include_mask

    def __call__(
        self, clean_frames: list[npt.NDArray[np.uint8]]
    ) -> (
        tuple[list[npt.NDArray[np.uint8]], list[npt.NDArray[np.uint8]]]
        | tuple[
            list[npt.NDArray[np.uint8]],
            list[npt.NDArray[np.uint8]],
            list[npt.NDArray[np.float32]],
        ]
    ):
        """
        Apply the degradation pipeline to a sequence of clean frames.

        Args:
            clean_frames: List of RGB uint8 [0-255] images, shape (H, W, 3) or (H, W, 1).

        Returns:
            Tuple of (degraded_frames, gt_frames) or
            (degraded_frames, gt_frames, degradation_masks):
                - degraded_frames: List of RGB uint8 [0-255] images (H', W', 3) or (H', W', 1)
                - gt_frames: List of RGB uint8 [0-255] images (H', W', 3) or (H', W', 1)
                    Note: H' and W' are crop_size (typically 256), smaller than H, W.
                - degradation_masks (optional): List of float32 masks (H', W') in [0, 1].
        """
        grayscale_stream = self._to_grayscale_stream(clean_frames)
        degradation_masks: list[npt.NDArray[np.float32]] = []

        pipeline_output = self.pipeline(grayscale_stream)
        degraded_frames, gt_frames, *optional_masks = pipeline_output
        if optional_masks:
            degradation_masks = optional_masks[0]

        input_channels = clean_frames[0].shape[2]
        degraded = [
            self._to_desired_channels(f, input_channels) for f in degraded_frames
        ]
        gt = [self._to_desired_channels(f, input_channels) for f in gt_frames]

        if self.include_mask and len(degradation_masks) > 0:
            masks = [self._to_float_mask_2d(m) for m in degradation_masks]
            return degraded, gt, masks

        if self.include_mask and len(degradation_masks) == 0:
            raise ValueError(
                "include_mask=True but degradation pipeline returned no masks; "
                "falling back to (degraded, gt) output."
            )

        return degraded, gt

    @staticmethod
    def _to_float_mask_2d(mask: npt.NDArray) -> npt.NDArray[np.float32]:
        """Convert mask to float32 (H, W) in [0, 1]."""
        if mask.dtype == np.uint8:
            mask = mask.astype(np.float32) / 255.0
        else:
            mask = mask.astype(np.float32)

        if mask.ndim == 3:
            raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

        return np.clip(mask, 0.0, 1.0)

    @staticmethod
    def _to_grayscale_stream(
        frames: list[npt.NDArray[np.uint8]],
    ) -> list[npt.NDArray[np.float32]]:
        """Convert RGB uint8 [0-255] frames to grayscale float32 [0-1]."""
        result = []
        for frame in frames:
            gray = (
                cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                if frame.shape[2] == 3
                else frame.squeeze(2)
            )
            result.append(gray.astype(np.float32) / 255.0)
        return result

    @staticmethod
    def _to_desired_channels(
        img: npt.NDArray, desired_channels: int
    ) -> npt.NDArray[np.uint8]:
        """Convert pipeline output (grayscale, possibly float or uint8) to (H, W, desired_channels) uint8."""
        if img.dtype == np.float32:
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

        if img.ndim == 2:
            img = OnlineDegradationFn._grayscale_to_desired_channels(
                img, desired_channels
            )
        elif img.ndim == 3:
            img = OnlineDegradationFn._rgb_to_desired_channels(img, desired_channels)

        return img

    @staticmethod
    def _grayscale_to_desired_channels(
        img: npt.NDArray, desired_channels: int
    ) -> npt.NDArray[np.uint8]:
        """Convert collapsed grayscale image (H, W) to desired number of channels (H, W, desired_channels)."""
        if desired_channels == 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif desired_channels == 1:
            img = np.expand_dims(img, axis=-1)
        else:
            raise ValueError(f"Invalid number of channels: {desired_channels}")
        return img

    @staticmethod
    def _rgb_to_desired_channels(
        img: npt.NDArray, desired_channels: int
    ) -> npt.NDArray[np.uint8]:
        """Convert RGB image (H, W, 3) to desired number of channels (H, W, desired_channels)."""
        if desired_channels == 3:
            pass
        elif desired_channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.expand_dims(img, axis=-1)
        else:
            raise ValueError(f"Invalid number of channels: {desired_channels}")
        return img

import logging
import os
import random
from pathlib import Path
from typing import Callable

import albumentations as A
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import get_method
from omegaconf import ListConfig
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from src.data.utils import filter_and_sort_files, load_rgb

logger = logging.getLogger(__name__)


def _test_collate_fn(batch: list[dict]) -> dict:
    """Collate for test dataloaders (batch_size=1, full_video_mode).

    PyTorch's default_collate transposes a list[str] of length N into
    N tuples of length 1, which destroys the per-frame relpath list.
    This collate instead:
    - adds a batch dim to tensors  (T,C,H,W) -> (1,T,C,H,W)
    - leaves list[str] intact      ["p1", ..., "pN"] -> ["p1", ..., "pN"]
    - leaves plain str intact       "000" -> "000"
    """
    assert len(batch) == 1, "_test_collate_fn requires batch_size=1"
    sample = batch[0]
    result: dict = {}
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            result[key] = val.unsqueeze(0)
        elif isinstance(val, list) and val and all(isinstance(v, str) for v in val):
            result[key] = val  # per-frame relpath list – keep as-is
        elif isinstance(val, str):
            result[key] = val  # video_name – keep as plain str
        else:
            result[key] = default_collate([val])
    return result


def _worker_init_fn(worker_id: int) -> None:
    """
    Re-seed all RNG sources in each DataLoader worker to ensure independent
    random streams. Without this, all forked workers share identical state
    (Python random, numpy global RNG, and FilmScratchGenerator.rng), producing
    identical degradation augmentations across the batch.
    """
    cv2.setNumThreads(0)

    worker_info = torch.utils.data.get_worker_info()
    # torch.initial_seed() is set uniquely per worker by PyTorch
    seed = worker_info.seed % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    # Re-seed FilmScratchGenerator's private rng if present
    dataset = worker_info.dataset
    if hasattr(dataset, "degradation_fn") and dataset.degradation_fn is not None:
        pipeline = getattr(dataset.degradation_fn, "pipeline", None)
        if pipeline is not None:
            scratch_gen = getattr(pipeline, "scratch_generator", None)
            if scratch_gen is not None:
                scratch_gen.rng = np.random.default_rng(seed)


class VideoFrameSequenceDataset(Dataset):
    """
    Dataset to load sequences of video frames with variable length.

    Supports:
    - Variable sequence length
    - Optional RAM pre-loading with shared memory for multi-worker efficiency
    - On-the-fly degradation with degradation_fn (online mode)
    - Precomputed degraded data when degradation_fn is None (offline mode)

    Online degradation mode:
        The degradation_fn receives clean RGB uint8 [0-255] frames and returns
        a tuple (degraded_frames, gt_frames) or
        (degraded_frames, gt_frames, degradation_masks). Both outputs may differ from the
        original input (e.g., cropped and converted to grayscale by the pipeline).
        When using online degradation, the pipeline handles cropping internally,
        so spatial crop transforms should NOT be included in the transform.

    Args:
        clean_root: Root directory for clean (GT) frames. Optional only for
            real data (include_clean=False, degradation_fn=None).
        degraded_root: Root directory for degraded frames (required in offline mode)
        num_frames: Number of frames per sequence (default: 7)
        transform: Albumentations Compose accepting 'images' (T, H, W, C) and
            'clean_images' (T, H, W, C) keys. All transforms are applied with
            identical random parameters across all temporal frames.
        image_extensions: Set of valid image extensions
        degradation_fn: Optional degradation function for on-the-fly degradation.
            Signature: (list[ndarray]) -> tuple[list[ndarray], list[ndarray]]
            or tuple[list[ndarray], list[ndarray], list[ndarray]].
            Returns degraded/gt as RGB uint8 [0-255], and optional degradation
            masks as float32 [0-1] with shape (H, W).
        preload: If True, load all frames into shared memory at init time.
            Eliminates disk I/O in __getitem__ and shares physical memory across
            DataLoader workers (via torch shared memory). Requires all frames
            to have identical dimensions.
        frame_loader_fn: Function to load frames from disk. Must return a numpy array.
        include_clean: If True, __getitem__ returns both degraded and clean tensors.
    """

    def __init__(
        self,
        clean_root: Path | None,
        degraded_root: Path | None = None,
        num_frames: int = 7,
        full_video_mode: bool = False,
        transform: A.Compose | None = None,
        image_extensions: set[str] | None = None,
        degradation_fn: Callable | None = None,
        frame_loader_fn: Callable[[Path], np.ndarray] = load_rgb,
        preload: bool = False,
        include_clean: bool = True,
    ):
        self.clean_root = clean_root
        self.degraded_root = degraded_root
        self.num_frames = num_frames
        self.full_video_mode = full_video_mode
        self.transform = transform
        self.degradation_fn = degradation_fn
        self.frame_loader_fn = frame_loader_fn
        self.preload = preload
        self.include_clean = include_clean

        self.clean_files: list[Path] | None
        if self.clean_root is not None:
            self.clean_files = filter_and_sort_files(self.clean_root, image_extensions)
            if not self.clean_files:
                raise FileNotFoundError(f"No files found in {self.clean_root}")
        else:
            self.clean_files = None

        # Load degraded files if not using on-the-fly degradation
        if self.degradation_fn is None:
            if self.degraded_root is None:
                raise ValueError(
                    "degraded_root must be provided when degradation_fn is None"
                )
            self.degraded_files = filter_and_sort_files(
                self.degraded_root, image_extensions
            )
            if not self.degraded_files:
                raise FileNotFoundError(f"No files found in {self.degraded_root}")
            if (
                self.include_clean
                and self.clean_files is not None
                and len(self.degraded_files) != len(self.clean_files)
            ):
                raise ValueError(
                    f"File counts mismatch: "
                    f"{len(self.degraded_files)} degraded vs {len(self.clean_files)} clean."
                )
        else:
            if self.clean_files is None:
                raise ValueError(
                    "clean_root must be provided when degradation_fn is not None"
                )
            self.degraded_files = None

        self.sequence_root: Path
        if self.clean_files is not None and self.clean_root is not None:
            self.sequence_files = self.clean_files
            self.sequence_root = self.clean_root
        elif self.degraded_files is not None and self.degraded_root is not None:
            self.sequence_files = self.degraded_files
            self.sequence_root = self.degraded_root
        else:
            raise ValueError("No frame source available for sequence indexing")

        # Find valid sequences
        self.valid_sequences = self._find_valid_sequences()

        if not self.valid_sequences:
            raise ValueError(
                f"No valid frame sequences found with {self.num_frames} frames. "
                f"Check directory structure and ensure videos have at least {self.num_frames} frames."
            )

        # Pre-load all frames into shared memory if requested
        self._clean_cache: torch.Tensor | None = None
        self._degraded_cache: torch.Tensor | None = None
        if self.preload:
            self._preload_to_shared_memory()

    def _load_frame(self, path: Path) -> np.ndarray:
        """Load a single frame from disk."""
        return self.frame_loader_fn(path)

    def _load_frames_into_array(
        self,
        files: list[Path],
        cache: torch.Tensor,
        progress_bar: bool = False,
    ) -> None:
        """Load frames into a numpy array."""
        h, w, c = cache.shape[1:]

        for i in tqdm(
            range(len(files)),
            desc="Loading frames",
            unit="frame",
            disable=not progress_bar,
        ):
            frame = self._load_frame(files[i])
            if frame.shape != (h, w, c):
                raise ValueError(
                    f"Frame {files[i]} has shape {frame.shape}, expected ({h}, {w}, {c}). "
                    f"All frames must have identical dimensions."
                )
            cache[i] = torch.from_numpy(frame)

    def _preload_frames(
        self, files: list[Path], cache: torch.Tensor, label: str
    ) -> None:
        """Load frames into torch shared memory tensors."""
        disable_bar = int(os.environ.get("LOCAL_RANK", "0")) != 0
        self._load_frames_into_array(files, cache, progress_bar=not disable_bar)
        cache_mb = cache.nelement() * cache.element_size() / (1024 * 1024)
        logger.info(
            "%s frames loaded: %.1f MB in shared memory.", label.capitalize(), cache_mb
        )

    def _preload_to_shared_memory(self) -> None:
        """Load all frames into torch shared memory tensors.

        Uses torch.Tensor.share_memory_() so that forked DataLoader workers
        access the same physical memory pages instead of duplicating data.
        All frames must have identical (H, W) dimensions.
        """
        source_files = (
            self.clean_files if self.clean_files is not None else self.degraded_files
        )

        if source_files is None:
            raise ValueError("No source files available for preloading")

        n = len(source_files)
        first_frame = self._load_frame(source_files[0])
        h, w, c = first_frame.shape
        cache_shape = (n, h, w, c)
        fmt_str = "Pre-loading %d %s frames (%d x %d x %d) into shared memory..."

        if self.clean_files is not None:
            logger.info(fmt_str, n, "clean", h, w, c)
            self._clean_cache = torch.empty(
                *cache_shape, dtype=torch.uint8
            ).share_memory_()
            self._preload_frames(self.clean_files, self._clean_cache, "clean")

        if self.degraded_files is not None:
            logger.info(fmt_str, n, "degraded", h, w, c)
            self._degraded_cache = torch.empty(
                *cache_shape, dtype=torch.uint8
            ).share_memory_()
            self._preload_frames(self.degraded_files, self._degraded_cache, "degraded")

    def _get_frame(self, idx: int, *, degraded: bool = False) -> np.ndarray:
        """Load a single frame, from shared-memory cache or disk."""
        if self.preload:
            cache = self._degraded_cache if degraded else self._clean_cache
            if cache is None:
                source = "degraded" if degraded else "clean"
                raise ValueError(f"{source} frame cache is not available")
            return cache[idx].numpy().copy()

        files = self.degraded_files if degraded else self.clean_files
        if files is None:
            source = "degraded" if degraded else "clean"
            raise ValueError(f"{source} frame source is not available")
        return self._load_frame(files[idx])

    def _find_valid_sequences(self) -> list[tuple[int, int]]:
        """
        Find valid frame sequences within each video clip.

        Returns:
            List of (start_idx, end_idx) tuples for valid sequences
        """
        valid_sequences = []

        # Group files by video clip.
        video_groups: dict[str, list[int]] = {}
        for idx, file_path in enumerate(self.sequence_files):
            video_key = str(file_path.parent.relative_to(self.sequence_root))
            if video_key not in video_groups:
                video_groups[video_key] = []
            video_groups[video_key].append(idx)

        # Create sequences within each video
        for video_name, indices in video_groups.items():
            # Sort indices to ensure chronological order
            indices = sorted(indices)

            if self.full_video_mode:
                # Test-time clip mode: evaluate the entire clip at once (or chunked in model)
                valid_sequences.append((indices[0], indices[-1]))
                continue

            # Create sliding window of num_frames
            for i in range(len(indices) - self.num_frames + 1):
                start_idx = indices[i]
                end_idx = indices[i + self.num_frames - 1]

                # Verify indices are consecutive
                expected_indices = list(range(start_idx, start_idx + self.num_frames))
                actual_indices = indices[i : i + self.num_frames]

                if expected_indices == actual_indices:
                    valid_sequences.append((start_idx, end_idx))

        return valid_sequences

    def __len__(self):
        return len(self.valid_sequences)

    def __getitem__(self, idx):
        """
        Get a sequence of frames.

        Returns:
            Dictionary with:
                - 'degraded': Tensor of degraded frames (T, C, H, W)
                - 'clean': Tensor of clean frames (T, C, H, W).
                    Only included if include_clean=True.
                - 'degradation_mask': Tensor (T, 1, H, W) float32 in [0, 1].
                    Included only when degradation_fn returns masks.
                - 'relpath': Relative path of the center frame (for bookkeeping)
                - 'video_name': Name of the video clip (derived from center frame)
        """
        start_idx, end_idx = self.valid_sequences[idx]
        frame_indices = list(range(start_idx, end_idx + 1))

        try:
            need_clean_frames = self.degradation_fn is not None or self.include_clean
            clean_frames = (
                [self._get_frame(i) for i in frame_indices]
                if need_clean_frames
                else None
            )

            degradation_masks: list[np.ndarray] | None = None
            if self.degradation_fn is None:
                # Offline mode: load precomputed degraded frames
                degraded_frames = [
                    self._get_frame(i, degraded=True) for i in frame_indices
                ]
            else:
                # Online mode: pipeline returns (degraded, gt) or
                # (degraded, gt, degradation_masks)
                # The GT is also returned because the pipeline may crop/transform it
                degradation_output = self.degradation_fn(clean_frames)
                if len(degradation_output) == 3:
                    degraded_frames, clean_frames, degradation_masks = (
                        degradation_output
                    )
                else:
                    degraded_frames, clean_frames = degradation_output

        except Exception as e:
            raise IndexError(f"Failed to load sequence {idx}: {e}")

        clean_stack: np.ndarray | None = None
        if self.include_clean and clean_frames is not None:
            clean_stack = np.stack(clean_frames)  # (T, H, W, C)

        degradation_mask_stack: np.ndarray | None = None
        if degradation_masks is not None:
            mh, mw = degraded_frames[0].shape[:2]
            normalized_masks = [
                cv2.resize(m, (mw, mh), interpolation=cv2.INTER_NEAREST)
                if m.shape[:2] != (mh, mw)
                else m
                for m in degradation_masks
            ]
            degradation_mask_stack = np.stack(normalized_masks).astype(np.float32)

        degraded_stack = np.stack(degraded_frames)  # (T, H, W, C)

        if self.transform:
            if self.include_clean:
                if degradation_mask_stack is not None:
                    transformed = self.transform(
                        images=degraded_stack,
                        clean_images=clean_stack,
                        masks=degradation_mask_stack,
                    )
                else:
                    transformed = self.transform(
                        images=degraded_stack,
                        clean_images=clean_stack,
                    )
                degraded_stack = transformed["images"]
                clean_stack = transformed["clean_images"]
                if degradation_mask_stack is not None:
                    degradation_mask_stack = np.asarray(
                        transformed["masks"], dtype=np.float32
                    )
            else:
                if degradation_mask_stack is not None:
                    transformed = self.transform(
                        images=degraded_stack,
                        masks=degradation_mask_stack,
                    )
                else:
                    transformed = self.transform(images=degraded_stack)
                degraded_stack = transformed["images"]
                if degradation_mask_stack is not None:
                    degradation_mask_stack = np.asarray(
                        transformed["masks"], dtype=np.float32
                    )

        # Convert (T, H, W, C) numpy → (T, C, H, W) tensor
        degraded_tensor = torch.from_numpy(
            np.ascontiguousarray(degraded_stack.transpose(0, 3, 1, 2))
        )
        clean_tensor: torch.Tensor | None = None
        if self.include_clean and clean_stack is not None:
            clean_tensor = torch.from_numpy(
                np.ascontiguousarray(clean_stack.transpose(0, 3, 1, 2))
            )

        degradation_mask_tensor: torch.Tensor | None = None
        if degradation_mask_stack is not None:
            degradation_mask_tensor = torch.from_numpy(
                np.ascontiguousarray(degradation_mask_stack[:, None, :, :])
            )

        if self.full_video_mode:
            relpath = [
                str(self.sequence_files[i].relative_to(self.sequence_root))
                for i in frame_indices
            ]
            video_name = str(Path(relpath[0]).parent)
        else:
            # derive a relative path and video name from the centre frame for bookkeeping
            center_idx = frame_indices[self.num_frames // 2]
            relpath = str(
                self.sequence_files[center_idx].relative_to(self.sequence_root)
            )
            video_name = str(Path(relpath).parent)

        result: dict = {
            "degraded": degraded_tensor,
            "relpath": relpath,
            "video_name": video_name,
        }
        if self.include_clean and clean_tensor is not None:
            result["clean"] = clean_tensor
        if degradation_mask_tensor is not None:
            result["degradation_mask"] = degradation_mask_tensor
        return result


class VideoFrameSequenceDataModule(pl.LightningDataModule):
    """DataModule supporting both offline and online degradation modes.

    Offline mode (degradation_fn=None):
        Loads precomputed degraded + clean frame pairs from disk.
        Geometric transforms (including crop) are applied via albumentations.

    Online mode (degradation_fn provided):
        Only clean frames are loaded from disk. The degradation_fn is called
        per sequence to produce degraded + GT frame pairs on-the-fly.
        The degradation pipeline handles cropping internally, so geometric
        transforms should NOT include spatial crops — only flips/rotations.
        train_degraded_root is not required in this mode.
    """

    def __init__(
        self,
        train_clean_root: str | Path,
        val_degraded_root: str | Path,
        val_clean_root: str | Path,
        train_degraded_root: str | Path | None = None,
        batch_size: int = 32,
        num_workers: int = 4,
        crop_size: int = 256,
        num_frames: int = 7,
        normalization_mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
        normalization_std: tuple[float, float, float] = (1.0, 1.0, 1.0),
        train_transform_geometric: ListConfig = ListConfig([]),
        train_transform_photometric: ListConfig = ListConfig([]),
        val_transform: ListConfig = ListConfig([]),
        image_extensions: set[str] | None = None,
        degradation_fn: Callable | None = None,
        frame_loader_fn: str = "src.data.utils.load_grayscale",
        preload: bool = False,
        # legacy args from VideoDataModule – kept for config compatibility
        test_synthetic_degraded_root: str | Path | None = None,
        test_synthetic_clean_root: str | Path | None = None,
        test_real_degraded_root: str | Path | None = None,
        test_transform: ListConfig = ListConfig([]),
        # optional second real test set (e.g. data/test/real alongside data/scenes_medium)
        test_real2_degraded_root: str | Path | None = None,
        # configurable subdirectory names for each test split (controls save_root/<name>/)
        # defaults keep backward compatibility with existing DART test configs
        test_real_eval_type_name: str = "real",
        test_real2_eval_type_name: str = "real2",
    ):
        super().__init__()
        self.train_degraded_root = (
            Path(train_degraded_root) if train_degraded_root else None
        )
        self.train_clean_root = Path(train_clean_root)
        self.val_degraded_root = Path(val_degraded_root)
        self.val_clean_root = Path(val_clean_root)

        # legacy test roots (may be None) – used only if provided
        self.test_synthetic_degraded_root = (
            Path(test_synthetic_degraded_root) if test_synthetic_degraded_root else None
        )
        self.test_synthetic_clean_root = (
            Path(test_synthetic_clean_root) if test_synthetic_clean_root else None
        )
        self.test_real_degraded_root = (
            Path(test_real_degraded_root) if test_real_degraded_root else None
        )
        self.test_real2_degraded_root = (
            Path(test_real2_degraded_root) if test_real2_degraded_root else None
        )
        self.test_real_eval_type_name = test_real_eval_type_name
        self.test_real2_eval_type_name = test_real2_eval_type_name

        self.batch_size = batch_size
        if os.name == "nt" and num_workers > 0:
            logger.warning(
                "Windows detected: forcing num_workers=0 (was %d) to avoid "
                "multiprocessing DataLoader spawn errors.",
                num_workers,
            )
            num_workers = 0
        self.num_workers = num_workers
        self.crop_size = crop_size
        self.image_extensions = image_extensions
        self.num_frames = num_frames
        self.degradation_fn = degradation_fn
        self.frame_loader_fn = get_method(frame_loader_fn)
        self.preload = preload

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        # retain both old-style synthetic/real test datasets and new unified one
        self.test_synthetic_dataset: Dataset | None = None
        self.test_real_dataset: Dataset | None = None
        self.test_real2_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None
        # populated in setup("test"): ordered list of eval_type names per DataLoader index
        self._test_loader_eval_types: list[str] = []

        # build transforms; allow separate test_transform if provided
        standard_pipeline = [
            A.Normalize(
                mean=normalization_mean, std=normalization_std, max_pixel_value=255.0
            ),
        ]

        self.train_transform = A.Compose(
            [
                A.Compose(
                    list(train_transform_geometric),
                    additional_targets={"clean_images": "images"},
                ),
                A.Compose(list(train_transform_photometric)),
                A.Compose(
                    standard_pipeline,
                    additional_targets={"clean_images": "images"},
                ),
            ],
            additional_targets={"clean_images": "images"},
        )

        self.val_transform = A.Compose(
            list(val_transform) + standard_pipeline,
            additional_targets={"clean_images": "images"},
        )

        # optional transform used only during testing (legacy config may provide it)
        self.test_transform = A.Compose(
            list(test_transform) + standard_pipeline,
            additional_targets={"clean_images": "images"},
        )

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = VideoFrameSequenceDataset(
                degraded_root=self.train_degraded_root,
                clean_root=self.train_clean_root,
                transform=self.train_transform,
                image_extensions=self.image_extensions,
                num_frames=self.num_frames,
                degradation_fn=self.degradation_fn,
                preload=self.preload,
                frame_loader_fn=self.frame_loader_fn,
            )
            self.val_dataset = VideoFrameSequenceDataset(
                degraded_root=self.val_degraded_root,
                clean_root=self.val_clean_root,
                transform=self.val_transform,
                image_extensions=self.image_extensions,
                num_frames=self.num_frames,
                degradation_fn=None,
                preload=False,
                frame_loader_fn=self.frame_loader_fn,
            )
        elif stage == "test":
            self._test_loader_eval_types = []
            # legacy behaviour: separate synthetic/real datasets if explicit roots passed
            if self.test_synthetic_degraded_root or self.test_synthetic_clean_root:
                self.test_synthetic_dataset = VideoFrameSequenceDataset(
                    degraded_root=self.test_synthetic_degraded_root
                    or self.val_degraded_root,
                    clean_root=self.test_synthetic_clean_root or self.val_clean_root,
                    transform=self.test_transform,
                    image_extensions=self.image_extensions,
                    num_frames=self.num_frames,
                    full_video_mode=True,
                    degradation_fn=None,
                    preload=False,
                    frame_loader_fn=self.frame_loader_fn,
                )
                self._test_loader_eval_types.append("synthetic")
            if self.test_real_degraded_root:
                self.test_real_dataset = VideoFrameSequenceDataset(
                    degraded_root=self.test_real_degraded_root,
                    clean_root=None,
                    transform=self.test_transform,
                    image_extensions=self.image_extensions,
                    num_frames=self.num_frames,
                    full_video_mode=True,
                    degradation_fn=None,
                    preload=False,
                    frame_loader_fn=self.frame_loader_fn,
                    include_clean=False,
                )
                self._test_loader_eval_types.append(self.test_real_eval_type_name)
            if self.test_real2_degraded_root:
                self.test_real2_dataset = VideoFrameSequenceDataset(
                    degraded_root=self.test_real2_degraded_root,
                    clean_root=None,
                    transform=self.test_transform,
                    image_extensions=self.image_extensions,
                    num_frames=self.num_frames,
                    full_video_mode=True,
                    degradation_fn=None,
                    preload=False,
                    frame_loader_fn=self.frame_loader_fn,
                    include_clean=False,
                )
                self._test_loader_eval_types.append(self.test_real2_eval_type_name)
            # default unified dataset if no legacy roots provided
            if (
                self.test_synthetic_dataset is None
                and self.test_real_dataset is None
                and self.test_real2_dataset is None
            ):
                self.test_dataset = VideoFrameSequenceDataset(
                    degraded_root=self.val_degraded_root,
                    clean_root=self.val_clean_root,
                    transform=self.val_transform,
                    image_extensions=self.image_extensions,
                    num_frames=self.num_frames,
                    full_video_mode=True,
                    degradation_fn=None,
                    preload=False,
                    frame_loader_fn=self.frame_loader_fn,
                )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Train dataset is not set up. Call setup('fit') first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=_worker_init_fn if self.num_workers > 0 else None,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise ValueError(
                "Validation dataset is not set up. Call setup('fit') first."
            )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader | list[DataLoader]:
        """Return one or two dataloaders depending on which datasets were created.

        - new unified API (default): returns single DataLoader wrapping ``self.test_dataset``
        - legacy mode when test_synthetic/_real roots were provided: returns list with
          real and synthetic loaders (same order as the old VideoDataModule).
        """
        if (
            self.test_synthetic_dataset
            or self.test_real_dataset
            or self.test_real2_dataset
        ):
            loaders: list[DataLoader] = []
            if self.test_synthetic_dataset is not None:
                loaders.append(
                    DataLoader(
                        self.test_synthetic_dataset,
                        batch_size=1,
                        num_workers=self.num_workers,
                        shuffle=False,
                        pin_memory=True,
                        persistent_workers=self.num_workers > 0,
                        collate_fn=_test_collate_fn,
                    )
                )
            if self.test_real_dataset is not None:
                loaders.append(
                    DataLoader(
                        self.test_real_dataset,
                        batch_size=1,
                        num_workers=self.num_workers,
                        shuffle=False,
                        pin_memory=True,
                        persistent_workers=self.num_workers > 0,
                        collate_fn=_test_collate_fn,
                    )
                )
            if self.test_real2_dataset is not None:
                loaders.append(
                    DataLoader(
                        self.test_real2_dataset,
                        batch_size=1,
                        num_workers=self.num_workers,
                        shuffle=False,
                        pin_memory=True,
                        persistent_workers=self.num_workers > 0,
                        collate_fn=_test_collate_fn,
                    )
                )
            return loaders

        if self.test_dataset is None:
            raise ValueError("Test dataset is not set up. Call setup('test') first.")
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=_test_collate_fn,
        )

    def eval_type_for_loader(self, idx: int) -> str:
        """Return the eval_type name for the test DataLoader at position *idx*.

        Loader order is always: [synthetic (if any), real1 (if any), real2 (if any)].
        The names are set by ``test_real_eval_type_name`` / ``test_real2_eval_type_name``
        config params and determine the output subdirectory under ``save_root``.
        Falls back to ``"real"`` for any out-of-range index.
        """
        if idx < len(self._test_loader_eval_types):
            return self._test_loader_eval_types[idx]
        return "real"

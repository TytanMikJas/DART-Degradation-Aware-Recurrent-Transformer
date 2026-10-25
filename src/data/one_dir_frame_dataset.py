import logging
from pathlib import Path

import albumentations as A
import numpy as np
from torch.utils.data import Dataset

from src.data.utils import filter_and_sort_files, load_rgb

logger = logging.getLogger(__name__)


class OneDirFrameDataset(Dataset):
    """
    Dataset to load frames.

    Assumes directories are structured like:
    root/
        000/
            00001.png
            00002.png
            ...
        001/
            00001.png
            00002.png
            ...
        ...
    """

    def __init__(
        self,
        root: Path,
        transform: A.Compose | None = None,
        sample_size: int | None = None,
        image_extensions: set[str] | None = None,
    ):
        self.root = root
        self.transform = transform

        self.frames_files: list[Path] = filter_and_sort_files(
            self.root, image_extensions
        )

        if not self.frames_files:
            raise FileNotFoundError(f"No files found in {self.root}")

        if sample_size:
            self._sample(sample_size)

    def _sample(self, n: int):
        """Randomly samples n images from the dataset."""
        if n > len(self.frames_files):
            logger.error(
                f"Sample size {n} is larger than dataset size {len(self.frames_files)}."
            )
            return
        else:
            self.frames_files = np.random.choice(
                self.frames_files, n, replace=False
            ).tolist()

    def get_group_indices(self) -> dict[str, list[int]]:
        """Group frame indices by parent directory (video)."""
        groups: dict[str, list[int]] = {}
        for idx, path in enumerate(self.frames_files):
            key = str(path.parent)
            groups.setdefault(key, []).append(idx)
        return groups

    def __len__(self):
        return len(self.frames_files)

    def __getitem__(self, idx):
        frame_path = self.frames_files[idx]

        try:
            frame = load_rgb(frame_path)
        except Exception as e:
            raise IndexError(f"Failed to load sample {idx}: {e}")

        if self.transform:
            frame = self.transform(image=frame)["image"]

        return frame

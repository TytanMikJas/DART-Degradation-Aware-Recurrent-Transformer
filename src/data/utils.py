from pathlib import Path

import cv2
import numpy as np


def filter_and_sort_files(root: Path, extensions: set[str] | None) -> list[Path]:
    """Filters a list of file paths by the given set of extensions."""
    filtered_paths = [
        p
        for p in root.rglob("*")
        if p.is_file() and (extensions is None or p.suffix.lower() in extensions)
    ]
    filtered_paths.sort()
    return filtered_paths


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_grayscale(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Could not read image: {path}")
    # Ensure output shape is (H, W, C) with a singleton channel dimension for grayscale
    img = np.expand_dims(img, axis=-1)
    return img

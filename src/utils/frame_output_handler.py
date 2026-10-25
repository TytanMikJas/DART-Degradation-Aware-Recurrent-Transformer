from pathlib import Path

import cv2
import numpy as np
import torch


class FrameOutputHandler:
    """
    Singleton class responsible for saving restored video frames to disk.
    """

    _instance = None
    _initalized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(FrameOutputHandler, cls).__new__(cls)
        return cls._instance

    def __init__(self, save_root: str | Path) -> None:
        if self._initalized:
            return
        self._initalized = True

        self.save_root = Path(save_root)
        self.save_root.mkdir(parents=True, exist_ok=True)

    def _save_image_tensor(self, img_tensor: np.ndarray, save_path: Path) -> None:
        """
        Saves a single image tensor to disk.
        Args:
            img_tensor: RGB numpy image to save. Shape: (H, W, C) in [0, 255] range.
            save_path: Path where the image should be saved
        """
        cv2.imwrite(str(save_path), cv2.cvtColor(img_tensor, cv2.COLOR_RGB2BGR))

    def save_batch(
        self, restored_t: torch.Tensor, relative_paths: list[str], eval_type: str
    ) -> None:
        """
        Saves a batch of image tensors to the specified relative paths from the save root directory.
        Args:
            restored_t: Batch of RGB image tensors to save. Shape: (B, C, H, W) in [0, 1] range.
            relative_paths: List of relative file paths where each image tensor should be saved.
            eval_type: Type of evaluation (use real or synthetic).
        """
        if len(restored_t) != len(relative_paths):
            raise ValueError(
                "Length of restored_t and relative_paths must be the same."
            )
        restored_t_np_uint8 = (
            (restored_t * 255).cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)
        )
        for i, relpath in enumerate(relative_paths):
            save_path = self.save_root / eval_type / relpath
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_image_tensor(restored_t_np_uint8[i], save_path)

    @classmethod
    def get_instance(cls) -> "FrameOutputHandler":
        """
        Returns the singleton instance of the FrameOutputHandler class.
        """
        if cls._instance is None:
            raise ValueError("FrameOutputHandler instance has not been created yet.")
        return cls._instance

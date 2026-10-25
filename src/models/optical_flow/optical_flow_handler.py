from abc import ABC, abstractmethod
from typing import Optional

import torch.nn as nn

from src.models.optical_flow.raft_backend import RAFTEstimator
from src.utils.denormalizer import Denormalizer


class OpticalFlowHandlerAbs(nn.Module, ABC):
    """
    A module to handle optical flow computation and warping.

    Args:
        denormalizer: Denormalizer instance for consistent preprocessing
        model_name: Name of the optical flow model to use
        ckpt_path: Path to the checkpoint file for the optical flow model
        freeze: Whether to freeze the optical flow model
    """

    def __init__(
        self,
        denormalizer: Denormalizer,
        model_name: str = "raft",
        ckpt_path: Optional[str] = None,
        freeze: bool = True,
    ):
        super().__init__()
        self.freeze = freeze
        self.denormalizer = denormalizer

        match model_name.lower():
            case "raft":
                self.flow_net = RAFTEstimator(weights_path=ckpt_path, small=False)
            case "identity":
                # ABLATION MODE: No optical flow backbone
                self.flow_net = None
            case _:
                raise ValueError(f"Unknown optical flow model: {model_name}")

        if self.flow_net:
            if self.freeze:
                self._freeze_all()
            else:
                self._unfreeze_weights_but_freeze_bn()

    @abstractmethod
    def forward(
        self,
        *args,
        **kwargs,
    ):
        pass

    def train(self, mode=True):
        """
        Override the default train() to handle freezing logic.
        """
        super().train(mode)
        if self.flow_net:
            if self.freeze:
                self.flow_net.eval()
            elif mode:
                self._unfreeze_weights_but_freeze_bn()

        return self

    def _freeze_all(self):
        """
        Freeze all model weights.
        """
        if self.flow_net is None:
            return

        self.flow_net.eval()
        for param in self.flow_net.parameters():
            param.requires_grad = False

    def _unfreeze_weights_but_freeze_bn(self):
        """
        Unfreeze all model weights but keep batch norm layers frozen.
        """
        if self.flow_net is None:
            return
        self.flow_net.train()

        for param in self.flow_net.parameters():
            param.requires_grad = True

        self.flow_net.freeze_bn()

import argparse
from typing import Optional

import torch
import torch.nn as nn

from src.models.optical_flow.third_party.RAFT_core.raft import RAFT
from src.models.optical_flow.warp_utilities import InputPadder


class RAFTEstimator(nn.Module):
    """
    RAFT-based optical flow estimator.
    Adapts https://github.com/princeton-vl/RAFT
    """

    def __init__(
        self,
        weights_path: Optional[str],
        small: bool = False,
        mixed_precision: bool = False,
    ):
        super().__init__()

        args = argparse.Namespace()
        args.small = small
        args.mixed_precision = mixed_precision
        args.alternate_corr = False
        args.iters = 24

        self.model = RAFT(args)

        if weights_path:
            self._load_weights(weights_path)

        self.model.eval()

    def _load_weights(self, path):
        """Load weights from checkpoint, handling 'module.' prefixes if present."""
        state_dict = torch.load(path, map_location="cpu")

        if "model" in state_dict:
            state_dict = state_dict["model"]

        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v

        self.model.load_state_dict(new_state_dict, strict=False)

    def forward(self, img1, img2):
        """
        Estimate optical flow from img1 to img2.
        RAFT model expects inputs to be 0-255 floats and padded to be divisible by 8.
        """
        padder = InputPadder(img1.shape)
        img1, img2 = padder.pad(img1, img2)

        _, flow_up = self.model(img1, img2, test_mode=True)

        flow_up = padder.unpad(flow_up)

        return flow_up

    def freeze_bn(self):
        """Freeze all batch norm layers."""
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

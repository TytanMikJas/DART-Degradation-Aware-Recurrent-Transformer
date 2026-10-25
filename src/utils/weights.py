"""Loading DART network weights from the released files or a training checkpoint."""

from pathlib import Path

import torch
import torch.nn as nn


def load_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Read a DART ``net`` state dict.

    Supported formats:
        * ``.safetensors`` / ``.pth``: released weights (state dict of ``VideoBackbone``)
        * ``.ckpt``: Lightning checkpoint written by ``scripts/train.py``
          (only the ``net.*`` entries are used)
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Weights not found: {path}. Download them with "
            "`python scripts/download_weights.py`."
        )
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))

    # Lightning checkpoints store pickled hyper-parameters next to the tensors.
    obj = torch.load(path, map_location="cpu", weights_only=path.suffix != ".ckpt")
    state_dict = obj.get("state_dict", obj)
    if any(key.startswith("net.") for key in state_dict):
        state_dict = {
            key[len("net.") :]: value
            for key, value in state_dict.items()
            if key.startswith("net.")
        }
    return state_dict


def load_net_weights(net: nn.Module, path: str | Path) -> None:
    """Load DART weights into a ``VideoBackbone`` (strict: every key must match)."""
    net.load_state_dict(load_state_dict(path), strict=True)

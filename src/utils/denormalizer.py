"""Denormalization utility for reversing image normalization."""

import torch


class Denormalizer(torch.nn.Module):
    """Denormalizer that reverses normalization: x_denorm = x * std + mean."""

    def __init__(
        self,
        mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
        std: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        super().__init__()
        _mean = torch.tensor(mean).view(1, -1, 1, 1)
        _std = torch.tensor(std).view(1, -1, 1, 1)
        self.register_buffer("denormalizer_mean", _mean)
        self.register_buffer("denormalizer_std", _std)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.denormalize(tensor)

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Denormalize tensor. Accepts (C,H,W) or (B,C,H,W) or (B,T,C,H,W). Returns same shape and dtype float32."""
        match tensor.ndim:
            case 3:
                batched = False
                t = tensor.unsqueeze(0).float()
            case 4 | 5:
                batched = True
                t = tensor.float()
            case _:
                raise ValueError(f"Unsupported tensor shape {tuple(tensor.shape)}")

        den = t * self.denormalizer_std + self.denormalizer_mean

        if not batched:
            den = den.squeeze(0)

        return den

import torch
import torch.nn.functional as F


class InputPadder:
    """
    Pads images such that dimensions are divisible by 8.
    This is required for RAFT architecture.
    """

    def __init__(self, dims, mode="sintel"):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8

        if mode == "sintel":
            self._pad = [
                pad_wd // 2,
                pad_wd - pad_wd // 2,
                pad_ht // 2,
                pad_ht - pad_ht // 2,
            ]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]


def warp_flow(frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Warps an image 'frame' using the optical flow 'flow'.

    Args:
        frame: The image to warp (B, C, H, W) in range [0, 1].
        flow: The optical flow (B, 2, H, W) with (x, y) displacements.

    Returns:
        The warped image (B, C, H, W).
    """
    B, _, H, W = frame.size()

    # Create a base grid
    # 'ij' indexing gives (H, W) matrices
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, H, device=frame.device, dtype=frame.dtype),
        torch.arange(0, W, device=frame.device, dtype=frame.dtype),
        indexing="ij",
    )

    # Stack to (2, H, W)
    grid = torch.stack((grid_x, grid_y), dim=0)
    # Expand to batch size (B, 2, H, W)
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)

    # Add flow to the grid
    # Flow (B, 2, H, W) has (x_disp, y_disp)
    new_grid = grid + flow

    # Normalize grid to [-1, 1] for grid_sample (with align_corners=True)
    # x-coordinates
    new_grid[:, 0, :, :] = 2.0 * new_grid[:, 0, :, :] / max(W - 1, 1) - 1.0
    # y-coordinates
    new_grid[:, 1, :, :] = 2.0 * new_grid[:, 1, :, :] / max(H - 1, 1) - 1.0

    # Permute grid to (B, H, W, 2) for grid_sample
    new_grid = new_grid.permute(0, 2, 3, 1)

    return F.grid_sample(
        frame, new_grid, mode="bilinear", padding_mode="border", align_corners=True
    )

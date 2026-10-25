import torch
import torch.nn.functional as F


def flow_warp(
    x: torch.Tensor,
    flow: torch.Tensor,
    interp_mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Warp an image or feature map with optical flow.

    Args:
        x: Tensor with size (n, c, h, w)
        flow: Tensor with size (n, h, w, 2), normal value
        interp_mode: 'nearest' or 'bilinear' (default: 'bilinear')
        padding_mode: 'zeros' or 'border' or 'reflection' (default: 'zeros')
        align_corners: Alignment for grid_sample (default: True)

    Returns:
        Warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()

    # Create mesh grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h, dtype=x.dtype, device=x.device),
        torch.arange(0, w, dtype=x.dtype, device=x.device),
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow

    # Scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)

    output = F.grid_sample(
        x,
        vgrid_scaled,
        mode=interp_mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    return output

#!/usr/bin/env python3
"""
Restore an archival clip with pretrained DART.

The input is a folder of frames (sorted by file name) or a video file. Frames are
restored in overlapping temporal chunks (7 frames, stride 3), exactly as in the
evaluation protocol (``scripts/test.py``), and written as PNG files. Optionally the
predicted soft defect masks are saved as well.

Output layout:
  <output>/frames/<name>.png      restored frames
  <output>/masks/<name>.png       predicted defect masks (with --save-masks)
  <output>/restored.mp4           restored video (with --save-video)

Examples:
  python scripts/inference.py --input path/to/frames --output outputs/my_clip
  python scripts/inference.py --input clip.mp4 --output outputs/clip --save-masks --save-video

Runs on a GPU when available and falls back to the CPU (slow, but it works).
"""

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import albumentations as A  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.utils import instantiate  # noqa: E402

from src.utils.weights import load_net_weights  # noqa: E402

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}
# Same input normalization as the datamodule of scripts/test.py (frames -> [-1, 1]).
NORMALIZE = A.Normalize(
    mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0
)


# ── Frame sources ─────────────────────────────────────────────────────────────
def iter_frames(path: Path, max_frames: int | None) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ``(name, RGB uint8 frame)`` from a frame folder or a video file."""
    if path.is_dir():
        files = sorted(
            p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            raise FileNotFoundError(f"No image files found in {path}")
        for i, file in enumerate(files):
            if max_frames is not None and i >= max_frames:
                return
            bgr = cv2.imread(str(file), cv2.IMREAD_COLOR)
            if bgr is None:
                raise OSError(f"Could not read {file}")
            yield file.stem, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif path.suffix.lower() in VIDEO_EXTENSIONS:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise OSError(f"Could not open video {path}")
        i = 0
        while max_frames is None or i < max_frames:
            ok, bgr = capture.read()
            if not ok:
                break
            yield f"frame_{i:06d}", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            i += 1
        capture.release()
    else:
        raise ValueError(f"{path} is neither a folder of frames nor a video file")


def video_fps(path: Path, default: float = 24.0) -> float:
    if path.is_dir():
        return default
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    return fps if fps and fps > 0 else default


# ── Model ─────────────────────────────────────────────────────────────────────
def load_dart(weights: Path, config_name: str, device: torch.device) -> torch.nn.Module:
    """Build the DART network from its Hydra config and load the weights."""
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "configs"), version_base=None
    ):
        # RAFT is part of the DART weights; no separate RAFT checkpoint is needed.
        cfg = compose(
            config_name=config_name, overrides=["optical_flow.ckpt_path=null"]
        )
    net = instantiate(cfg.net)
    load_net_weights(net, weights)
    return net.eval().to(device)


def pad_to_divisor(tensor: torch.Tensor, divisor: int = 16):
    _, _, _, h, w = tensor.shape
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
    if pad_h == 0 and pad_w == 0:
        return tensor, pad
    return F.pad(tensor, pad, mode="constant", value=0), pad


def unpad(tensor: torch.Tensor, pad: tuple[int, int, int, int]) -> torch.Tensor:
    left, right, top, bottom = pad
    _, _, _, h, w = tensor.shape
    return tensor[:, :, :, top : h - bottom, left : w - right]


def assemble_masks(net: torch.nn.Module, num_frames: int) -> torch.Tensor | None:
    """Per-frame soft mask (1, T, 1, H, W) from the cached MaskNet logits.

    ``last_gate_mask_logits_backward[:, k]`` belongs to frame k and
    ``last_gate_mask_logits_forward[:, k]`` to frame k+1 (head frames have no
    gate). Frames covered by both propagation directions get the mean of both.
    """
    back_logits = net.last_gate_mask_logits_backward
    fwd_logits = net.last_gate_mask_logits_forward
    if back_logits is None or fwd_logits is None or num_frames < 2:
        return None
    back = torch.sigmoid(back_logits)  # frames 0..T-2
    fwd = torch.sigmoid(fwd_logits)  # frames 1..T-1
    if num_frames == 2:
        return torch.cat([back[:, 0:1], fwd[:, 0:1]], dim=1)
    middle = (back[:, 1:] + fwd[:, :-1]) / 2
    return torch.cat([back[:, 0:1], middle, fwd[:, -1:]], dim=1)


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def restore(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    net = load_dart(args.weights, args.config, device)

    frames_dir = args.output / "frames"
    masks_dir = args.output / "masks"
    frames_dir.mkdir(parents=True, exist_ok=True)
    if args.save_masks:
        masks_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    fps = video_fps(args.input)

    # Sliding temporal window, identical to the chunk plan of scripts/test.py:
    # the first chunk keeps all frames, later chunks keep only the newly advanced tail.
    overlap = args.chunk_size - args.stride
    source = iter_frames(args.input, args.max_frames)
    buffer: list[tuple[str, np.ndarray]] = []  # frames start, start + 1, ...
    start, exhausted, done = 0, False, 0
    while True:
        while not exhausted and len(buffer) < args.chunk_size:
            try:
                buffer.append(next(source))
            except StopIteration:
                exhausted = True
        end = start + min(args.chunk_size, len(buffer))
        commit_start = 0 if start == 0 else min(end, start + overlap)

        if commit_start < end:
            clip = np.stack(
                [NORMALIZE(image=frame)["image"] for _, frame in buffer[: end - start]]
            )
            if clip.shape[0] < 2:
                raise ValueError("DART needs a clip of at least 2 frames.")
            x = torch.from_numpy(clip)  # (T, H, W, 3) float32 in [-1, 1]
            x = x.permute(0, 3, 1, 2).unsqueeze(0).to(device)  # (1, T, 3, H, W)
            x_padded, pad = pad_to_divisor(x)
            restored = unpad(net(x_padded), pad)
            restored = torch.clamp(restored * 0.5 + 0.5, 0.0, 1.0)
            masks = assemble_masks(net, x.shape[1]) if args.save_masks else None
            if masks is not None:
                masks = unpad(masks, pad)

            for t in range(commit_start - start, end - start):
                name = buffer[t][0]
                rgb = restored[0, t].permute(1, 2, 0).cpu().numpy()
                bgr = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(frames_dir / f"{name}.png"), bgr)
                if masks is not None:
                    mask = masks[0, t, 0].cpu().numpy()
                    cv2.imwrite(
                        str(masks_dir / f"{name}.png"),
                        (mask * 255).round().astype(np.uint8),
                    )
                if args.save_video:
                    if writer is None:
                        h, w = bgr.shape[:2]
                        writer = cv2.VideoWriter(
                            str(args.output / "restored.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (w, h),
                        )
                    writer.write(bgr)
                done += 1
            print(f"  restored {done} frames", flush=True)

        if exhausted and end >= start + len(buffer):
            break
        buffer = buffer[args.stride :]
        start += args.stride

    if writer is not None:
        writer.release()
    print(f"Saved {done} restored frames to {frames_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore archival footage with DART.")
    parser.add_argument(
        "--input", type=Path, required=True, help="Folder of frames or a video file."
    )
    parser.add_argument("--output", type=Path, required=True, help="Output folder.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=REPO_ROOT / "pretrained_models" / "dart.safetensors",
        help="DART weights (.safetensors, .pth or a Lightning .ckpt).",
    )
    parser.add_argument(
        "--config", default="train", help="Hydra config of the network."
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Also save the predicted defect masks.",
    )
    parser.add_argument(
        "--save-video", action="store_true", help="Also write restored.mp4."
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Only restore the first N frames."
    )
    parser.add_argument(
        "--chunk-size", type=int, default=7, help="Temporal window (default: 7)."
    )
    parser.add_argument(
        "--stride", type=int, default=3, help="Window stride (default: 3)."
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:1 or cpu.")
    args = parser.parse_args()
    if not 0 < args.stride <= args.chunk_size:
        parser.error("--stride must be in [1, --chunk-size]")
    return args


if __name__ == "__main__":
    restore(parse_args())

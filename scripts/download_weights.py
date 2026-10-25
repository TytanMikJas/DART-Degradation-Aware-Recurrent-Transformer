"""Download the pretrained DART weights and verify their SHA-256.

    python scripts/download_weights.py                 # dart.safetensors
    python scripts/download_weights.py --format pth    # dart.pth (plain torch state dict)
    python scripts/download_weights.py --verify        # only check an existing file
    python scripts/download_weights.py --extract-raft  # also write raft-sintel.pth for training

The file contains the full DART network, including the frozen RAFT-Sintel flow
estimator (BSD-3-Clause, princeton-vl). Training needs the RAFT weights on their own
(``pretrained_models/raft-sintel.pth``); ``--extract-raft`` writes them from the DART
file, tensor for tensor identical to the official RAFT-Sintel checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

PRETRAINED_DIR = Path(__file__).resolve().parent.parent / "pretrained_models"
GITHUB_RELEASE = (
    "https://github.com/TytanMikJas/DART-Degradation-Aware-Recurrent-Transformer"
    "/releases/download/v1.0.0"
)
HUGGINGFACE_REPO = (
    "https://huggingface.co/TytanMikJas/DART-Degradation-Aware-Recurrent-Transformer"
    "/resolve/main"
)
FILES = {
    "safetensors": (
        "dart.safetensors",
        "d78a388c76ffb2d71cd0fee55f44e3dde695112a40c540a7b67cc0590cfa73d3",
    ),
    "pth": (
        "dart.pth",
        "dcdb10f4a9b047af557eb415a6999c9ea0258c252807c18e368e5a2100926010",
    ),
}
RAFT_PREFIX = "optical_flow.flow_net.model."


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(name: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    for base in (GITHUB_RELEASE, HUGGINGFACE_REPO):
        url = f"{base}/{name}"
        try:
            print(f"Downloading {url}\n         -> {out}")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(out)
            return
        except OSError as exc:  # HTTPError / URLError are OSError subclasses
            tmp.unlink(missing_ok=True)
            print(f"  failed: {exc}", file=sys.stderr)
    raise SystemExit(f"Could not download {name} from GitHub or Hugging Face.")


def extract_raft(weights: Path, out: Path) -> None:
    """Write the bundled RAFT-Sintel weights as a standalone RAFT checkpoint."""
    import torch

    if weights.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(weights))
    else:
        state_dict = torch.load(weights, map_location="cpu", weights_only=True)
    raft = {
        key[len(RAFT_PREFIX) :]: value
        for key, value in state_dict.items()
        if key.startswith(RAFT_PREFIX)
    }
    if not raft:
        raise SystemExit(f"No RAFT weights found in {weights}")
    torch.save(raft, out)
    print(f"OK  {out}  ({len(raft)} RAFT tensors)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--format", choices=sorted(FILES), default="safetensors")
    parser.add_argument("--out-dir", type=Path, default=PRETRAINED_DIR)
    parser.add_argument(
        "--verify", action="store_true", help="Only verify an existing file."
    )
    parser.add_argument(
        "--extract-raft",
        action="store_true",
        help="Also write <out-dir>/raft-sintel.pth (needed for training).",
    )
    args = parser.parse_args()

    name, expected = FILES[args.format]
    out = args.out_dir / name
    if not args.verify:
        download(name, out)
    if not out.exists():
        print(f"File not found: {out}", file=sys.stderr)
        return 1

    digest = sha256sum(out)
    if digest != expected:
        print(
            f"SHA-256 mismatch for {out}\n  expected {expected}\n  got      {digest}",
            file=sys.stderr,
        )
        return 1
    print(f"OK  {out}  (sha256 {digest[:16]}...)")

    if args.extract_raft:
        extract_raft(out, args.out_dir / "raft-sintel.pth")
    return 0


if __name__ == "__main__":
    sys.exit(main())

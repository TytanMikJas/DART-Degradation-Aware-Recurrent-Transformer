# Pretrained models

Weights are not stored in git. Download them into this folder:

```bash
python scripts/download_weights.py                 # dart.safetensors (+ SHA-256 check)
python scripts/download_weights.py --extract-raft  # also raft-sintel.pth (training only)
```

| File | Content | Used by |
|---|---|---|
| `dart.safetensors` | DART network of the paper ("DART (Ours)" in Tables 2–4), fp32, 6.58M parameters incl. the frozen RAFT-Sintel flow estimator | `inference.py`, `test.py` |
| `dart.pth` | Same weights as a plain PyTorch state dict (`--format pth`) | optional |
| `raft-sintel.pth` | RAFT-Sintel weights used to initialise the frozen flow estimator | `train.py` |

SHA-256:

```
d78a388c76ffb2d71cd0fee55f44e3dde695112a40c540a7b67cc0590cfa73d3  dart.safetensors
dcdb10f4a9b047af557eb415a6999c9ea0258c252807c18e368e5a2100926010  dart.pth
```

The weights are also available from the GitHub release `v1.0.0` and from [Hugging Face](https://huggingface.co/TytanMikJas/DART-Degradation-Aware-Recurrent-Transformer).

RAFT: Teed & Deng, *RAFT: Recurrent All-Pairs Field Transforms for Optical Flow*,
ECCV 2020 ([code and original weights](https://github.com/princeton-vl/RAFT), BSD-3-Clause).
The RAFT weights inside `dart.safetensors` are identical to the official `raft-sintel.pth`.

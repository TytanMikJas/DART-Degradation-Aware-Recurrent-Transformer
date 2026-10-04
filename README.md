# DART: A Degradation-Aware Recurrent Transformer for Archival Film Restoration

**ACCV 2026** · [Mikołaj Jastrzębski](https://www.mikjas.com), Wojciech Kozłowski, Kamil Adamczewski · Wrocław University of Science and Technology

[![arXiv](https://img.shields.io/badge/arXiv-2607.21219-b31b1b.svg)](https://arxiv.org/abs/2607.21219)
[![Project page](https://img.shields.io/badge/Project-page-d9a441.svg)](https://www.mikjas.com/dart)
[![Benchmark](https://img.shields.io/badge/Benchmark-AbsoluteDegradation-2b6cb0.svg)](https://github.com/TytanMikJas/AbsoluteDegradation)

> **Status:** code and pretrained weights will be released by the end of October 2026.
> Watch or star the repository to be notified. A video comparison is on the [project page](https://www.mikjas.com/dart).

## Overview

Old film carries compound damage: scratches, dust, blur, noise, flicker and photometric aging. No clean reference exists, and most video restoration models reconstruct frames without knowing where the damage is or how severe it is.

DART makes the damage explicit. It predicts a soft defect mask, propagates it through time, and uses it both to guide temporal fusion and to condition the restoration network on damage location and severity.

## Highlights

- **Explicit damage awareness.** A multi-scale Dilation Pyramid MaskNet, trained with direct continuous-mask supervision, localises film artifacts and estimates their severity.
- **Degradation conditioning.** The restoration backbone is modulated through AdaLN-Zero, driven by the predicted mask, so it adapts to how damaged each frame is.
- **State of the art on real archival footage.** DART outperforms BasicVSR, BasicVSR++, ShiftNet, DeepRemaster, RTN and MambaOFR in no-reference perceptual quality.
- **Compact.** 6.6M parameters and 0.35 GB of memory.

## Release plan

- [ ] Inference code and pretrained weight, training code and configs, Evaluation scripts for AbsoluteDegradation and SRWOV (30.10.2026)

## Related

DART is trained and evaluated with [AbsoluteDegradation](https://github.com/TytanMikJas/AbsoluteDegradation) (NeurIPS 2026, Evaluations and Datasets Track), our physics-inspired film-degradation pipeline and archival benchmark.

## Citation

```bibtex
@inproceedings{jastrzebski2026dart,
  title     = {{DART}: A Degradation-Aware Recurrent Transformer for Archival Film Restoration},
  author    = {Jastrz{\k{e}}bski, Miko{\l}aj and Koz{\l}owski, Wojciech and Adamczewski, Kamil},
  booktitle = {Asian Conference on Computer Vision (ACCV)},
  year      = {2026}
}
```

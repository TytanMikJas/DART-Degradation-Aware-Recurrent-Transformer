import sys
from collections import defaultdict
from itertools import repeat
from pathlib import Path
from typing import Iterable

import albumentations as A
import hydra
import pandas as pd
import pytorch_lightning as pl
import tabulate
import torch
import wandb
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import Logger, WandbLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

# Make the repo root importable so "python scripts/evaluate.py" works from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.grouped_batch_sampler import GroupedBatchSampler  # noqa: E402
from src.data.one_dir_frame_dataset import OneDirFrameDataset  # noqa: E402
from src.metrics.metrics_handler import MetricsHandler  # noqa: E402

ALLOWED_DATASET_PATH_TYPES = ["generated", "gt"]


def _create_dataloader(
    root: Path,
    batch_size: int,
    num_workers: int,
    image_extensions: set[str],
    transform: A.Compose,
) -> DataLoader:
    dataset = OneDirFrameDataset(
        root=root,
        transform=transform,
        image_extensions=image_extensions,
    )
    batch_sampler = GroupedBatchSampler(dataset.get_group_indices(), batch_size)
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )


def _compute_per_image_metrics(
    generated_loader: DataLoader,
    gt_loader: DataLoader | None,
    metrics_handler: MetricsHandler,
    device: torch.device,
) -> dict[str, float]:
    accum: dict[str, dict[str, float]] = defaultdict(lambda: {"sum": 0.0, "count": 0})
    gt_iter = gt_loader if gt_loader is not None else repeat(None)
    data_iter: Iterable[tuple[torch.Tensor, torch.Tensor | None]] = zip(
        generated_loader, gt_iter
    )

    for generated_batch, gt_batch in tqdm(
        data_iter,
        total=len(generated_loader),
        desc="Computing per-image metrics",
        file=sys.stdout,
    ):
        n = generated_batch.shape[0]
        generated_batch = generated_batch.to(device)

        computed_metrics = {}

        if gt_loader:
            gt_batch = gt_batch.to(device)
            computed_metrics |= metrics_handler.eval_fr_metrics(
                generated_batch, gt_batch
            )
        computed_metrics |= metrics_handler.eval_nr_metrics(generated_batch)

        for name, val in computed_metrics.items():
            accum[name]["sum"] += val.sum().item()
            accum[name]["count"] += n

    return {name: s["sum"] / s["count"] for name, s in accum.items()}


def _get_transform(dataset_name: str, config: DictConfig) -> A.Compose:
    per_dataset = config.data.datasets[dataset_name].get("additional_transform", [])
    dataset_config = list(config.data.transform) + list(per_dataset)
    return A.Compose([*[hydra.utils.instantiate(t) for t in dataset_config]])


@hydra.main(config_path="../configs", config_name="evaluate", version_base=None)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)

    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    logger: Logger | None = (
        hydra.utils.instantiate(cfg.logger) if cfg.get("logger") else None
    )
    if isinstance(logger, WandbLogger):
        _ = logger.experiment
    metrics_handler: MetricsHandler = hydra.utils.instantiate(
        cfg.metrics_handler, device=device
    )

    image_extensions = set(cfg.data.image_extensions)
    datasets_config: dict[str, dict[str, dict[str, str]]] = cfg.data.datasets
    results: list[dict[str, str | float]] = []

    for dataset_name, dataset_config in datasets_config.items():
        for key in dataset_config["paths"].keys():
            if key not in ALLOWED_DATASET_PATH_TYPES:
                raise ValueError(
                    f"Invalid key '{key}' found in datasets_config['{dataset_name}']['paths']. "
                    f"Allowed values: {', '.join(ALLOWED_DATASET_PATH_TYPES)}"
                )

    # Sets without restored frames (e.g. SRWOV not downloaded) are skipped.
    available = {}
    for dataset_name, dataset_config in datasets_config.items():
        generated = Path(dataset_config["paths"]["generated"])
        if generated.is_dir():
            available[dataset_name] = dataset_config
        else:
            print(f"Skipping {dataset_name}: {generated} not found.")
    if not available:
        raise FileNotFoundError("No restored frames found; run scripts/test.py first.")
    datasets_config = available

    standard_transform = A.Compose(
        [
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ]
    )

    dataloaders = {
        dataset_name: {
            path_type: _create_dataloader(
                Path(path),
                cfg.data.batch_size,
                cfg.data.num_workers_per_dataset,
                image_extensions,
                A.Compose([_get_transform(dataset_name, cfg), standard_transform]),
            )
            for path_type, path in dataset_config["paths"].items()
        }
        for dataset_name, dataset_config in datasets_config.items()
    }

    for dataset_name, _dataloaders in dataloaders.items():
        generated_loader = _dataloaders.get("generated")
        gt_loader = _dataloaders.get("gt")

        if metrics_handler.get_available_metrics():
            print(f"Evaluating per image metrics for dataset: {dataset_name}...")
            if gt_loader:
                assert len(generated_loader.dataset) == len(gt_loader.dataset), (
                    f"Dataset size mismatch: generated has {len(generated_loader.dataset)} images, "
                    f"GT has {len(gt_loader.dataset)} images. Directories must contain matching frames."
                )
            results += [
                {"Type": dataset_name, "Metric": name, "Value": value}
                for name, value in _compute_per_image_metrics(
                    generated_loader, gt_loader, metrics_handler, device
                ).items()
            ]

    df = pd.DataFrame(results)
    print("----- Metrics Summary -----")
    print(tabulate.tabulate(df, headers="keys", tablefmt="psql"))
    if cfg.get("output_csv"):
        out_csv = Path(cfg.output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"Saved metrics to {out_csv}")
    if logger and isinstance(logger, WandbLogger):
        metrics_table = wandb.Table(dataframe=df)
        logger.experiment.log({"evaluation/metrics_summary": metrics_table})
        logger.experiment.finish()


if __name__ == "__main__":
    main()

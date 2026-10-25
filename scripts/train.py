import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.loggers import Logger, WandbLogger

torch.backends.cudnn.conv.fp32_precision = "tf32"
torch.backends.cuda.matmul.fp32_precision = "ieee"


def _num_devices(cfg: DictConfig) -> int:
    """Number of devices the Trainer will use on this node."""
    devices = cfg.trainer.get("devices", "auto")
    if not torch.cuda.is_available():
        return 1
    if devices in (-1, "-1", "auto"):
        return torch.cuda.device_count()
    if isinstance(devices, (list, ListConfig)):
        return len(devices)
    return int(devices)


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def train(cfg: DictConfig) -> None:
    """
    Main training script driven by Hydra configuration.
    """
    pl.seed_everything(cfg.seed, workers=True)

    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------")

    print("--- Hardware Check ---")
    if torch.cuda.is_available():
        print("CUDA (GPU) is available.")
        print(f"Device count: {torch.cuda.device_count()}")
        print(
            f"Current device: {torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})"
        )
    print("----------------------")

    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)
    mode = None
    if hasattr(datamodule, "degradation_fn"):
        mode = "online" if datamodule.degradation_fn is not None else "offline"
    elif hasattr(cfg.data, "degradation_fn"):
        mode = "online" if cfg.data.degradation_fn is not None else "offline"
    else:
        mode = "offline"
    print(
        f"--- Data mode: {mode} ({'on-the-fly' if mode == 'online' else 'precomputed'}) ---"
    )

    checkpoint_path = cfg.get("checkpoint_path")

    # DDP is only needed with several processes. On a single device (e.g. a
    # Windows workstation, where NCCL is unavailable) fall back to "auto".
    trainer_overrides = {}
    if not torch.cuda.is_available():
        # CPU-only machines (e.g. a quick fast_dev_run smoke test): one process,
        # full precision. GPU training is unaffected.
        print("--- No CUDA device: using devices=1, precision='32-true' ---")
        trainer_overrides["devices"] = 1
        trainer_overrides["precision"] = "32-true"
    if (
        "ddp" in str(cfg.trainer.get("strategy", ""))
        and _num_devices(cfg) <= 1
        and cfg.trainer.get("num_nodes", 1) == 1
    ):
        print("--- Single device detected: using strategy='auto' instead of DDP ---")
        trainer_overrides["strategy"] = "auto"

    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)
    logger: Logger | bool = (
        hydra.utils.instantiate(cfg.logger) if cfg.get("logger") else False
    )
    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer, logger=logger, **trainer_overrides
    )

    if checkpoint_path:
        print(f"--- Resuming from checkpoint: {checkpoint_path} ---")
    trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint_path)

    if isinstance(logger, WandbLogger):
        logger.experiment.finish()


if __name__ == "__main__":
    train()

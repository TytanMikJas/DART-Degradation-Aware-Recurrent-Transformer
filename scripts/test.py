import sys
from pathlib import Path

# Make the repo root importable so "python scripts/test.py" works from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra  # noqa: E402
import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from pytorch_lightning.loggers import Logger, WandbLogger  # noqa: E402

from src.utils.weights import load_net_weights  # noqa: E402

torch.backends.cudnn.conv.fp32_precision = "tf32"
torch.backends.cuda.matmul.fp32_precision = "ieee"


def load_checkpoint(cfg: DictConfig) -> pl.LightningModule:
    """
    Builds the model and loads the DART weights given by ``checkpoint.path``
    (released ``.safetensors`` / ``.pth`` weights or a Lightning ``.ckpt``).
    """
    checkpoint_path = cfg.checkpoint.path
    if checkpoint_path is None:
        raise ValueError("Checkpoint path must be specified in the configuration.")
    print(f"Loading DART weights from: {checkpoint_path}")
    model = hydra.utils.instantiate(cfg.model)
    load_net_weights(model.net, checkpoint_path)
    model.eval()
    return model


@hydra.main(config_path="../configs", config_name="test", version_base=None)
def test(cfg: DictConfig) -> None:
    """
    Main test script driven by Hydra configuration. Loads a model checkpoint and evaluates it on the test set.
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

    # initialize frame output handler singleton
    hydra.utils.instantiate(cfg.frame_output_handler)

    model: pl.LightningModule = load_checkpoint(cfg)
    logger: Logger | bool = (
        hydra.utils.instantiate(cfg.logger) if cfg.get("logger") else False
    )
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)
    datamodule.setup("test")
    trainer.test(model=model, datamodule=datamodule)

    if isinstance(logger, WandbLogger):
        logger.experiment.finish()


if __name__ == "__main__":
    test()

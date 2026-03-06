import hydra
from omegaconf import DictConfig, OmegaConf
import logging

logger = logging.getLogger(__name__)

from src.train import train
from src.evaluation import evaluate
from src.utils import setup_reproducibility
from src.visualization import visualize

import wandb
wandb.login(key="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")


@hydra.main(version_base=None, config_path="configs", config_name="training")
def main(cfg: DictConfig):

    logger.info(OmegaConf.to_yaml(cfg))

    setup_reproducibility(cfg.seed)

    if cfg.get("mode", "train") == "train":
        train(cfg)
    elif cfg.get("mode", "train") == "visualize":
        visualize(cfg)
    else:
        evaluate(cfg)


if __name__ == "__main__":
    main()
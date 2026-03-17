import matplotlib.pyplot as plt
import random
import os
import numpy as np
import torch
import logging


logger = logging.getLogger(__name__)


def viz_image_pairs(dataset, name, num_pairs = 5):
    """
    Visualizes a few seal-schema pairs from the dataset.

    Args:
        dataset: instance of MonogramPairDataset
        num_pairs: number of pairs to show
    """
    def denormalize(img_tensor):
        # Convert [-1,1] back to [0,1] for visualization
        img = img_tensor.clone()
        img = img * 0.5 + 0.5  # undo T.Normalize(mean=0.5, std=0.5)
        return img

    plt.figure(figsize=(8, 4 * num_pairs))

    
    for i in range(num_pairs):
        idx = random.randint(0, len(dataset) - 1)
        schema, seal = dataset[idx]

        schema = denormalize(schema)
        seal = denormalize(seal)

        # convert tensor to HWC for matplotlib
        schema_np = schema.permute(1, 2, 0).numpy()
        seal_np = seal.permute(1, 2, 0).numpy()

        # display schema
        plt.subplot(num_pairs, 2, 2*i + 1)
        plt.imshow(schema_np)
        plt.title("Schema")
        plt.axis('off')

        # display seal
        plt.subplot(num_pairs, 2, 2*i + 2)
        plt.imshow(seal_np)
        plt.title("Seal")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"/scratch/mahantas/cross_modal_retrieval/visualizations/{name}.png")


def setup_reproducibility(seed):
    """
    Sets random seeds for reproducibility.

    Args:
        seed: integer seed value
    """
    seed = int(seed)
    logger.info(f"Setting seed to {seed}")
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed


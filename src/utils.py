import matplotlib.pyplot as plt
import random
import os
import numpy as np
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import wandb


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
        batch = dataset[idx]
        schema, seal = batch["schema"], batch["seal"]

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


def _log_metrics_to_wandb(prefix, metrics):
    wandb.log({
        f"{prefix}/R@1_se2sc": metrics["seal2schema"]["R@1"],
        f"{prefix}/R@5_se2sc": metrics["seal2schema"]["R@5"],
        f"{prefix}/R@10_se2sc": metrics["seal2schema"]["R@10"],
        f"{prefix}/MRR_se2sc": metrics["seal2schema"]["MRR"],
        f"{prefix}/MedianRank_se2sc": metrics["seal2schema"]["MedianRank"],

        f"{prefix}/R@1_sc2se": metrics["schema2seal"]["R@1"],
        f"{prefix}/R@5_sc2se": metrics["schema2seal"]["R@5"],
        f"{prefix}/R@10_sc2se": metrics["schema2seal"]["R@10"],
        f"{prefix}/MRR_sc2se": metrics["schema2seal"]["MRR"],
        f"{prefix}/MedianRank_sc2se": metrics["schema2seal"]["MedianRank"],
    })


def _log_metrics_to_console(title, metrics, topk=None):
    if topk is None:
        logger.info(f"{title} Seal -> Schema")
        for k, v in metrics["seal2schema"].items():
            logger.info(f"{k}: {v:.4f}")

        logger.info(f"{title} Schema -> Seal")
        for k, v in metrics["schema2seal"].items():
            logger.info(f"{k}: {v:.4f}")
    else:
        logger.info(f"{title} Seal -> Schema")
        for k, v in metrics["seal2schema"].items():
            logger.info(f"Reranked_top{topk} {k}: {v:.4f}")

        logger.info(f"{title} Schema -> Seal")
        for k, v in metrics["schema2seal"].items():
            logger.info(f"Reranked_top{topk} {k}: {v:.4f}")



class Attention(nn.Module):
    """
    Attention mechanism computing attention on a CLS token.
    It considers the CLS token as a sequence chunks, and performs attention at the chunks level.
    Each head will be able to attend to different aspects of the feature space, allowing for more complex feature interactions.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, C = x.shape

        # Apply qkv transformation
        qkv = self.qkv(x)

        # Reshape to separate Q, K, V and split heads
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention and combine heads
        x = (attn @ v).transpose(1, 2).reshape(B, C)

        # Final projection
        x = self.proj(x)
        x = self.proj_drop(x)

        return x



class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit (GLU) layer.

    GLUs allow the network to control information flow by learning which information to pass through and which to filter out.
    They can be seen as a learnable activation function that can adapt to the data.
    GLUs can help mitigate the vanishing gradient problem by providing a gating mechanism that allows gradients to flow more easily through the network.
    """

    def __init__(self, input_size, output_size):
        super(GatedLinearUnit, self).__init__()
        self.linear = nn.Linear(input_size, output_size * 2)

    def forward(self, x):
        x = self.linear(x)
        return F.glu(x, dim=-1)
    

class ResidualGatedAttention(nn.Module):
    """
    Residual Gated Attention (RGA) layer.

    RGA layers combine residual connections with gated linear units and attention mechanisms.
    - The attention mechanism allows the network to focus on relevant parts of the input.
    - The GLU provides adaptive gating of information flow.
    - The residual connection (x + residual) helps with gradient flow in deep networks and allows the network to learn incremental transformations.
    - Layer normalization helps stabilize the learning process by normalizing the inputs to each layer.
    """

    def __init__(self, in_features, num_heads=8):
        super().__init__()
        self.attention = Attention(dim=in_features, num_heads=num_heads)
        self.glu = GatedLinearUnit(in_features, in_features)
        self.layer_norm = nn.LayerNorm(in_features)

    def forward(self, x):
        residual = x
        x = self.attention(x)
        x = self.glu(x)
        return self.layer_norm(x + residual)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
import math

from src.models import DualEncoder
from src.dataset import MonogramPairDataset

logger = logging.getLogger(__name__)


# Encode schemas and seals
def encode_data(model, dataloader, device):
    model.eval()
    all_z_schema = []
    all_z_seal = []
    with torch.no_grad():
        for schema, seal in dataloader:
            schema, seal = schema.to(device), seal.to(device)
            z_schema, z_seal = model(schema, seal)
            all_z_schema.append(z_schema.cpu())
            all_z_seal.append(z_seal.cpu())
    
    all_z_schema = torch.cat(all_z_schema)
    all_z_seal = torch.cat(all_z_seal)

    return all_z_schema, all_z_seal


def evaluate_retrieval_accuracy(model, dataloader, device):
    z_schema, z_seal = encode_data(model, dataloader, device)

    # seal -> schema similarity
    similarity_matrix = z_seal @ z_schema.T  # [N, N]
    N = similarity_matrix.size(0)

    # >>> ADDED: Sort once (descending similarity)
    rankings = torch.argsort(similarity_matrix, dim=1, descending=True)

    # >>> ADDED: Compute ranks (position of correct index)
    correct = torch.arange(N).unsqueeze(1)
    ranks = (rankings == correct).nonzero()[:, 1] + 1  # +1 for 1-based rank

    # >>> ADDED: Recall@K
    recall_at_1 = (ranks <= 1).float().mean().item()
    recall_at_5 = (ranks <= 5).float().mean().item()
    recall_at_10 = (ranks <= 10).float().mean().item()

    # >>> ADDED: MRR
    mrr = (1.0 / ranks.float()).mean().item()

    # >>> ADDED: Median Rank
    median_rank = ranks.median().item()

    # ----------------------------------------------------
    # schema -> seal retrieval
    similarity_matrix_T = z_schema @ z_seal.T
    rankings_T = torch.argsort(similarity_matrix_T, dim=1, descending=True)

    ranks_T = (rankings_T == correct).nonzero()[:, 1] + 1

    recall_at_1_T = (ranks_T <= 1).float().mean().item()
    recall_at_5_T = (ranks_T <= 5).float().mean().item()
    recall_at_10_T = (ranks_T <= 10).float().mean().item()

    mrr_T = (1.0 / ranks_T.float()).mean().item()
    median_rank_T = ranks_T.median().item()

    return {
        "seal2schema": {
            "R@1": recall_at_1,
            "R@5": recall_at_5,
            "R@10": recall_at_10,
            "MRR": mrr,
            "MedianRank": median_rank,
        },
        "schema2seal": {
            "R@1": recall_at_1_T,
            "R@5": recall_at_5_T,
            "R@10": recall_at_10_T,
            "MRR": mrr_T,
            "MedianRank": median_rank_T,
        },
    }



def evaluate(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualEncoder(cfg).to(device)

    # Load the trained model
    # state_dict = torch.load(f"/scratch/mahantas/cross_modal_retrieval/checkpoints/fold_{cfg.data.fold}/best_model.pth", map_location=device)
    state_dict = torch.load(cfg.test.checkpoint_dir, map_location=device)
    model.load_state_dict(state_dict)

    # Test dataset
    test_dataset = MonogramPairDataset(
        data_dir=cfg.data.data_dir,
        splits_dir=cfg.data.splits_dir,
        fold = cfg.data.fold,
        split = "test",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size = cfg.data.batch_size, 
        shuffle = False, 
        num_workers = cfg.data.num_workers,
        pin_memory = True
        )


    metrics = evaluate_retrieval_accuracy(model, test_loader, device)
    logger.info("Seal -> Schema")
    for k, v in metrics["seal2schema"].items():
        logger.info(f"{k}: {v:.4f}")

    logger.info("-----------------------------")

    logger.info("Schema -> Seal")
    for k, v in metrics["schema2seal"].items():
        logger.info(f"{k}: {v:.4f}")

if __name__ == "__main__":
    evaluate()
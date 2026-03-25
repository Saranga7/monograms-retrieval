import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging

from src.models import DualEncoder
from src.dataset import MonogramPairDataset
from src.reranker import rerank_topk_with_fused_scores

logger = logging.getLogger(__name__)


def _compute_metrics_from_rankings(rankings: torch.Tensor):
    """
    rankings: [N, N], each row is ranked gallery indices for one query
    assumes GT index = query index
    """
    N = rankings.size(0)
    correct = torch.arange(N, device=rankings.device).unsqueeze(1)
    ranks = (rankings == correct).nonzero(as_tuple=False)[:, 1] + 1  # 1-based

    recall_at_1 = (ranks <= 1).float().mean().item()
    recall_at_5 = (ranks <= 5).float().mean().item()
    recall_at_10 = (ranks <= 10).float().mean().item()
    mrr = (1.0 / ranks.float()).mean().item()
    median_rank = ranks.median().item()

    return {
        "R@1": recall_at_1,
        "R@5": recall_at_5,
        "R@10": recall_at_10,
        "MRR": mrr,
        "MedianRank": median_rank,
    }


# Encode schemas and seals
def encode_data(model, dataloader, device, rerank_mode=None):
    model.eval()

    all_z_schema = []
    all_z_seal = []
    all_schema_tokens = []
    all_seal_tokens = []

    with torch.no_grad():
        for batch in dataloader:
            schema, seal = batch["schema"].to(device), batch["seal"].to(device)
            
            z_schema, z_seal = model(schema, seal)
            all_z_schema.append(z_schema.cpu())
            all_z_seal.append(z_seal.cpu())

            if rerank_mode is None:
                continue

            schema_patch_tokens = model.extract_tokens(schema, model.schema_encoder)["patch_tokens"]  # [B, T, D]
            seal_patch_tokens = model.extract_tokens(seal, model.seal_encoder)["patch_tokens"]        # [B, T, D]
            all_schema_tokens.append(schema_patch_tokens.cpu())
            all_seal_tokens.append(seal_patch_tokens.cpu())
    
    all_z_schema = torch.cat(all_z_schema, dim = 0)
    all_z_seal = torch.cat(all_z_seal, dim = 0)
    if rerank_mode is None:
        return all_z_schema, all_z_seal
    all_schema_tokens = torch.cat(all_schema_tokens, dim = 0)
    all_seal_tokens = torch.cat(all_seal_tokens, dim = 0)

    return all_z_schema, all_z_seal, all_schema_tokens, all_seal_tokens


def evaluate_retrieval_accuracy(
    model,
    dataloader,
    device,
    top_k = None,
    rerank_mode = "none",          # "none", "patch_only", "fused"
    alpha = 0.5,
    normalize_mode = "minmax",
):
    
    if rerank_mode is None:
        z_schema, z_seal = encode_data(model, dataloader, device, rerank_mode)
    else:
        z_schema, z_seal, schema_tokens, seal_tokens = encode_data(model, dataloader, device, rerank_mode)

    # baseline: seal -> schema
    similarity_matrix = z_seal @ z_schema.T   # [N, N]
    rankings = torch.argsort(similarity_matrix, dim=1, descending=True)

    # baseline: schema -> seal
    similarity_matrix_T = z_schema @ z_seal.T
    rankings_T = torch.argsort(similarity_matrix_T, dim=1, descending=True)

    # baseline only
    if top_k is None or rerank_mode == "none":
        baseline_seal2schema = _compute_metrics_from_rankings(rankings)
        baseline_schema2seal = _compute_metrics_from_rankings(rankings_T)
        return {
            "seal2schema": baseline_seal2schema,
            "schema2seal": baseline_schema2seal,
        }

    if rerank_mode not in ["patch_only", "fused"]:
        raise ValueError(f"Unknown rerank_mode: {rerank_mode}")

    if rerank_mode == "patch_only":
        # patch-only is equivalent to fused with alpha=0
        alpha_used = 0.0
    else:
        alpha_used = alpha

    # RERANKED: seal -> schema
    reranked_rankings = rerank_topk_with_fused_scores(
        initial_rankings=rankings,
        initial_similarity=similarity_matrix,
        query_tokens_all=seal_tokens,
        gallery_tokens_all=schema_tokens,
        top_k=top_k,
        alpha=alpha_used,
        normalize_mode=normalize_mode,
    )
    reranked_seal2schema = _compute_metrics_from_rankings(reranked_rankings)

    # RERANKED: schema -> seal
    reranked_rankings_T = rerank_topk_with_fused_scores(
        initial_rankings=rankings_T,
        initial_similarity=similarity_matrix_T,
        query_tokens_all=schema_tokens,
        gallery_tokens_all=seal_tokens,
        top_k=top_k,
        alpha=alpha_used,
        normalize_mode=normalize_mode,
    )
    reranked_schema2seal = _compute_metrics_from_rankings(reranked_rankings_T)

    return {
        "seal2schema": reranked_seal2schema,
        "schema2seal": reranked_schema2seal,
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
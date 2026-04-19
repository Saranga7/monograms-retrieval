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






###------------For generalization regime--------###
#TODO: refactor to avoid code duplication with above functions. Can merge by having optional gallery_loader in the above functions, and if provided, do fixed-gallery evaluation instead of within-batch evaluation.

def _encode_data_with_ids(model, dataloader, device, rerank_mode=None):
    model.eval()

    all_z_schema = []
    all_z_seal = []
    all_pair_ids = []
    all_schema_tokens = []
    all_seal_tokens = []

    with torch.no_grad():
        for batch in dataloader:
            schema = batch["schema"].to(device)
            seal = batch["seal"].to(device)

            z_schema, z_seal = model(schema, seal)
            all_z_schema.append(z_schema.cpu())
            all_z_seal.append(z_seal.cpu())
            all_pair_ids.extend(batch["pair_id"])

            if rerank_mode is None or rerank_mode == "none":
                continue

            schema_patch_tokens = model.extract_tokens(schema, model.schema_encoder)["patch_tokens"]
            seal_patch_tokens = model.extract_tokens(seal, model.seal_encoder)["patch_tokens"]
            all_schema_tokens.append(schema_patch_tokens.cpu())
            all_seal_tokens.append(seal_patch_tokens.cpu())

    all_z_schema = torch.cat(all_z_schema, dim=0)
    all_z_seal = torch.cat(all_z_seal, dim=0)

    if rerank_mode is None or rerank_mode == "none":
        return all_z_schema, all_z_seal, all_pair_ids

    all_schema_tokens = torch.cat(all_schema_tokens, dim=0)
    all_seal_tokens = torch.cat(all_seal_tokens, dim=0)

    return all_z_schema, all_z_seal, all_pair_ids, all_schema_tokens, all_seal_tokens


def _compute_metrics_from_rankings_with_gt(rankings: torch.Tensor, gt_indices: torch.Tensor):
    """
    rankings: [N_query, N_gallery]
    gt_indices: [N_query], each query's correct gallery index
    """
    gt_indices = gt_indices.to(rankings.device)
    correct = gt_indices.unsqueeze(1)
    ranks = (rankings == correct).nonzero(as_tuple=False)[:, 1] + 1

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


def evaluate_retrieval_with_fixed_gallery(
    model,
    query_loader,
    gallery_loader,
    device,
    top_k=None,
    rerank_mode="none",
    alpha=0.5,
    normalize_mode="minmax",
):
    """
    Query items are evaluated against a fixed gallery.
    Matching is done by pair_id.
    """

    if rerank_mode is None:
        rerank_mode = "none"

    if rerank_mode == "none":
        q_schema, q_seal, q_ids = _encode_data_with_ids(model, query_loader, device, rerank_mode="none")
        g_schema, g_seal, g_ids = _encode_data_with_ids(model, gallery_loader, device, rerank_mode="none")
    else:
        q_schema, q_seal, q_ids, q_schema_tokens, q_seal_tokens = _encode_data_with_ids(
            model, query_loader, device, rerank_mode=rerank_mode
        )
        g_schema, g_seal, g_ids, g_schema_tokens, g_seal_tokens = _encode_data_with_ids(
            model, gallery_loader, device, rerank_mode=rerank_mode
        )

    gallery_index = {pid: idx for idx, pid in enumerate(g_ids)}

    valid_query_indices = []
    gt_gallery_indices = []

    for i, pid in enumerate(q_ids):
        if pid in gallery_index:
            valid_query_indices.append(i)
            gt_gallery_indices.append(gallery_index[pid])

    if len(valid_query_indices) == 0:
        raise ValueError("No query pair_ids were found in the gallery.")

    valid_query_indices = torch.tensor(valid_query_indices, dtype=torch.long)
    gt_gallery_indices = torch.tensor(gt_gallery_indices, dtype=torch.long)

    q_schema = q_schema[valid_query_indices]
    q_seal = q_seal[valid_query_indices]

    similarity_matrix = q_seal @ g_schema.T          # seal -> schema
    similarity_matrix_T = q_schema @ g_seal.T        # schema -> seal

    rankings = torch.argsort(similarity_matrix, dim=1, descending=True)
    rankings_T = torch.argsort(similarity_matrix_T, dim=1, descending=True)

    if top_k is None or rerank_mode == "none":
        return {
            "seal2schema": _compute_metrics_from_rankings_with_gt(rankings, gt_gallery_indices),
            "schema2seal": _compute_metrics_from_rankings_with_gt(rankings_T, gt_gallery_indices),
        }

    if rerank_mode not in ["patch_only", "fused"]:
        raise ValueError(f"Unknown rerank_mode: {rerank_mode}")

    if rerank_mode == "patch_only":
        alpha_used = 0.0
    else:
        alpha_used = alpha

    q_schema_tokens = q_schema_tokens[valid_query_indices]
    q_seal_tokens = q_seal_tokens[valid_query_indices]

    reranked_rankings = rerank_topk_with_fused_scores(
        initial_rankings=rankings,
        initial_similarity=similarity_matrix,
        query_tokens_all=q_seal_tokens,
        gallery_tokens_all=g_schema_tokens,
        top_k=top_k,
        alpha=alpha_used,
        normalize_mode=normalize_mode,
    )

    reranked_rankings_T = rerank_topk_with_fused_scores(
        initial_rankings=rankings_T,
        initial_similarity=similarity_matrix_T,
        query_tokens_all=q_schema_tokens,
        gallery_tokens_all=g_seal_tokens,
        top_k=top_k,
        alpha=alpha_used,
        normalize_mode=normalize_mode,
    )

    return {
        "seal2schema": _compute_metrics_from_rankings_with_gt(reranked_rankings, gt_gallery_indices),
        "schema2seal": _compute_metrics_from_rankings_with_gt(reranked_rankings_T, gt_gallery_indices),
    }





   


def evaluate(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualEncoder(cfg).to(device)

    state_dict = torch.load(cfg.test.checkpoint_dir, map_location=device)
    model.load_state_dict(state_dict)

    batch_size = cfg.data.batch_size

    if cfg.data.split_regime == "stratified_5fold":
        test_dataset = MonogramPairDataset(
            data_dir=cfg.data.data_dir,
            splits_dir=cfg.data.splits_dir,
            split_regime=cfg.data.split_regime,
            fold=cfg.data.fold,
            split="test",
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=torch.cuda.is_available()
        )

        metrics = evaluate_retrieval_accuracy(model, test_loader, device)

        logger.info("Seal -> Schema")
        for k, v in metrics["seal2schema"].items():
            logger.info(f"{k}: {v:.4f}")

        logger.info("-----------------------------")

        logger.info("Schema -> Seal")
        for k, v in metrics["schema2seal"].items():
            logger.info(f"{k}: {v:.4f}")

    elif cfg.data.split_regime in {"generalization", "generalization_strict"}:
        gallery_dataset = MonogramPairDataset(
            data_dir=cfg.data.data_dir,
            splits_dir=cfg.data.splits_dir,
            split_regime=cfg.data.split_regime,
            fold=None,
            split="test_gallery",
        )

        gallery_loader = DataLoader(
            gallery_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=torch.cuda.is_available()
        )

        if cfg.data.split_regime == "generalization":
            query_splits = ["test_easy_q0q1", "test_medium_q2", "test_hard_q3"]
        else:
            query_splits = ["test_medium_q2", "test_hard_q3"]

        overall_metrics = evaluate_retrieval_with_fixed_gallery(
            model=model,
            query_loader=gallery_loader,
            gallery_loader=gallery_loader,
            device=device,
        )

        logger.info("[test_gallery -> test_gallery] Seal -> Schema")
        for k, v in overall_metrics["seal2schema"].items():
            logger.info(f"{k}: {v:.4f}")

        logger.info("[test_gallery -> test_gallery] Schema -> Seal")
        for k, v in overall_metrics["schema2seal"].items():
            logger.info(f"{k}: {v:.4f}")

        for split_name in query_splits:
            query_dataset = MonogramPairDataset(
                data_dir=cfg.data.data_dir,
                splits_dir=cfg.data.splits_dir,
                split_regime=cfg.data.split_regime,
                fold=None,
                split=split_name,
            )

            query_loader = DataLoader(
                query_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=cfg.data.num_workers,
                pin_memory=torch.cuda.is_available()
            )

            metrics = evaluate_retrieval_with_fixed_gallery(
                model=model,
                query_loader=query_loader,
                gallery_loader=gallery_loader,
                device=device,
            )

            logger.info(f"[{split_name} -> test_gallery] Seal -> Schema")
            for k, v in metrics["seal2schema"].items():
                logger.info(f"{k}: {v:.4f}")

            logger.info(f"[{split_name} -> test_gallery] Schema -> Seal")
            for k, v in metrics["schema2seal"].items():
                logger.info(f"{k}: {v:.4f}")

if __name__ == "__main__":
    evaluate()
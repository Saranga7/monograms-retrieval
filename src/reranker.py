import torch


def bidirectional_token_match_score(q_tokens, c_tokens):
    """
    q_tokens: [Tq, D]
    c_tokens: [Tc, D]

    Returns a scalar score.
    Assumes tokens are already L2-normalized.
    """
    sim = q_tokens @ c_tokens.T  # [Tq, Tc]

    q_to_c = sim.max(dim=1).values.mean()
    c_to_q = sim.max(dim=0).values.mean()

    return 0.5 * (q_to_c + c_to_q)


def _normalize_scores(scores: torch.Tensor, mode: str = "none", eps: float = 1e-8):
    """
    scores: [K]

    mode:
        - "none": no normalization
        - "minmax": normalize to [0, 1] within top-k
        - "zscore": standardize within top-k
    """
    if mode == "none":
        return scores

    if mode == "minmax":
        s_min = scores.min()
        s_max = scores.max()
        return (scores - s_min) / (s_max - s_min + eps)

    if mode == "zscore":
        mean = scores.mean()
        std = scores.std(unbiased=False)
        return (scores - mean) / (std + eps)

    raise ValueError(f"Unknown normalize mode: {mode}")


def rerank_topk_with_fused_scores(
    initial_rankings: torch.Tensor,      # [N, N]
    initial_similarity: torch.Tensor,    # [N, N], CLS/global similarity matrix
    query_tokens_all: torch.Tensor,      # [N, Tq, D]
    gallery_tokens_all: torch.Tensor,    # [N, Tg, D]
    top_k: int = 10,
    alpha: float = 0.7,
    normalize_mode: str = "minmax",
):
    """
    Rerank only the top-k candidates for each query using a fusion of:
        final_score = alpha * cls_score + (1 - alpha) * patch_score

    Everything after top-k stays unchanged.

    alpha:
        weight for CLS/global score
        patch weight = (1 - alpha)
    """
    N = initial_rankings.size(0)
    reranked_rankings = []

    for i in range(N):
        topk_idx = initial_rankings[i, :top_k]  # [K]

        # Original stage-1 CLS scores for the shortlisted candidates
        cls_scores = initial_similarity[i, topk_idx]  # [K]

        # Query tokens and candidate tokens
        query_tokens = query_tokens_all[i]                 # [Tq, D]
        candidate_tokens = gallery_tokens_all[topk_idx]   # [K, Tg, D]

        patch_scores = []
        for j in range(candidate_tokens.size(0)):
            s = bidirectional_token_match_score(query_tokens, candidate_tokens[j])
            patch_scores.append(s)

        patch_scores = torch.stack(patch_scores, dim=0)  # [K]

        # Normalize within top-k so one score type does not dominate just by scale
        cls_scores_norm = _normalize_scores(cls_scores, mode=normalize_mode)
        patch_scores_norm = _normalize_scores(patch_scores, mode=normalize_mode)

        fused_scores = alpha * cls_scores_norm + (1.0 - alpha) * patch_scores_norm

        rerank_order = torch.argsort(fused_scores, descending=True)

        reranked_topk = topk_idx[rerank_order]
        remaining = initial_rankings[i, top_k:]
        full_reranked = torch.cat([reranked_topk, remaining], dim=0)

        reranked_rankings.append(full_reranked)

    reranked_rankings = torch.stack(reranked_rankings, dim=0)  # [N, N]
    return reranked_rankings
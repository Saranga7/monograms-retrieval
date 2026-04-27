from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from tqdm import tqdm

from src.models import DualEncoder
from src.dataset import MonogramPairDataset


# =========================================================c
# CONFIG
# =========================================================

RUN_DIR = Path(
    "multirun/training/2026-04-16/"
    "stratified_23-09-27_dinov3_H+_emb512_bestmodel"
)

FOLD = 4
SPLIT = "test"

TOP_K = 10
NUM_QUERIES = 5
BATCH_SIZE = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CFG_PATH = RUN_DIR / ".hydra" / "config.yaml"
CHECKPOINT_PATH = RUN_DIR / "checkpoints" / f"fold_{FOLD}" / "best_model.pth"
OUTPUT_DIR = Path("visualizations") / "patch_similarity_visualizations" / f"fold_{FOLD}"


# =========================================================
# HEATMAP UTILS
# =========================================================

def denormalize_tensor(img_tensor, use_grayscale=False):
    x = img_tensor.detach().cpu().clone()

    if use_grayscale:
        mean = torch.tensor([0.5, 0.5, 0.5])[:, None, None]
        std = torch.tensor([0.5, 0.5, 0.5])[:, None, None]
    else:
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    x = x * std + mean
    x = x.clamp(0, 1)
    return (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def normalize_heatmap(hm):
    hm = hm - hm.min()
    hm = hm / (hm.max() + 1e-8)
    return hm


def overlay_heatmap(image_rgb, heatmap, alpha=0.45):
    h, w = image_rgb.shape[:2]

    heatmap = normalize_heatmap(heatmap)
    heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)

    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    return cv2.addWeighted(image_rgb, 1 - alpha, heatmap_color, alpha, 0)


# =========================================================
# PATCH SIMILARITY
# =========================================================

@torch.no_grad()
def patch_similarity_heatmaps(model, schema, seal):
    schema_out = model.extract_tokens_for_viz(schema, model.schema_encoder)
    seal_out = model.extract_tokens_for_viz(seal, model.seal_encoder)

    schema_tokens = schema_out["patch_tokens"]
    seal_tokens = seal_out["patch_tokens"]

    schema_tokens = F.normalize(schema_tokens, dim=-1)
    seal_tokens = F.normalize(seal_tokens, dim=-1)

    sim = schema_tokens[0] @ seal_tokens[0].T

    schema_scores = sim.max(dim=1).values
    seal_scores = sim.max(dim=0).values

    schema_hm = schema_scores.reshape(
        schema_out["grid_h"],
        schema_out["grid_w"],
    ).cpu().numpy()

    seal_hm = seal_scores.reshape(
        seal_out["grid_h"],
        seal_out["grid_w"],
    ).cpu().numpy()

    return schema_hm, seal_hm


# =========================================================
# DATA
# =========================================================

def build_dataset(cfg, split):
    return MonogramPairDataset(
        data_dir=cfg.data.data_dir,
        splits_dir=cfg.data.splits_dir,
        split_regime=cfg.data.split_regime,
        fold=cfg.data.fold,
        split=split,
        transform=True,
        return_paths=False,
        use_processed_seals=False,
        kwargs={
            "use_strong_augmentation": cfg.data.transforms.use_strong_augmentation,
            "use_grayscale": cfg.data.transforms.use_grayscale,
        },
    )


@torch.no_grad()
def compute_embeddings(model, loader, device):
    schema_embs = []
    seal_embs = []
    pair_ids = []
    schema_imgs = []
    seal_imgs = []

    for batch in tqdm(loader, desc="Computing embeddings"):
        schema = batch["schema"].to(device)
        seal = batch["seal"].to(device)

        z_schema = model.encode_schema(schema)
        z_seal = model.encode_seal(seal)

        schema_embs.append(z_schema.cpu())
        seal_embs.append(z_seal.cpu())

        pair_ids.extend(batch["pair_id"])
        schema_imgs.append(batch["schema"].cpu())
        seal_imgs.append(batch["seal"].cpu())

    return {
        "schema_embs": torch.cat(schema_embs, dim=0),
        "seal_embs": torch.cat(seal_embs, dim=0),
        "pair_ids": pair_ids,
        "schema_imgs": torch.cat(schema_imgs, dim=0),
        "seal_imgs": torch.cat(seal_imgs, dim=0),
    }


# =========================================================
# FIGURE SAVING
# =========================================================

def save_patch_similarity_figure(
    model,
    schema_tensor,
    seal_tensor,
    schema_id,
    seal_id,
    output_path,
    device,
    use_grayscale=False,
    title=None,
):
    schema = schema_tensor.unsqueeze(0).to(device)
    seal = seal_tensor.unsqueeze(0).to(device)

    schema_hm, seal_hm = patch_similarity_heatmaps(model, schema, seal)

    schema_rgb = denormalize_tensor(schema_tensor, use_grayscale=use_grayscale)
    seal_rgb = denormalize_tensor(seal_tensor, use_grayscale=use_grayscale)

    schema_overlay = overlay_heatmap(schema_rgb, schema_hm)
    seal_overlay = overlay_heatmap(seal_rgb, seal_hm)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    axes[0].imshow(seal_rgb)
    axes[0].set_title(f"Seal\n{seal_id}")

    axes[1].imshow(seal_overlay)
    axes[1].set_title("Seal heatmap")

    axes[2].imshow(schema_rgb)
    axes[2].set_title(f"Schema\n{schema_id}")

    axes[3].imshow(schema_overlay)
    axes[3].set_title("Schema heatmap")

    for ax in axes:
        ax.axis("off")

    if title is not None:
        fig.suptitle(title, fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# MAIN
# =========================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(CFG_PATH)
    cfg.data.fold = FOLD

    dataset = build_dataset(cfg, SPLIT)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = DualEncoder(cfg).to(DEVICE)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    data = compute_embeddings(model, loader, DEVICE)

    schema_embs = F.normalize(data["schema_embs"], dim=-1)
    seal_embs = F.normalize(data["seal_embs"], dim=-1)

    pair_ids = data["pair_ids"]
    schema_imgs = data["schema_imgs"]
    seal_imgs = data["seal_imgs"]

    sim = seal_embs @ schema_embs.T

    n_queries = min(NUM_QUERIES, len(pair_ids))
    use_grayscale = cfg.data.transforms.use_grayscale

    for q_idx in range(n_queries):
        query_id = pair_ids[q_idx]
        query_dir = OUTPUT_DIR / f"query_{q_idx:04d}_{query_id}"
        query_dir.mkdir(parents=True, exist_ok=True)

        scores = sim[q_idx]
        top_scores, top_indices = torch.topk(scores, k=TOP_K)

        save_patch_similarity_figure(
            model=model,
            schema_tensor=schema_imgs[q_idx],
            seal_tensor=seal_imgs[q_idx],
            schema_id=pair_ids[q_idx],
            seal_id=query_id,
            output_path=query_dir / "ground_truth.png",
            device=DEVICE,
            use_grayscale=use_grayscale,
            title=f"Ground truth pair | query={query_id}",
        )

        for rank, (score, schema_idx) in enumerate(zip(top_scores, top_indices), start=1):
            schema_idx = int(schema_idx)
            retrieved_id = pair_ids[schema_idx]
            label = "correct" if schema_idx == q_idx else "wrong"

            save_patch_similarity_figure(
                model=model,
                schema_tensor=schema_imgs[schema_idx],
                seal_tensor=seal_imgs[q_idx],
                schema_id=retrieved_id,
                seal_id=query_id,
                output_path=query_dir / f"top_{rank}_{label}_score_{float(score):.3f}.png",
                device=DEVICE,
                use_grayscale=use_grayscale,
                title=f"Top-{rank} | score={float(score):.3f} | {label}",
            )

        print(f"Saved query {q_idx}: {query_id}")

    print(f"\nDone. Outputs saved to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
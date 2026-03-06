import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from tqdm import tqdm
import matplotlib.pyplot as plt
import umap
import plotly.express as px
import pandas as pd
import numpy as np

from src.dataset import MonogramPairDataset
from src.models import DualEncoder


def visualize(cfg, checkpoint_path=None, split="test", output_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MonogramPairDataset(
        data_dir=cfg.data.data_dir,
        splits_dir=cfg.data.splits_dir if split != "all" else None,
        fold=cfg.data.fold if split != "all" else None,
        split=split,
        return_paths=True
    )

    batch_size = len(dataset) if cfg.data.batch_size == "full" else cfg.data.batch_size

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True
    )

    print(f"Visualization dataset size: {len(dataset)}")

    model = DualEncoder(cfg).to(device)

    ckpt_path = checkpoint_path if checkpoint_path is not None else cfg.test.checkpoint_path
    checkpoint = torch.load(ckpt_path, map_location=device)

    # supports both raw state_dict and richer checkpoint dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    records = []
    pair_lines = []

    with torch.no_grad():
        for schema, seal, schema_paths, seal_paths in tqdm(loader):
            schema = schema.to(device)
            seal = seal.to(device)

            z_schema, z_seal = model(schema, seal)

            z_schema = z_schema.cpu()
            z_seal = z_seal.cpu()

            for i in range(len(schema_paths)):
                schema_path = schema_paths[i]
                seal_path = seal_paths[i]

                pair_id = os.path.splitext(os.path.basename(schema_path))[0]

                schema_idx = len(records)
                records.append({
                    "pair_id": pair_id,
                    "type": "schema",
                    "path": schema_path,
                    "emb_0": z_schema[i].numpy()
                })

                seal_idx = len(records)
                records.append({
                    "pair_id": pair_id,
                    "type": "seal",
                    "path": seal_path,
                    "emb_0": z_seal[i].numpy()
                })

                pair_lines.append((schema_idx, seal_idx))

    emb = [r["emb_0"] for r in records]
    emb = pd.DataFrame({"embedding": emb})

    
    all_emb = np.stack(emb["embedding"].values, axis=0)

    # output directory
    if output_dir is None:
        output_dir = Path("visualizations") / f"fold_{cfg.data.fold}" / split
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- t-SNE ----------
    tsne = TSNE(
        n_components=2,
        perplexity=min(20, max(5, (len(all_emb) - 1) // 3)),
        random_state=cfg.seed
    )
    proj_tsne = tsne.fit_transform(all_emb)

    df_tsne = pd.DataFrame({
        "x": proj_tsne[:, 0],
        "y": proj_tsne[:, 1],
        "pair_id": [r["pair_id"] for r in records],
        "type": [r["type"] for r in records],
        "path": [r["path"] for r in records],
    })

    df_tsne.to_csv(output_dir / "tsne_coordinates.csv", index=False)

    plt.figure(figsize=(10, 8))

    for t in ["schema", "seal"]:
        mask = df_tsne["type"] == t
        plt.scatter(
            df_tsne.loc[mask, "x"],
            df_tsne.loc[mask, "y"],
            s=20,
            label=t,
            alpha=0.8
        )

    for i, j in pair_lines:
        plt.plot(
            [df_tsne.loc[i, "x"], df_tsne.loc[j, "x"]],
            [df_tsne.loc[i, "y"], df_tsne.loc[j, "y"]],
            linewidth=0.5,
            alpha=0.2
        )

    plt.title(f"t-SNE of Embedding Space ({split})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "tsne_plot.png", dpi=300)
    plt.close()

    fig_tsne = px.scatter(
        df_tsne,
        x="x",
        y="y",
        color="type",
        hover_data=["pair_id", "path"]
    )
    fig_tsne.write_html(output_dir / "tsne_interactive.html")

    # ---------- UMAP ----------
    reducer = umap.UMAP(
        n_neighbors=min(15, max(5, len(all_emb) // 10)),
        min_dist=0.1,
        random_state=cfg.seed
    )
    proj_umap = reducer.fit_transform(all_emb)

    df_umap = pd.DataFrame({
        "x": proj_umap[:, 0],
        "y": proj_umap[:, 1],
        "pair_id": [r["pair_id"] for r in records],
        "type": [r["type"] for r in records],
        "path": [r["path"] for r in records],
    })

    df_umap.to_csv(output_dir / "umap_coordinates.csv", index=False)

    plt.figure(figsize=(10, 8))

    for t in ["schema", "seal"]:
        mask = df_umap["type"] == t
        plt.scatter(
            df_umap.loc[mask, "x"],
            df_umap.loc[mask, "y"],
            s=20,
            label=t,
            alpha=0.8
        )

    for i, j in pair_lines:
        plt.plot(
            [df_umap.loc[i, "x"], df_umap.loc[j, "x"]],
            [df_umap.loc[i, "y"], df_umap.loc[j, "y"]],
            linewidth=0.5,
            alpha=0.2
        )

    plt.title(f"UMAP of Embedding Space ({split})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "umap_plot.png", dpi=300)
    plt.close()

    fig_umap = px.scatter(
        df_umap,
        x="x",
        y="y",
        color="type",
        hover_data=["pair_id", "path"]
    )
    fig_umap.write_html(output_dir / "umap_interactive.html")

    print(f"Visualizations saved in: {output_dir}")
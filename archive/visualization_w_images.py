import os
import base64
from io import BytesIO
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from tqdm import tqdm
import matplotlib.pyplot as plt
import umap
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np 
from PIL import Image

from src.dataset import MonogramPairDataset
from src.models import DualEncoder


def image_to_base64(image_path, max_size=(120, 120)):
    """
    Convert an image to a small base64 thumbnail for Plotly hover.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail(max_size)

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        print(f"Warning: could not encode image {image_path}: {e}")
        return ""


def save_static_plot(df, pair_lines, output_path, title, draw_lines=False):
    """
    Save static matplotlib scatter plot.
    If draw_lines=True, draw schema-seal pair lines.
    """
    plt.figure(figsize=(10, 8))

    for t in sorted(df["type"].unique()):
        mask = df["type"] == t
        plt.scatter(
            df.loc[mask, "x"],
            df.loc[mask, "y"],
            s=20,
            label=t,
            alpha=0.8
        )

    if draw_lines:
        for i, j in pair_lines:
            if i in df.index and j in df.index:
                plt.plot(
                    [df.loc[i, "x"], df.loc[j, "x"]],
                    [df.loc[i, "y"], df.loc[j, "y"]],
                    linewidth=0.5,
                    alpha=0.2
                )

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_interactive_plot(df, pair_lines, output_path, title, draw_lines=False):
    """
    Save interactive Plotly scatter plot with image hover.
    If draw_lines=True, add schema-seal pair lines.
    """
    fig = go.Figure()

    types_present = sorted(df["type"].unique())

    for t in types_present:
        dft = df[df["type"] == t]

        fig.add_trace(
            go.Scattergl(
                x=dft["x"],
                y=dft["y"],
                mode="markers",
                name=t,
                marker=dict(size=7),
                customdata=np.stack(
                    [
                        dft["pair_id"].values,
                        dft["type"].values,
                        dft["path"].values,
                        dft["image_b64"].values,
                    ],
                    axis=-1
                ),
                hovertemplate=
                "<b>%{customdata[0]}</b><br>" +
                "Type: %{customdata[1]}<br>" +
                "Path: %{customdata[2]}<br><br>" +
                "<img src='%{customdata[3]}' width='140'><br>" +
                "<extra></extra>",
            )
        )

    if draw_lines:
        for i, j in pair_lines:
            if i in df.index and j in df.index:
                fig.add_trace(
                    go.Scattergl(
                        x=[df.loc[i, "x"], df.loc[j, "x"]],
                        y=[df.loc[i, "y"], df.loc[j, "y"]],
                        mode="lines",
                        line=dict(width=1),
                        opacity=0.2,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

    fig.update_layout(
        title=title,
        template="plotly_white",
        width=1000,
        height=800,
    )

    fig.write_html(output_path)


def build_projection_dataframe(records, pair_lines, projection, method_name):
    """
    Build dataframe from 2D projection coordinates.
    """
    df = pd.DataFrame({
        "x": projection[:, 0],
        "y": projection[:, 1],
        "pair_id": [r["pair_id"] for r in records],
        "type": [r["type"] for r in records],
        "path": [r["path"] for r in records],
        "image_b64": [r["image_b64"] for r in records],
    })

    df.index = np.arange(len(df))
    return df


def visualize(cfg, checkpoint_path = None, split="all", output_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MonogramPairDataset(
        data_dir=cfg.data.data_dir,
        splits_dir=cfg.data.splits_dir if split != "all" else None,
        fold=cfg.data.fold if split != "all" else None,
        split=split,
        return_paths=True,
        use_processed_seals=cfg.train.use_processed_seals,
        kwargs={
            "use_strong_augmentation": cfg.data.transforms.use_strong_augmentation,
            "use_grayscale": cfg.data.transforms.use_grayscale,
        }
    )

    batch_size = len(dataset) if cfg.data.batch_size == "full" else cfg.data.batch_size

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    print(f"Visualization dataset size: {len(dataset)}")

    model = DualEncoder(cfg).to(device)

    print("Using checkpoint:", checkpoint_path if checkpoint_path is not None else cfg.test.checkpoint_path)
    ckpt_path = checkpoint_path if checkpoint_path is not None else cfg.test.checkpoint_path
    checkpoint = torch.load(ckpt_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    records = []
    pair_lines = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting embeddings"):
            schema = batch["schema"].to(device)
            seal = batch["seal"].to(device)
            schema_paths = batch["schema_path"]
            seal_paths = batch["seal_path"]

            z_schema, z_seal = model(schema, seal)

            z_schema = z_schema.cpu().numpy()
            z_seal = z_seal.cpu().numpy()

            for i in range(len(schema_paths)):
                schema_path = schema_paths[i]
                seal_path = seal_paths[i]

                pair_id = os.path.splitext(os.path.basename(schema_path))[0]

                schema_idx = len(records)
                records.append({
                    "pair_id": pair_id,
                    "type": "schema",
                    "path": schema_path,
                    "embedding": z_schema[i],
                    "image_b64": image_to_base64(schema_path, max_size=(120, 120)),
                })

                seal_idx = len(records)
                records.append({
                    "pair_id": pair_id,
                    "type": "seal",
                    "path": seal_path,
                    "embedding": z_seal[i],
                    "image_b64": image_to_base64(seal_path, max_size=(120, 120)),
                })

                pair_lines.append((schema_idx, seal_idx))

    all_emb = np.stack([r["embedding"] for r in records], axis=0)

    if output_dir is None:
        if split == "all":
            output_dir = Path("visualizations") / split
        else:
            output_dir = Path("visualizations") / f"fold_{cfg.data.fold}" / split
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # t-SNE
    # -----------------------------
    tsne = TSNE(
        n_components=2,
        perplexity=min(20, max(5, (len(all_emb) - 1) // 3)),
        random_state=cfg.seed,
    )
    proj_tsne = tsne.fit_transform(all_emb)

    df_tsne = build_projection_dataframe(records, pair_lines, proj_tsne, "tsne")
    df_tsne.to_csv(output_dir / "tsne_coordinates.csv", index=False)

    # schema only
    df_tsne_schema = df_tsne[df_tsne["type"] == "schema"].copy()
    df_tsne_schema.to_csv(output_dir / "tsne_schema_coordinates.csv", index=False)
    save_static_plot(
        df_tsne_schema,
        pair_lines=[],
        output_path=output_dir / "tsne_schema_plot.png",
        title=f"t-SNE of Schema Embeddings ({split})",
        draw_lines=False,
    )
    save_interactive_plot(
        df_tsne_schema,
        pair_lines=[],
        output_path=output_dir / "tsne_schema_interactive.html",
        title=f"t-SNE of Schema Embeddings ({split})",
        draw_lines=False,
    )

    # seal only
    df_tsne_seal = df_tsne[df_tsne["type"] == "seal"].copy()
    df_tsne_seal.to_csv(output_dir / "tsne_seal_coordinates.csv", index=False)
    save_static_plot(
        df_tsne_seal,
        pair_lines=[],
        output_path=output_dir / "tsne_seal_plot.png",
        title=f"t-SNE of Seal Embeddings ({split})",
        draw_lines=False,
    )
    save_interactive_plot(
        df_tsne_seal,
        pair_lines=[],
        output_path=output_dir / "tsne_seal_interactive.html",
        title=f"t-SNE of Seal Embeddings ({split})",
        draw_lines=False,
    )

    # both
    save_static_plot(
        df_tsne,
        pair_lines=pair_lines,
        output_path=output_dir / "tsne_both_plot.png",
        title=f"t-SNE of Embedding Space ({split})",
        draw_lines=True,
    )
    save_interactive_plot(
        df_tsne,
        pair_lines=pair_lines,
        output_path=output_dir / "tsne_both_interactive.html",
        title=f"t-SNE of Embedding Space ({split})",
        draw_lines=True,
    )

    # -----------------------------
    # UMAP
    # -----------------------------
    reducer = umap.UMAP(
        n_neighbors=min(15, max(5, len(all_emb) // 10)),
        min_dist=0.1,
        random_state=cfg.seed,
    )
    proj_umap = reducer.fit_transform(all_emb)

    df_umap = build_projection_dataframe(records, pair_lines, proj_umap, "umap")
    df_umap.to_csv(output_dir / "umap_coordinates.csv", index=False)

    # schema only
    df_umap_schema = df_umap[df_umap["type"] == "schema"].copy()
    df_umap_schema.to_csv(output_dir / "umap_schema_coordinates.csv", index=False)
    save_static_plot(
        df_umap_schema,
        pair_lines=[],
        output_path=output_dir / "umap_schema_plot.png",
        title=f"UMAP of Schema Embeddings ({split})",
        draw_lines=False,
    )
    save_interactive_plot(
        df_umap_schema,
        pair_lines=[],
        output_path=output_dir / "umap_schema_interactive.html",
        title=f"UMAP of Schema Embeddings ({split})",
        draw_lines=False,
    )

    # seal only
    df_umap_seal = df_umap[df_umap["type"] == "seal"].copy()
    df_umap_seal.to_csv(output_dir / "umap_seal_coordinates.csv", index=False)
    save_static_plot(
        df_umap_seal,
        pair_lines=[],
        output_path=output_dir / "umap_seal_plot.png",
        title=f"UMAP of Seal Embeddings ({split})",
        draw_lines=False,
    )
    save_interactive_plot(
        df_umap_seal,
        pair_lines=[],
        output_path=output_dir / "umap_seal_interactive.html",
        title=f"UMAP of Seal Embeddings ({split})",
        draw_lines=False,
    )

    # both
    save_static_plot(
        df_umap,
        pair_lines=pair_lines,
        output_path=output_dir / "umap_both_plot.png",
        title=f"UMAP of Embedding Space ({split})",
        draw_lines=True,
    )
    save_interactive_plot(
        df_umap,
        pair_lines=pair_lines,
        output_path=output_dir / "umap_both_interactive.html",
        title=f"UMAP of Embedding Space ({split})",
        draw_lines=True,
    )

    print(f"Visualizations saved in: {output_dir}")
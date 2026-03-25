import os
import json
import base64
from io import BytesIO
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from tqdm import tqdm
import matplotlib.pyplot as plt
import umap
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from PIL import Image

from src.dataset import MonogramPairDataset
from src.models import DualEncoder


def image_to_base64(image_path, max_size=(180, 180)):
    """
    Convert an image to a small base64 thumbnail for the HTML inspector panel.
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


def _make_html_wrapper(fig, title, include_pair_panel=False):
    """
    Build a self-contained HTML page:
    - Plotly figure embedded locally
    - metadata on hover
    - image(s) shown in an inspector panel on click
    - inspector shown on the right side
    """
    plot_div_id = "plotly-div"

    plot_html = fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        div_id=plot_div_id,
        config={"responsive": True}
    )

    if include_pair_panel:
        pair_html = """
        <div class="panel-image-block">
            <div class="panel-label">Paired image</div>
            <img id="pair-image" class="panel-image" src="" alt="Paired image will appear here"/>
        </div>
        """
    else:
        pair_html = ""

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background: #ffffff;
            color: #222;
        }}

        .container {{
            width: 100%;
            max-width: 1800px;
            margin: 0 auto;
            padding: 16px;
            box-sizing: border-box;
        }}

        .main-layout {{
            display: flex;
            flex-direction: row;
            align-items: flex-start;
            gap: 20px;
        }}

        .plot-panel {{
            flex: 1 1 auto;
            min-width: 0;
        }}

        .inspector {{
            width: 320px;
            flex: 0 0 320px;
            padding: 16px;
            border: 1px solid #ddd;
            border-radius: 10px;
            background: #fafafa;
            box-sizing: border-box;
            position: sticky;
            top: 16px;
            max-height: calc(100vh - 32px);
            overflow-y: auto;
        }}

        .inspector h3 {{
            margin: 0 0 12px 0;
            font-size: 18px;
        }}

        .meta {{
            margin-bottom: 14px;
            line-height: 1.5;
            word-break: break-word;
            font-size: 14px;
        }}

        .image-row {{
            display: flex;
            flex-direction: column;
            gap: 18px;
            align-items: stretch;
        }}

        .panel-image-block {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}

        .panel-label {{
            font-size: 14px;
            font-weight: 600;
        }}

        .panel-image {{
            width: 100%;
            max-width: 100%;
            border: 1px solid #ccc;
            border-radius: 6px;
            background: white;
            object-fit: contain;
        }}

        .hint {{
            color: #666;
            font-size: 14px;
        }}

        @media (max-width: 1100px) {{
            .main-layout {{
                flex-direction: column;
            }}

            .inspector {{
                width: 100%;
                flex: 1 1 auto;
                position: static;
                max-height: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="main-layout">
            <div class="plot-panel">
                {plot_html}
            </div>

            <div class="inspector">
                <h3>Selected point</h3>
                <div id="meta" class="meta">
                    <span class="hint">Click a point to inspect the image.</span>
                </div>
                <div class="image-row">
                    <div class="panel-image-block">
                        <div class="panel-label">Selected image</div>
                        <img id="main-image" class="panel-image" src="" alt="Selected image will appear here"/>
                    </div>
                    {pair_html}
                </div>
            </div>
        </div>
    </div>

    <script>
        window.addEventListener("load", function() {{
            const gd = document.getElementById("{plot_div_id}");
            if (!gd) {{
                console.error("Plot div not found.");
                return;
            }}

            gd.on('plotly_click', function(evt) {{
                if (!evt || !evt.points || evt.points.length === 0) return;

                const pt = evt.points[0];
                const cd = pt.customdata;

                // customdata:
                // [pair_id, type, path, image_b64, pair_type, pair_path, pair_image_b64]
                const pairId = cd[0] || "";
                const itemType = cd[1] || "";
                const itemPath = cd[2] || "";
                const itemImage = cd[3] || "";
                const pairType = cd[4] || "";
                const pairPath = cd[5] || "";
                const pairImage = cd[6] || "";

                const meta = document.getElementById('meta');
                const mainImg = document.getElementById('main-image');

                let metaHtml = `
                    <div><b>Pair ID:</b> ${{pairId}}</div>
                    <div><b>Type:</b> ${{itemType}}</div>
                    <div><b>Path:</b> ${{itemPath}}</div>
                `;

                if (pairType || pairPath) {{
                    metaHtml += `
                        <div style="margin-top:8px;"><b>Paired type:</b> ${{pairType}}</div>
                        <div><b>Paired path:</b> ${{pairPath}}</div>
                    `;
                }}

                meta.innerHTML = metaHtml;
                mainImg.src = itemImage || "";

                const pairImg = document.getElementById('pair-image');
                if (pairImg) {{
                    pairImg.src = pairImage || "";
                }}
            }});
        }});
    </script>
</body>
</html>
"""
    return html


# def _make_html_wrapper(fig, title, include_pair_panel=False):
#     """
#     Build a self-contained HTML page:
#     - Plotly figure
#     - metadata on hover
#     - image(s) on click in a fixed inspector panel
#     """
#     fig_json = fig.to_json()

#     if include_pair_panel:
#         pair_html = """
#         <div class="panel-image-block">
#             <div class="panel-label">Paired image</div>
#             <img id="pair-image" class="panel-image" src="" alt="Paired image will appear here"/>
#         </div>
#         """
#     else:
#         pair_html = ""

#     html = f"""
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="utf-8"/>
#     <title>{title}</title>
#     <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
#     <style>
#         body {{
#             font-family: Arial, sans-serif;
#             margin: 0;
#             padding: 0;
#             background: #ffffff;
#             color: #222;
#         }}
#         .container {{
#             width: 100%;
#             max-width: 1400px;
#             margin: 0 auto;
#             padding: 16px;
#             box-sizing: border-box;
#         }}
#         #plot {{
#             width: 100%;
#             height: 820px;
#         }}
#         .inspector {{
#             margin-top: 14px;
#             padding: 16px;
#             border: 1px solid #ddd;
#             border-radius: 10px;
#             background: #fafafa;
#         }}
#         .inspector h3 {{
#             margin: 0 0 12px 0;
#             font-size: 18px;
#         }}
#         .meta {{
#             margin-bottom: 14px;
#             line-height: 1.5;
#             word-break: break-word;
#         }}
#         .image-row {{
#             display: flex;
#             gap: 24px;
#             flex-wrap: wrap;
#             align-items: flex-start;
#         }}
#         .panel-image-block {{
#             display: flex;
#             flex-direction: column;
#             gap: 8px;
#         }}
#         .panel-label {{
#             font-size: 14px;
#             font-weight: 600;
#         }}
#         .panel-image {{
#             width: 220px;
#             max-width: 100%;
#             border: 1px solid #ccc;
#             border-radius: 6px;
#             background: white;
#             object-fit: contain;
#         }}
#         .hint {{
#             color: #666;
#             font-size: 14px;
#         }}
#     </style>
# </head>
# <body>
#     <div class="container">
#         <div id="plot"></div>

#         <div class="inspector">
#             <h3>Selected point</h3>
#             <div id="meta" class="meta">
#                 <span class="hint">Click a point to inspect the image.</span>
#             </div>
#             <div class="image-row">
#                 <div class="panel-image-block">
#                     <div class="panel-label">Selected image</div>
#                     <img id="main-image" class="panel-image" src="" alt="Selected image will appear here"/>
#                 </div>
#                 {pair_html}
#             </div>
#         </div>
#     </div>

#     <script>
#         const fig = {fig_json};
#         Plotly.newPlot('plot', fig.data, fig.layout, {{responsive: true}}).then(function(gd) {{

#             gd.on('plotly_click', function(evt) {{
#                 if (!evt || !evt.points || evt.points.length === 0) return;

#                 const pt = evt.points[0];
#                 const cd = pt.customdata;

#                 // customdata layout:
#                 // [pair_id, type, path, image_b64, pair_type, pair_path, pair_image_b64]
#                 const pairId = cd[0] || "";
#                 const itemType = cd[1] || "";
#                 const itemPath = cd[2] || "";
#                 const itemImage = cd[3] || "";
#                 const pairType = cd[4] || "";
#                 const pairPath = cd[5] || "";
#                 const pairImage = cd[6] || "";

#                 const meta = document.getElementById('meta');
#                 const mainImg = document.getElementById('main-image');

#                 let metaHtml = `
#                     <div><b>Pair ID:</b> ${{pairId}}</div>
#                     <div><b>Type:</b> ${{itemType}}</div>
#                     <div><b>Path:</b> ${{itemPath}}</div>
#                 `;

#                 if (pairType || pairPath) {{
#                     metaHtml += `
#                         <div style="margin-top:8px;"><b>Paired type:</b> ${{pairType}}</div>
#                         <div><b>Paired path:</b> ${{pairPath}}</div>
#                     `;
#                 }}

#                 meta.innerHTML = metaHtml;
#                 mainImg.src = itemImage;

#                 const pairImg = document.getElementById('pair-image');
#                 if (pairImg) {{
#                     pairImg.src = pairImage || "";
#                 }}
#             }});
#         }});
#     </script>
# </body>
# </html>
# """
#     return html


def save_interactive_plot(df, pair_lines, output_path, title, draw_lines=False, include_pair_panel=False):
    """
    Save interactive Plotly scatter plot:
    - metadata on hover
    - image(s) shown in an inspector panel on click
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
                        dft["pair_type"].values,
                        dft["pair_path"].values,
                        dft["pair_image_b64"].values,
                    ],
                    axis=-1
                ),
                hovertemplate=
                "<b>%{customdata[0]}</b><br>"
                "Type: %{customdata[1]}<br>"
                "Path: %{customdata[2]}<br>"
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

    html = _make_html_wrapper(fig, title=title, include_pair_panel=include_pair_panel)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def build_projection_dataframe(records, projection):
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
        "pair_type": [r["pair_type"] for r in records],
        "pair_path": [r["pair_path"] for r in records],
        "pair_image_b64": [r["pair_image_b64"] for r in records],
    })

    df.index = np.arange(len(df))
    return df


def visualize(cfg, checkpoint_path=None, split="all", output_dir=None):
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
        num_workers=4,
        pin_memory=True,
    )

    print(f"Visualization dataset size: {len(dataset)}")

    model = DualEncoder(cfg).to(device)

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

                schema_img = image_to_base64(schema_path, max_size=(180, 180))
                seal_img = image_to_base64(seal_path, max_size=(180, 180))

                schema_idx = len(records)
                records.append({
                    "pair_id": pair_id,
                    "type": "schema",
                    "path": schema_path,
                    "embedding": z_schema[i],
                    "image_b64": schema_img,
                    "pair_type": "seal",
                    "pair_path": seal_path,
                    "pair_image_b64": seal_img,
                })

                seal_idx = len(records)
                records.append({
                    "pair_id": pair_id,
                    "type": "seal",
                    "path": seal_path,
                    "embedding": z_seal[i],
                    "image_b64": seal_img,
                    "pair_type": "schema",
                    "pair_path": schema_path,
                    "pair_image_b64": schema_img,
                })

                pair_lines.append((schema_idx, seal_idx))

    all_emb = np.stack([r["embedding"] for r in records], axis=0)

    if output_dir is None:
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

    df_tsne = build_projection_dataframe(records, proj_tsne)
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
        include_pair_panel=True,
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
        include_pair_panel=True,
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
        include_pair_panel=True,
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

    df_umap = build_projection_dataframe(records, proj_umap)
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
        include_pair_panel=True,
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
        include_pair_panel=True,
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
        include_pair_panel=True,
    )

    print(f"Visualizations saved in: {output_dir}")
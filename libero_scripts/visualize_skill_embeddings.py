#!/usr/bin/env python3
"""Visualize saved GR00T skill embedding banks.

This script compares the learned skill embedding banks saved from the weighted
and top-1 skill routers. It intentionally works from saved pickle files only,
so it does not need to instantiate a GR00T policy or load a full checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any

_MPLCONFIGDIR = Path("/tmp/gr00t_skill_embedding_matplotlib")
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_WEIGHTED_LABELS = ["close", "open", "pick", "place", "turn"]
DEFAULT_TOP1_LABELS = ["open", "pick", "place", "turn"]
DEFAULT_WEIGHTED_ROW_SLICE = "1:5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize weighted-router and top1-router skill embedding distributions."
    )
    parser.add_argument(
        "--weighted",
        type=Path,
        default=Path("weight_router_skill.pkl"),
        help="Path to weighted-router skill embedding pickle.",
    )
    parser.add_argument(
        "--top1",
        type=Path,
        default=Path("top1_router_skill.pkl"),
        help="Path to top1-router skill embedding pickle.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("skill_embedding_viz"),
        help="Directory where plots and summary.json will be written.",
    )
    parser.add_argument(
        "--weighted_labels",
        nargs="+",
        default=DEFAULT_WEIGHTED_LABELS,
        help="Skill labels for rows in the weighted-router embedding bank.",
    )
    parser.add_argument(
        "--top1_labels",
        nargs="+",
        default=DEFAULT_TOP1_LABELS,
        help="Skill labels for rows in the top1-router embedding bank.",
    )
    parser.add_argument(
        "--weighted_row_slice",
        type=str,
        default=DEFAULT_WEIGHTED_ROW_SLICE,
        help=(
            "Optional Python-style row slice applied to weighted embeddings and labels "
            "before visualization. Default '1:5' drops the weighted-only 'close' row."
        ),
    )
    return parser.parse_args()


def parse_row_slice(value: str | None, num_rows: int) -> slice:
    if value is None or value.lower() in {"", "none", "all"}:
        return slice(None)
    parts = value.split(":")
    if len(parts) > 3:
        raise ValueError(f"invalid row slice {value!r}; expected START:STOP[:STEP]")

    parsed: list[int | None] = []
    for part in parts:
        parsed.append(int(part) if part else None)
    while len(parsed) < 3:
        parsed.append(None)

    row_slice = slice(parsed[0], parsed[1], parsed[2])
    indices = range(num_rows)[row_slice]
    if len(indices) == 0:
        raise ValueError(f"row slice {value!r} selects no weighted embedding rows")
    return row_slice


def load_embedding(path: Path) -> tuple[torch.Tensor, str]:
    with path.open("rb") as f:
        obj: Any = pickle.load(f)

    if isinstance(obj, torch.Tensor):
        tensor = obj.detach().cpu()
        source_dtype = str(obj.dtype)
    else:
        tensor = torch.as_tensor(obj)
        source_dtype = str(tensor.dtype)

    tensor = tensor.to(dtype=torch.float32, device="cpu")
    if tensor.ndim != 2:
        raise ValueError(f"{path} must contain a rank-2 embedding bank, got shape {tuple(tensor.shape)}")
    return tensor, source_dtype


def validate_labels(name: str, labels: list[str], embeddings: torch.Tensor) -> None:
    if len(labels) != embeddings.shape[0]:
        raise ValueError(
            f"{name} label count mismatch: got {len(labels)} labels for "
            f"{embeddings.shape[0]} embedding rows. Labels: {labels}"
        )
    if len(set(labels)) != len(labels):
        raise ValueError(f"{name} labels must be unique, got: {labels}")


def cosine_matrix(embeddings: torch.Tensor) -> np.ndarray:
    normalized = torch.nn.functional.normalize(embeddings, dim=-1)
    return (normalized @ normalized.T).numpy()


def pairwise_cosine_stats(cosine: np.ndarray) -> dict[str, float | None]:
    if cosine.shape[0] < 2:
        return {"offdiag_min": None, "offdiag_max": None, "offdiag_mean": None, "offdiag_std": None}
    mask = ~np.eye(cosine.shape[0], dtype=bool)
    values = cosine[mask]
    return {
        "offdiag_min": float(values.min()),
        "offdiag_max": float(values.max()),
        "offdiag_mean": float(values.mean()),
        "offdiag_std": float(values.std()),
    }


def pca_2d(data: np.ndarray) -> tuple[np.ndarray, str]:
    data = np.asarray(data, dtype=np.float32)
    centered = data - data.mean(axis=0, keepdims=True)

    try:
        from sklearn.decomposition import PCA

        reduced = PCA(n_components=2).fit_transform(centered)
        return reduced, "sklearn"
    except Exception:
        pass

    try:
        u, s, _ = torch.pca_lowrank(torch.from_numpy(centered), q=2, center=False)
        reduced = (u[:, :2] * s[:2]).numpy()
        return reduced, "torch.pca_lowrank"
    except Exception:
        pass

    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    reduced = centered @ vt[:2].T
    if reduced.shape[1] == 1:
        reduced = np.concatenate([reduced, np.zeros_like(reduced)], axis=1)
    return reduced[:, :2], "numpy.svd"


def save_cosine_heatmap(cosine: np.ndarray, labels: list[str], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cosine, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cosine[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="cosine similarity")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_pca_scatter(
    weighted: torch.Tensor,
    top1: torch.Tensor,
    weighted_labels: list[str],
    top1_labels: list[str],
    path: Path,
) -> str:
    combined = torch.cat([weighted, top1], dim=0).numpy()
    reduced, method = pca_2d(combined)

    all_labels = weighted_labels + top1_labels
    routers = ["weighted"] * len(weighted_labels) + ["top1"] * len(top1_labels)
    unique_labels = sorted(set(all_labels))
    cmap = plt.get_cmap("tab10")
    color_by_label = {label: cmap(i % 10) for i, label in enumerate(unique_labels)}
    marker_by_router = {"weighted": "o", "top1": "^"}

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, (label, router) in enumerate(zip(all_labels, routers)):
        ax.scatter(
            reduced[idx, 0],
            reduced[idx, 1],
            color=color_by_label[label],
            marker=marker_by_router[router],
            s=95,
            edgecolor="black",
            linewidth=0.6,
        )
        ax.annotate(
            f"{label}\n{router}",
            (reduced[idx, 0], reduced[idx, 1]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    for label in sorted(set(weighted_labels).intersection(top1_labels)):
        w_idx = weighted_labels.index(label)
        t_idx = len(weighted_labels) + top1_labels.index(label)
        ax.plot(
            [reduced[w_idx, 0], reduced[t_idx, 0]],
            [reduced[w_idx, 1], reduced[t_idx, 1]],
            color=color_by_label[label],
            alpha=0.35,
            linewidth=1.2,
        )

    for label, color in color_by_label.items():
        ax.scatter([], [], color=color, marker="o", label=label)
    ax.scatter([], [], color="white", edgecolor="black", marker="o", label="weighted")
    ax.scatter([], [], color="white", edgecolor="black", marker="^", label="top1")
    ax.set_title(f"Skill Embeddings PCA ({method})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return method


def save_single_pca_scatter(
    embeddings: torch.Tensor,
    labels: list[str],
    title: str,
    path: Path,
) -> str:
    reduced, method = pca_2d(embeddings.numpy())
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(7, 5))
    for idx, label in enumerate(labels):
        ax.scatter(
            reduced[idx, 0],
            reduced[idx, 1],
            color=cmap(idx % 10),
            s=100,
            edgecolor="black",
            linewidth=0.6,
        )
        ax.annotate(
            label,
            (reduced[idx, 0], reduced[idx, 1]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=9,
        )

    ax.set_title(f"{title} PCA ({method})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return method


def save_norms_plot(
    weighted: torch.Tensor,
    top1: torch.Tensor,
    weighted_labels: list[str],
    top1_labels: list[str],
    path: Path,
) -> None:
    weighted_norms = weighted.norm(dim=-1).numpy()
    top1_norms = top1.norm(dim=-1).numpy()
    x_weighted = np.arange(len(weighted_labels))
    x_top1 = np.arange(len(top1_labels)) + len(weighted_labels) + 1

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x_weighted, weighted_norms, color="#4c78a8", label="weighted")
    ax.bar(x_top1, top1_norms, color="#f58518", label="top1")
    ax.set_xticks(
        np.concatenate([x_weighted, x_top1]),
        [f"{label}\nweighted" for label in weighted_labels]
        + [f"{label}\ntop1" for label in top1_labels],
        rotation=0,
    )
    ax.set_ylabel("L2 norm")
    ax.set_title("Per-Skill Embedding Norms")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_single_norms_plot(
    embeddings: torch.Tensor,
    labels: list[str],
    title: str,
    path: Path,
) -> None:
    norms = embeddings.norm(dim=-1).numpy()
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x, norms, color="#4c78a8")
    ax.set_xticks(x, labels)
    ax.set_ylabel("L2 norm")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_value_distribution(weighted: torch.Tensor, top1: torch.Tensor, path: Path) -> None:
    weighted_values = weighted.flatten().numpy()
    top1_values = top1.flatten().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(weighted_values, bins=40, alpha=0.7, color="#4c78a8", density=True, label="weighted")
    axes[0].hist(top1_values, bins=40, alpha=0.7, color="#f58518", density=True, label="top1")
    axes[0].set_title("Raw Value Histogram")
    axes[0].set_xlabel("embedding value")
    axes[0].set_ylabel("density")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].boxplot([weighted_values, top1_values], showfliers=False)
    axes[1].set_xticks([1, 2], ["weighted", "top1"])
    axes[1].set_title("Raw Value Boxplot")
    axes[1].set_ylabel("embedding value")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_single_value_distribution(
    embeddings: torch.Tensor,
    labels: list[str],
    title: str,
    path: Path,
) -> None:
    values = embeddings.flatten().numpy()
    rows = [embeddings[idx].numpy() for idx in range(embeddings.shape[0])]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(values, bins=40, alpha=0.8, color="#4c78a8", density=True)
    axes[0].set_title("All Values")
    axes[0].set_xlabel("embedding value")
    axes[0].set_ylabel("density")
    axes[0].grid(alpha=0.25)

    axes[1].boxplot(rows, showfliers=False)
    axes[1].set_xticks(np.arange(1, len(labels) + 1), labels, rotation=30, ha="right")
    axes[1].set_title("Per-Skill Values")
    axes[1].set_ylabel("embedding value")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def shared_skill_distances(
    weighted: torch.Tensor,
    top1: torch.Tensor,
    weighted_labels: list[str],
    top1_labels: list[str],
) -> list[dict[str, float | str]]:
    records: list[dict[str, float | str]] = []
    for label in sorted(set(weighted_labels).intersection(top1_labels)):
        w = weighted[weighted_labels.index(label)]
        t = top1[top1_labels.index(label)]
        cos = torch.nn.functional.cosine_similarity(w.unsqueeze(0), t.unsqueeze(0)).item()
        records.append(
            {
                "skill": label,
                "l2_distance": float(torch.linalg.vector_norm(w - t).item()),
                "cosine_similarity": float(cos),
                "cosine_distance": float(1.0 - cos),
            }
        )
    return records


def save_shared_distance_plot(records: list[dict[str, float | str]], path: Path) -> None:
    if not records:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No shared skill labels", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return

    labels = [str(row["skill"]) for row in records]
    l2 = np.array([float(row["l2_distance"]) for row in records])
    cos_dist = np.array([float(row["cosine_distance"]) for row in records])
    x = np.arange(len(labels))
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(x - width / 2, l2, width, color="#4c78a8", label="L2 distance")
    ax1.set_ylabel("L2 distance")
    ax1.set_xticks(x, labels)
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, cos_dist, width, color="#e45756", label="cosine distance")
    ax2.set_ylabel("cosine distance")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")
    ax1.set_title("Shared Skill Distance Between Routers")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def embedding_summary(
    name: str,
    tensor: torch.Tensor,
    source_dtype: str,
    labels: list[str],
    cosine: np.ndarray,
) -> dict[str, Any]:
    values = tensor.numpy()
    return {
        "name": name,
        "shape": list(tensor.shape),
        "source_dtype": source_dtype,
        "analysis_dtype": str(tensor.dtype),
        "labels": labels,
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "row_norms": {label: float(tensor[idx].norm().item()) for idx, label in enumerate(labels)},
        "pairwise_cosine": pairwise_cosine_stats(cosine),
    }


def main() -> None:
    args = parse_args()

    weighted, weighted_dtype = load_embedding(args.weighted)
    top1, top1_dtype = load_embedding(args.top1)
    weighted_labels = list(args.weighted_labels)
    top1_labels = list(args.top1_labels)

    validate_labels("weighted", weighted_labels, weighted)
    weighted_row_slice = parse_row_slice(args.weighted_row_slice, weighted.shape[0])
    weighted = weighted[weighted_row_slice]
    weighted_labels = weighted_labels[weighted_row_slice]

    validate_labels("weighted", weighted_labels, weighted)
    validate_labels("top1", top1_labels, top1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    weighted_cos = cosine_matrix(weighted)
    top1_cos = cosine_matrix(top1)
    shared_records = shared_skill_distances(weighted, top1, weighted_labels, top1_labels)

    pca_method = save_pca_scatter(
        weighted,
        top1,
        weighted_labels,
        top1_labels,
        args.out_dir / "pca_scatter.png",
    )
    weighted_pca_method = save_single_pca_scatter(
        weighted,
        weighted_labels,
        "Weighted Router Skill Embeddings",
        args.out_dir / "pca_weighted.png",
    )
    top1_pca_method = save_single_pca_scatter(
        top1,
        top1_labels,
        "Top1 Router Skill Embeddings",
        args.out_dir / "pca_top1.png",
    )
    save_cosine_heatmap(
        weighted_cos,
        weighted_labels,
        "Weighted Router Skill Cosine Similarity",
        args.out_dir / "cosine_weighted.png",
    )
    save_cosine_heatmap(
        top1_cos,
        top1_labels,
        "Top1 Router Skill Cosine Similarity",
        args.out_dir / "cosine_top1.png",
    )
    save_norms_plot(weighted, top1, weighted_labels, top1_labels, args.out_dir / "norms.png")
    save_single_norms_plot(
        weighted,
        weighted_labels,
        "Weighted Router Per-Skill Norms",
        args.out_dir / "norms_weighted.png",
    )
    save_single_norms_plot(
        top1,
        top1_labels,
        "Top1 Router Per-Skill Norms",
        args.out_dir / "norms_top1.png",
    )
    save_value_distribution(weighted, top1, args.out_dir / "value_distribution.png")
    save_single_value_distribution(
        weighted,
        weighted_labels,
        "Weighted Router Skill Embedding Value Distribution",
        args.out_dir / "value_distribution_weighted.png",
    )
    save_single_value_distribution(
        top1,
        top1_labels,
        "Top1 Router Skill Embedding Value Distribution",
        args.out_dir / "value_distribution_top1.png",
    )
    save_shared_distance_plot(shared_records, args.out_dir / "shared_skill_distance.png")

    summary = {
        "weighted_path": str(args.weighted),
        "top1_path": str(args.top1),
        "out_dir": str(args.out_dir),
        "weighted_row_slice": args.weighted_row_slice,
        "pca_method": pca_method,
        "weighted_pca_method": weighted_pca_method,
        "top1_pca_method": top1_pca_method,
        "weighted": embedding_summary("weighted", weighted, weighted_dtype, weighted_labels, weighted_cos),
        "top1": embedding_summary("top1", top1, top1_dtype, top1_labels, top1_cos),
        "shared_skill_distances": shared_records,
        "weighted_only_labels": sorted(set(weighted_labels) - set(top1_labels)),
        "top1_only_labels": sorted(set(top1_labels) - set(weighted_labels)),
    }
    summary_path = args.out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"weighted shape: {tuple(weighted.shape)} from {args.weighted} "
        f"({weighted_dtype}), row slice={args.weighted_row_slice}"
    )
    print(f"top1 shape: {tuple(top1.shape)} from {args.top1} ({top1_dtype})")
    print(f"shared labels: {[row['skill'] for row in shared_records]}")
    print(f"wrote visualizations and summary to: {args.out_dir}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None

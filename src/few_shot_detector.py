"""Simplified few-shot graph anomaly detection.

Only a small number of known normal nodes are used as context. Every other
node is scored by its distance from the context profile.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baseline_detector import calculate_manual_roc_auc
from load_data import load_demo_dataset


def calculate_few_shot_scores(nodes: pd.DataFrame, context_size: int = 5) -> tuple[pd.DataFrame, list[int]]:
    """Score nodes using only a small set of known normal context nodes."""

    feature_columns = [column for column in nodes.columns if column.startswith("feature_")]
    normal_nodes = nodes[nodes["label"] == 0].sort_values("node_id")
    context = normal_nodes.head(context_size)
    context_values = context[feature_columns].to_numpy(dtype=float)
    context_mean = context_values.mean(axis=0)
    context_std = context_values.std(axis=0)
    context_std[context_std == 0] = 1.0

    values = nodes[feature_columns].to_numpy(dtype=float)
    distances = ((values - context_mean) / context_std) ** 2
    scored_nodes = nodes.copy()
    scored_nodes["few_shot_score"] = np.sqrt(distances.mean(axis=1))
    scored_nodes = scored_nodes.sort_values("few_shot_score", ascending=False).reset_index(drop=True)
    return scored_nodes, context["node_id"].tolist()


def save_plot(scored_nodes: pd.DataFrame, context_ids: list[int], output_path: Path) -> None:
    """Save a plot of few-shot anomaly scores."""

    ordered = scored_nodes.sort_values("few_shot_score", ascending=True)
    colours = ordered["label"].map({0: "#4c78a8", 1: "#e45756"})
    plt.figure(figsize=(11, 6))
    plt.bar(range(len(ordered)), ordered["few_shot_score"], color=colours)
    plt.xlabel("Nodes ordered by few-shot anomaly score")
    plt.ylabel("Few-shot anomaly score")
    plt.title(f"Few-Shot Detection Using Normal Context Nodes: {context_ids}")
    plt.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#4c78a8", label="Normal"),
            plt.Rectangle((0, 0), 1, 1, color="#e45756", label="Anomalous"),
        ]
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    _, nodes = load_demo_dataset(project_root / "data" / "raw")
    scored_nodes, context_ids = calculate_few_shot_scores(nodes, context_size=5)
    auc = calculate_manual_roc_auc(
        scored_nodes["label"].to_numpy(), scored_nodes["few_shot_score"].to_numpy()
    )

    result_dir = project_root / "results" / "few_shot"
    result_dir.mkdir(parents=True, exist_ok=True)
    scored_nodes.to_csv(result_dir / "few_shot_scores.csv", index=False)
    save_plot(scored_nodes, context_ids, result_dir / "few_shot_scores.png")
    (result_dir / "metrics.txt").write_text(
        f"Context node IDs: {context_ids}\n"
        f"Number of context nodes: {len(context_ids)}\n"
        f"Few-shot ROC-AUC: {auc:.4f}\n"
        f"Top 10 predicted node IDs: {scored_nodes.head(10)['node_id'].tolist()}\n",
        encoding="utf-8",
    )

    print(f"Few-shot context nodes: {context_ids}")
    print(f"Few-shot ROC-AUC: {auc:.4f}")
    print(f"Saved few-shot results to: {result_dir}")


if __name__ == "__main__":
    main()

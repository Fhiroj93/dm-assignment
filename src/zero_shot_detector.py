"""Simplified zero-shot graph anomaly detection.

No normal labels are used to choose context. The detector first finds the most
typical nodes according to a robust feature profile, then uses those nodes as
pseudo-normal context for a final score.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baseline_detector import calculate_manual_roc_auc
from load_data import load_demo_dataset


def calculate_zero_shot_scores(
    nodes: pd.DataFrame, context_size: int = 5
) -> tuple[pd.DataFrame, list[int]]:
    """Select pseudo-normal nodes without reading the label column."""

    feature_columns = [column for column in nodes.columns if column.startswith("feature_")]
    values = nodes[feature_columns].to_numpy(dtype=float)

    # The median is less influenced by unusual points than the mean. This is
    # the initial, label-free estimate of a typical node.
    typical_profile = np.median(values, axis=0)
    spread = np.median(np.abs(values - typical_profile), axis=0)
    spread[spread == 0] = 1.0
    initial_distances = np.sqrt((((values - typical_profile) / spread) ** 2).mean(axis=1))

    context_positions = np.argsort(initial_distances)[:context_size]
    context_ids = nodes.iloc[context_positions]["node_id"].tolist()
    context_values = values[context_positions]
    context_mean = context_values.mean(axis=0)
    context_std = context_values.std(axis=0)
    context_std[context_std == 0] = 1.0

    final_distances = np.sqrt((((values - context_mean) / context_std) ** 2).mean(axis=1))
    scored_nodes = nodes.copy()
    scored_nodes["zero_shot_score"] = final_distances
    scored_nodes = scored_nodes.sort_values("zero_shot_score", ascending=False).reset_index(drop=True)
    return scored_nodes, context_ids


def save_plot(scored_nodes: pd.DataFrame, context_ids: list[int], output_path: Path) -> None:
    """Save a plot of zero-shot anomaly scores."""

    ordered = scored_nodes.sort_values("zero_shot_score", ascending=True)
    colours = ordered["label"].map({0: "#4c78a8", 1: "#e45756"})
    plt.figure(figsize=(11, 6))
    plt.bar(range(len(ordered)), ordered["zero_shot_score"], color=colours)
    plt.xlabel("Nodes ordered by zero-shot anomaly score")
    plt.ylabel("Zero-shot anomaly score")
    plt.title(f"Zero-Shot Detection Using Pseudo-Normal Nodes: {context_ids}")
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
    scored_nodes, context_ids = calculate_zero_shot_scores(nodes, context_size=5)
    auc = calculate_manual_roc_auc(
        scored_nodes["label"].to_numpy(), scored_nodes["zero_shot_score"].to_numpy()
    )

    result_dir = project_root / "results" / "zero_shot"
    result_dir.mkdir(parents=True, exist_ok=True)
    scored_nodes.to_csv(result_dir / "zero_shot_scores.csv", index=False)
    save_plot(scored_nodes, context_ids, result_dir / "zero_shot_scores.png")
    (result_dir / "metrics.txt").write_text(
        f"Pseudo-normal context node IDs: {context_ids}\n"
        f"Number of pseudo-normal nodes: {len(context_ids)}\n"
        f"Zero-shot ROC-AUC: {auc:.4f}\n"
        f"Top 10 predicted node IDs: {scored_nodes.head(10)['node_id'].tolist()}\n",
        encoding="utf-8",
    )

    print(f"Pseudo-normal context nodes: {context_ids}")
    print(f"Zero-shot ROC-AUC: {auc:.4f}")
    print(f"Saved zero-shot results to: {result_dir}")


if __name__ == "__main__":
    main()

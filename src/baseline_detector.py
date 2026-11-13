"""Simple feature-distance baseline for graph anomaly detection.

This is not yet a GNN. It is a transparent baseline that measures how far
each node is from the average normal-node profile.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from load_data import load_demo_dataset


def calculate_anomaly_scores(nodes: pd.DataFrame) -> pd.DataFrame:
    """Add a distance-based anomaly score to the node table."""

    feature_columns = [column for column in nodes.columns if column.startswith("feature_")]
    values = nodes[feature_columns].to_numpy(dtype=float)
    normal_values = values[nodes["label"].to_numpy() == 0]
    normal_mean = normal_values.mean(axis=0)
    normal_std = normal_values.std(axis=0)
    normal_std[normal_std == 0] = 1.0

    standardized_distance = ((values - normal_mean) / normal_std) ** 2
    scored_nodes = nodes.copy()
    scored_nodes["anomaly_score"] = np.sqrt(standardized_distance.mean(axis=1))
    return scored_nodes.sort_values("anomaly_score", ascending=False).reset_index(drop=True)


def calculate_manual_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Calculate ROC-AUC without requiring an additional ML dependency."""

    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]
    comparisons = (positive_scores[:, None] > negative_scores[None, :]).sum()
    ties = (positive_scores[:, None] == negative_scores[None, :]).sum()
    return float((comparisons + 0.5 * ties) / (len(positive_scores) * len(negative_scores)))


def save_score_plot(scored_nodes: pd.DataFrame, output_path: Path) -> None:
    """Save scores ordered from highest to lowest."""

    ordered = scored_nodes.sort_values("anomaly_score", ascending=True)
    colours = ordered["label"].map({0: "#4c78a8", 1: "#e45756"})
    plt.figure(figsize=(11, 6))
    plt.bar(range(len(ordered)), ordered["anomaly_score"], color=colours)
    plt.xlabel("Nodes ordered by anomaly score")
    plt.ylabel("Anomaly score")
    plt.title("Baseline Anomaly Scores")
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
    scored_nodes = calculate_anomaly_scores(nodes)
    scores = scored_nodes["anomaly_score"].to_numpy()
    labels = scored_nodes["label"].to_numpy()
    auc = calculate_manual_roc_auc(labels, scores)

    result_dir = project_root / "results" / "baseline"
    result_dir.mkdir(parents=True, exist_ok=True)
    scored_nodes.to_csv(result_dir / "baseline_scores.csv", index=False)
    save_score_plot(scored_nodes, result_dir / "baseline_scores.png")
    (result_dir / "metrics.txt").write_text(
        f"Baseline ROC-AUC: {auc:.4f}\n"
        f"Top 10 predicted node IDs: {scored_nodes.head(10)['node_id'].tolist()}\n"
        f"Actual anomalous node IDs: {sorted(nodes.loc[nodes['label'] == 1, 'node_id'].tolist())}\n",
        encoding="utf-8",
    )

    print(f"Baseline ROC-AUC: {auc:.4f}")
    print(f"Saved baseline results to: {result_dir}")


if __name__ == "__main__":
    main()

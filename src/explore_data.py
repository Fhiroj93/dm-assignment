"""Explore the demo graph and save simple visualizations."""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from load_data import load_demo_dataset


def save_graph_visualization(
    graph: nx.Graph, nodes: pd.DataFrame, output_path: Path
) -> None:
    """Save a graph image with normal and anomalous nodes coloured separately."""

    labels = nodes.set_index("node_id")["label"].to_dict()
    normal = [node for node in graph.nodes if labels.get(node, 0) == 0]
    anomalies = [node for node in graph.nodes if labels.get(node, 0) == 1]

    layout = nx.spring_layout(graph, seed=42)
    plt.figure(figsize=(12, 8))
    nx.draw_networkx_edges(graph, layout, alpha=0.18, width=0.7, edge_color="gray")
    nx.draw_networkx_nodes(
        graph,
        layout,
        nodelist=normal,
        node_color="#4c78a8",
        node_size=65,
        label="Normal node",
    )
    nx.draw_networkx_nodes(
        graph,
        layout,
        nodelist=anomalies,
        node_color="#e45756",
        node_size=150,
        node_shape="X",
        label="Anomalous node",
    )
    plt.title("Demo Graph: Normal and Anomalous Nodes")
    plt.legend(scatterpoints=1)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_feature_visualization(nodes: pd.DataFrame, output_path: Path) -> None:
    """Save a feature summary comparing normal and anomalous nodes."""

    feature_columns = [column for column in nodes.columns if column.startswith("feature_")]
    normal_means = nodes.loc[nodes["label"] == 0, feature_columns].mean()
    anomaly_means = nodes.loc[nodes["label"] == 1, feature_columns].mean()

    positions = range(len(feature_columns))
    width = 0.36
    plt.figure(figsize=(10, 6))
    plt.bar([position - width / 2 for position in positions], normal_means, width, label="Normal")
    plt.bar([position + width / 2 for position in positions], anomaly_means, width, label="Anomalous")
    plt.xticks(list(positions), feature_columns)
    plt.ylabel("Average feature value")
    plt.title("Average Features by Node Class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_summary(graph: nx.Graph, nodes: pd.DataFrame, output_path: Path) -> None:
    """Write basic graph statistics for the project report."""

    summary = [
        "Graph Exploration Summary",
        "=========================",
        f"Number of nodes: {graph.number_of_nodes()}",
        f"Number of edges: {graph.number_of_edges()}",
        f"Average degree: {sum(dict(graph.degree).values()) / graph.number_of_nodes():.2f}",
        f"Normal nodes: {int((nodes['label'] == 0).sum())}",
        f"Anomalous nodes: {int((nodes['label'] == 1).sum())}",
        f"Largest degree: {nodes['degree'].max()}",
    ]
    output_path.write_text("\n".join(summary), encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    graph, nodes = load_demo_dataset(project_root / "data" / "raw")
    result_dir = project_root / "results" / "exploration"
    result_dir.mkdir(parents=True, exist_ok=True)

    save_graph_visualization(graph, nodes, result_dir / "graph_overview.png")
    save_feature_visualization(nodes, result_dir / "feature_comparison.png")
    write_summary(graph, nodes, result_dir / "summary.txt")
    print(f"Saved exploration results to: {result_dir}")


if __name__ == "__main__":
    main()

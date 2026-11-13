"""Create a small graph dataset for the classroom demonstration.

The generated graph represents a network. Most nodes belong to normal
communities; a small number are deliberately made anomalous by giving them
unusual features and unusual connections.
"""

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def create_demo_graph(
    normal_nodes: int = 80,
    anomalous_nodes: int = 10,
    seed: int = 42,
) -> tuple[nx.Graph, pd.DataFrame]:
    """Return a graph and a node table containing features and labels."""

    rng = np.random.default_rng(seed)
    total_nodes = normal_nodes + anomalous_nodes

    # Two dense normal communities make the graph more realistic than a
    # completely random network.
    community_sizes = [normal_nodes // 2, normal_nodes - normal_nodes // 2]
    graph = nx.stochastic_block_model(
        community_sizes,
        [[0.16, 0.025], [0.025, 0.16]],
        seed=seed,
    )

    # Add anomalous nodes with sparse, unusual cross-community connections.
    graph.add_nodes_from(range(normal_nodes, total_nodes))
    for node in range(normal_nodes, total_nodes):
        neighbours = rng.choice(normal_nodes, size=5, replace=False)
        graph.add_edges_from((node, int(neighbour)) for neighbour in neighbours)

    # Normal nodes have compact feature distributions. Anomalies are shifted
    # away from that distribution and have a larger activity value.
    normal_features = rng.normal(loc=0.0, scale=1.0, size=(normal_nodes, 4))
    anomaly_features = rng.normal(loc=4.0, scale=1.0, size=(anomalous_nodes, 4))
    features = np.vstack([normal_features, anomaly_features])

    records = []
    for node in range(total_nodes):
        row = {f"feature_{index + 1}": features[node, index] for index in range(4)}
        row["node_id"] = node
        row["label"] = int(node >= normal_nodes)
        row["degree"] = graph.degree[node]
        records.append(row)

    return graph, pd.DataFrame(records).sort_values("node_id").reset_index(drop=True)


def save_demo_dataset(output_dir: str | Path) -> tuple[Path, Path]:
    """Generate and save the graph edge list and node table."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    graph, node_table = create_demo_graph()

    edges_path = output_path / "demo_edges.csv"
    nodes_path = output_path / "demo_nodes.csv"
    nx.write_edgelist(graph, edges_path, data=False, delimiter=",")
    node_table.to_csv(nodes_path, index=False)
    return edges_path, nodes_path


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    edges, nodes = save_demo_dataset(project_root / "data" / "raw")
    print(f"Saved edges to: {edges}")
    print(f"Saved nodes to: {nodes}")

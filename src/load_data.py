"""Load the demonstration graph and its node table."""

from pathlib import Path

import networkx as nx
import pandas as pd


def load_demo_dataset(data_dir: str | Path) -> tuple[nx.Graph, pd.DataFrame]:
    """Load the saved edge list and node table."""

    data_path = Path(data_dir)
    edges_path = data_path / "demo_edges.csv"
    nodes_path = data_path / "demo_nodes.csv"

    if not edges_path.exists() or not nodes_path.exists():
        raise FileNotFoundError(
            "Demo data is missing. Run `python src/data_generator.py` first."
        )

    graph = nx.read_edgelist(edges_path, delimiter=",", nodetype=int)
    node_table = pd.read_csv(nodes_path)
    graph.add_nodes_from(node_table["node_id"].tolist())
    return graph, node_table


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    graph, nodes = load_demo_dataset(project_root / "data" / "raw")
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")
    print(f"Anomalies: {int(nodes['label'].sum())}")

# Lecturer Presentation Notes

## Project title

Graph Anomaly Detection: A Simplified Few-Shot and Zero-Shot Demonstration

## Problem statement

Graphs are used to represent networks such as computer systems, social
networks, financial transactions, and communication systems. Some nodes may
behave unusually. The goal is to identify these anomalous nodes.

Traditional anomaly detection often trains a separate model for every dataset.
This project demonstrates a more general idea: use normal examples, or
automatically selected typical examples, as context for detecting anomalies.

## Implementation summary

1. A synthetic graph with 90 nodes is generated.
2. It contains 80 normal nodes and 10 deliberately anomalous nodes.
3. Each node has four numerical features.
4. The graph and feature data are visualized.
5. A baseline distance-based detector is implemented.
6. A few-shot detector uses five known normal nodes.
7. A zero-shot detector automatically selects five pseudo-normal nodes.
8. The methods are evaluated and displayed in Streamlit.

## Explanation of the three methods

### Baseline

The detector calculates the distance between each node and the average profile
of normal nodes. A larger distance produces a higher anomaly score.

### Few-shot

Only a small number of normal nodes are provided. Their average feature profile
is used as context, and all other nodes are compared with it.

### Zero-shot

No normal examples are provided. The detector first finds the most typical
nodes using a median feature profile. Those nodes become pseudo-normal context
for the final anomaly scores.

## Results

The synthetic dataset produces perfect results for all three methods because
the anomalous features were intentionally separated from the normal features.
This confirms that the implementation works, but it should not be interpreted
as real-world performance.

## Future improvements

- Use a public real-world graph dataset.
- Add a Graph Convolutional Network or GraphSAGE model.
- Implement a learned feature-alignment layer.
- Add cross-attention between target nodes and context nodes.
- Test on multiple datasets and compare computational cost.

## Short conclusion

The project demonstrates how graph structure and node features can be used to
rank suspicious nodes. It also shows the difference between using labelled
normal examples and selecting context automatically in a zero-shot setting.

# Graph Anomaly Detection - DM Assignment

This project is an educational implementation inspired by the paper **From Few-Shot to Zero-Shot: Towards Generalist Graph Anomaly Detection**.

## Planned demonstration

1. Load a small graph dataset.
2. Visualize normal and anomalous nodes.
3. Train a simple graph anomaly detector.
4. Add simplified few-shot detection using normal context nodes.
5. Add simplified zero-shot detection using automatically selected pseudo-normal nodes.
6. Compare results and display them in a small dashboard.

The implementation is intentionally simplified for a clear academic demonstration.

## What the project demonstrates

The project identifies nodes that behave differently from the majority of
nodes in a graph. It compares three approaches:

- Baseline: compares each node with the average normal-node profile.
- Few-shot: uses five known normal nodes as context.
- Zero-shot: automatically selects five typical nodes as pseudo-normal context.

## Run the complete project

Run these commands from this folder in the listed order:

```powershell
python src/data_generator.py
python src/explore_data.py
python src/baseline_detector.py
python src/few_shot_detector.py
python src/zero_shot_detector.py
python src/evaluate_models.py
python -m streamlit run app.py
```

The browser dashboard will open locally. If it does not open automatically,
visit the local URL printed in the terminal.

## How to explain the result

Each node receives an anomaly score. A higher score means the node is more
different from the normal or pseudo-normal feature profile. The dashboard
sorts nodes by this score and displays the most suspicious nodes first.

## Important limitation

The included dataset is synthetic and intentionally easy to understand. The
project demonstrates the implementation idea inspired by the paper; it is not
a full reproduction of the paper's ARC and ARCzero experiments across 17
real-world graph datasets.

## Project status

- Part 1: project setup - started
- Part 2: dataset preparation - complete (synthetic demonstration graph)
- Part 3: graph exploration - complete
- Part 4: baseline detector - complete
- Part 5: few-shot detector - complete
- Part 6: zero-shot detector - complete
- Part 7: evaluation and comparison - complete
- Part 8: Streamlit dashboard - complete

## Running later

The exact installation and run commands will be added as the project is built and tested.

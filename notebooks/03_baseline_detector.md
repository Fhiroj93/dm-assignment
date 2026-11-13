# Part 4 - Baseline anomaly detector

Run this command from the `dm-assignment` folder:

```powershell
python src/baseline_detector.py
```

The baseline calculates how far each node's features are from the average
normal-node features. A larger distance means a higher anomaly score.

Results are saved in `results/baseline/`:

- `baseline_scores.csv`: every node and its score.
- `baseline_scores.png`: score visualization.
- `metrics.txt`: ROC-AUC and top predictions.

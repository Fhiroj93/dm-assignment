# Part 7 - Evaluation and comparison

Run this command from the `dm-assignment` folder:

```powershell
python src/evaluate_models.py
```

The comparison evaluates the top 10 predictions from each method using:

- Precision: how many of the top predictions were actually anomalous.
- Recall: how many actual anomalies were found.
- F1-score: a combined precision and recall score.

Results are saved in `results/comparison/`.

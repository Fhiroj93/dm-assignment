# Part 6 - Simplified zero-shot detection

Run this command from the `dm-assignment` folder:

```powershell
python src/zero_shot_detector.py
```

This version receives no known normal-node examples. It estimates the typical
feature profile using the median, selects the most typical nodes as
pseudo-normal context, and then calculates final anomaly scores.

Results are saved in `results/zero_shot/`.

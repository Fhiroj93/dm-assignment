# Part 5 - Simplified few-shot detection

Run this command from the `dm-assignment` folder:

```powershell
python src/few_shot_detector.py
```

This version uses only five known normal nodes as context. It calculates the
average feature profile of those nodes and scores all nodes by their distance
from that profile.

Results are saved in `results/few_shot/`.

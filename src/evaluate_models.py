"""Compare the baseline, few-shot, and zero-shot detectors."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def top_k_metrics(labels: list[int], scores: list[float], k: int) -> tuple[float, float, float]:
    """Return precision, recall, and F1 for the top-k scored nodes."""

    ranked = sorted(zip(scores, labels), reverse=True)
    predicted = [label for _, label in ranked[:k]]
    true_positives = sum(predicted)
    actual_positives = sum(labels)
    precision = true_positives / k if k else 0.0
    recall = true_positives / actual_positives if actual_positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    result_dir = project_root / "results"
    model_files = {
        "Baseline": result_dir / "baseline" / "baseline_scores.csv",
        "Few-shot": result_dir / "few_shot" / "few_shot_scores.csv",
        "Zero-shot": result_dir / "zero_shot" / "zero_shot_scores.csv",
    }
    score_columns = {
        "Baseline": "anomaly_score",
        "Few-shot": "few_shot_score",
        "Zero-shot": "zero_shot_score",
    }

    rows = []
    for model_name, file_path in model_files.items():
        table = pd.read_csv(file_path)
        precision, recall, f1 = top_k_metrics(
            table["label"].tolist(), table[score_columns[model_name]].tolist(), k=10
        )
        rows.append(
            {
                "model": model_name,
                "top_k": 10,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }
        )

    comparison = pd.DataFrame(rows)
    output_dir = result_dir / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)

    chart_data = comparison.set_index("model")[["precision", "recall", "f1_score"]]
    chart_data.plot(kind="bar", figsize=(10, 6), ylim=(0, 1.05), color=["#4c78a8", "#f2cf5b", "#e45756"])
    plt.title("Model Comparison at Top 10 Predictions")
    plt.ylabel("Score")
    plt.xlabel("Method")
    plt.xticks(rotation=0)
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=180)
    plt.close()

    lines = ["Model Comparison", "=================", ""]
    for row in rows:
        lines.append(
            f"{row['model']}: Precision={row['precision']:.2f}, "
            f"Recall={row['recall']:.2f}, F1={row['f1_score']:.2f}"
        )
    (output_dir / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved model comparison to: {output_dir}")


if __name__ == "__main__":
    main()

import os
import pandas as pd

MODEL_DIR = os.path.join("app", "models", "saved")


def read_report(file_path):
    """Lit un rapport .txt et retourne les métriques sous forme de dict"""
    metrics = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("Accuracy"):
                    metrics["Accuracy"] = float(line.split(":")[1].strip())
                elif line.startswith("Precision"):
                    metrics["Precision"] = float(line.split(":")[1].strip())
                elif line.startswith("Recall"):
                    metrics["Recall"] = float(line.split(":")[1].strip())
                elif line.startswith("F1-score"):
                    metrics["F1-score"] = float(line.split(":")[1].strip())
    except FileNotFoundError:
        return None
    return metrics


def main():
    model_names = ["random_forest", "naive_bayes", "logistic_regression", "lstm"]
    results = []

    for model in model_names:
        report_path = os.path.join(MODEL_DIR, f"{model}_report.txt")
        metrics = read_report(report_path)
        if metrics:
            metrics["Model"] = model
            results.append(metrics)
        else:
            print(f"⚠️ Rapport introuvable pour {model}")

    if results:
        df = pd.DataFrame(results)


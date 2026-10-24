
from pathlib import Path
import csv

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from preprocess import prepare_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    X_train, X_test, y_train, y_test = prepare_data()

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "Linear SVM": LinearSVC(
            random_state=42,
            max_iter=5000,
        ),
    }

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1": f1_score(
                y_test, predictions, zero_division=0
            ),
        }

        results.append(metrics)

        print(f"\n{name} RESULTS")
        for key, value in metrics.items():
            if key != "Model":
                print(f"{key}: {value:.4f}")

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, predictions))

    output = RESULTS_DIR / "baseline_metrics.csv"

    with output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["Model", "Accuracy", "Precision", "Recall", "F1"],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()

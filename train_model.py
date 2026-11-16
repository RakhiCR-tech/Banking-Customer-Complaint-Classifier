"""Train and evaluate the banking complaint classifier."""
import csv
import json
import pickle
import random
from pathlib import Path
from naive_bayes import NaiveBayesClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset" / "complaints.csv"
MODEL_PATH = ROOT / "model.pkl"
METRICS_PATH = ROOT / "metrics.json"

def load_dataset():
    with DATA_PATH.open("r", encoding="utf-8", newline="") as file:
        records = list(csv.DictReader(file))
    if len(records) < 10:
        raise ValueError("Dataset is too small. Add at least 10 labelled examples.")
    return records

def main():
    records = load_dataset()
    random.Random(42).shuffle(records)
    split_at = max(1, int(len(records) * 0.8))
    train_records, test_records = records[:split_at], records[split_at:]

    model = NaiveBayesClassifier(alpha=1.0)
    model.fit([r["complaint"] for r in train_records],
              [r["category"] for r in train_records])

    correct = 0
    for record in test_records:
        prediction, _, _ = model.predict_one(record["complaint"])
        correct += prediction == record["category"]
    accuracy = correct / len(test_records) * 100

    with MODEL_PATH.open("wb") as file:
        pickle.dump(model, file)
    metrics = {
        "dataset_records": len(records),
        "training_records": len(train_records),
        "testing_records": len(test_records),
        "accuracy": round(accuracy, 2),
        "categories": sorted(model.class_counts.keys())
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=4), encoding="utf-8")
    print("Training completed.")
    print(f"Dataset records: {len(records)}")
    print(f"Test accuracy: {accuracy:.2f}%")
    print(f"Saved model: {MODEL_PATH.name}")
    print("Note: This is a small demonstration dataset; evaluation results are illustrative.")

if __name__ == "__main__":
    main()

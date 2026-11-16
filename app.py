"""Flask web app for classifying banking customer complaints."""
import json
import pickle
from pathlib import Path
from flask import Flask, render_template, request

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pkl"
METRICS_PATH = ROOT / "metrics.json"

app = Flask(__name__)
model = None
metrics = {"accuracy": 0, "dataset_records": 0, "categories": []}

def load_artifacts():
    global model, metrics
    if MODEL_PATH.exists():
        with MODEL_PATH.open("rb") as file:
            model = pickle.load(file)
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    complaint = ""
    if request.method == "POST":
        complaint = request.form.get("complaint", "").strip()
        if len(complaint) < 5:
            error = "Please enter a complaint with at least 5 characters."
        elif model is None:
            error = "The model is not trained yet. Run: python train_model.py"
        else:
            category, confidence, probabilities = model.predict_one(complaint)
            departments = {
                "UPI and Payments": "Digital Payments Support",
                "Cards": "Cards Operations Team",
                "Loans": "Loan Services Team",
                "ATM and Cash": "ATM Operations Team",
                "Internet Banking": "Digital Banking Support",
                "Account Services": "Account Services Team",
            }
            result = {
                "category": category,
                "confidence": round(confidence * 100, 2),
                "department": departments.get(category, "Customer Support"),
                "probabilities": sorted(
                    [(name, round(value * 100, 1)) for name, value in probabilities.items()],
                    key=lambda item: item[1], reverse=True
                )
            }
    return render_template("index.html", result=result, error=error,
                           complaint=complaint, metrics=metrics)

@app.route("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    load_artifacts()
    app.run(debug=True)

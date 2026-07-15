from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "loan_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "metadata.joblib"

app = Flask(__name__)


def load_artifacts():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise RuntimeError("Model files are missing. Run `python train.py` before starting the app.")
    return joblib.load(MODEL_PATH), joblib.load(METADATA_PATH)


model, metadata = load_artifacts()

FORM_TO_FEATURE = {
    "dependents": "no_of_dependents",
    "education": "education",
    "self_employed": "self_employed",
    "income": "income_annum",
    "loan_amount": "loan_amount",
    "loan_term": "loan_term",
    "cibil_score": "cibil_score",
    "residential_assets": "residential_assets_value",
    "commercial_assets": "commercial_assets_value",
    "luxury_assets": "luxury_assets_value",
    "bank_assets": "bank_asset_value",
}

NUMERIC_FIELDS = {
    "dependents": int,
    "education": int,
    "self_employed": int,
    "income": float,
    "loan_amount": float,
    "loan_term": int,
    "cibil_score": int,
    "residential_assets": float,
    "commercial_assets": float,
    "luxury_assets": float,
    "bank_assets": float,
}


@app.get("/")
def home():
    return render_template("index.html", model_name=metadata.get("model_name", "Loan Model"))


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model": metadata.get("model_name"),
            "accuracy": metadata.get("accuracy"),
        }
    )


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}

    try:
        row = {}
        for form_key, feature_name in FORM_TO_FEATURE.items():
            if form_key not in payload:
                return jsonify({"error": f"Missing field: {form_key}"}), 400
            row[feature_name] = NUMERIC_FIELDS[form_key](payload[form_key])

        if row["cibil_score"] < 300 or row["cibil_score"] > 900:
            return jsonify({"error": "CIBIL score must be between 300 and 900."}), 400
        if row["loan_term"] <= 0:
            return jsonify({"error": "Loan term must be greater than 0."}), 400
        if any(value < 0 for key, value in row.items() if key != "cibil_score"):
            return jsonify({"error": "Financial values and dependents cannot be negative."}), 400

        features = pd.DataFrame([row], columns=metadata["feature_columns"])
        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]

        approval_probability = float(probabilities[0] * 100)
        rejection_probability = float(probabilities[1] * 100)
        approved = prediction == 0

        return jsonify(
            {
                "approved": approved,
                "decision": "Approved" if approved else "Rejected",
                "approval_probability": approval_probability,
                "rejection_probability": rejection_probability,
                "model": metadata.get("model_name"),
            }
        )
    except ValueError as exc:
        return jsonify({"error": f"Invalid input: {exc}"}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

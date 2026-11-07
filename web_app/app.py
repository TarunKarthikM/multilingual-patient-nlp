"""
Flask Web Application & Clinical NLP Dashboard
Department of CSE (AI&ML) - Dayananda Sagar University
Course: Natural Language Models (24AM3503) - Review-1
Project: A Multilingual Code-Mixed Dataset and NLP Framework for Patient Symptom and Risk-Indicator Extraction
"""

import os
import sys
import json
from flask import Flask, render_template, request, jsonify

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from extraction.extractor import PatientInformationExtractor

app = Flask(__name__)
extractor = PatientInformationExtractor()

# Load benchmark and dataset statistics for dashboard tabs
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation")

def load_summary():
    path = os.path.join(DATA_DIR, "dataset_summary.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def load_benchmarks():
    path = os.path.join(EVAL_DIR, "benchmark_results.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

@app.route("/")
def index():
    summary = load_summary()
    benchmarks = load_benchmarks()
    return render_template(
        "index.html",
        summary=summary,
        benchmarks=benchmarks,
        models=extractor.ner_pipeline.available_models
    )

@app.route("/api/extract", methods=["POST"])
def extract_endpoint():
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    model_name = data.get("model", "Ensemble (Consensus)")

    if not text:
        return jsonify({"error": "No input text provided"}), 400

    result = extractor.process(text, model_name=model_name)
    return jsonify(result)

@app.route("/api/benchmarks", methods=["GET"])
def benchmarks_endpoint():
    return jsonify(load_benchmarks())

@app.route("/api/dataset-summary", methods=["GET"])
def summary_endpoint():
    return jsonify(load_summary())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"Starting Clinical NLP Web App at http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)

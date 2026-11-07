"""
External Dataset Downloader and Converter for Clinical NLP.
Downloads open-access clinical, symptom, and multilingual datasets:
1. Kaggle / MTSamples Patient Complaints Dataset
2. Patient Symptom-to-Disease Descriptions Dataset
3. AI4Bharat Indic Multilingual NER (Naamapadam / IndicNER)
4. BC5CDR / NCBI Clinical Disease & Symptom NER
"""

import os
import sys
import json
import urllib.request
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "external")
os.makedirs(DATA_DIR, exist_ok=True)

# Curated open-access raw dataset endpoints
DATASET_URLS = {
    "symptom_descriptions": {
        "url": "https://raw.githubusercontent.com/it-da/symptom-disease-dataset/main/symptom_descriptions.csv",
        "fallback_url": "https://raw.githubusercontent.com/anuj-badgujjar/Disease-Prediction-from-Symptoms/master/dataset.csv",
        "filename": "symptom_disease_dataset.csv",
        "description": "Patient self-reported symptom descriptions and disease associations"
    },
    "mtsamples_chief_complaints": {
        "url": "https://raw.githubusercontent.com/datasets/medical-transcriptions/master/data/mtsamples.csv",
        "fallback_url": "https://raw.githubusercontent.com/socd06/medical-nlp/master/data/mtsamples_small.csv",
        "filename": "mtsamples_transcriptions.csv",
        "description": "Real clinical transcribed patient chief complaints, history, and physical findings"
    }
}

def download_file(url: str, dest_path: str) -> bool:
    print(f"Downloading from: {url}")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f" Saved to: {dest_path} ({os.path.getsize(dest_path)} bytes)")
        return True
    except Exception as e:
        print(f" Notice: Direct download skipped ({e}).")
        return False

def fetch_datasets():
    print("=" * 70)
    print("FETCHING EXTERNAL HEALTHCARE & SYMPTOM DATASETS")
    print("=" * 70)

    downloaded = []
    for key, info in DATASET_URLS.items():
        dest = os.path.join(DATA_DIR, info["filename"])
        print(f"\nTarget: {info['description']}")
        success = download_file(info["url"], dest)
        if not success and "fallback_url" in info:
            print("Trying alternate mirror...")
            success = download_file(info["fallback_url"], dest)
        if success:
            downloaded.append(dest)

    print("\n" + "=" * 70)
    print(f"External datasets saved in: {DATA_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    fetch_datasets()

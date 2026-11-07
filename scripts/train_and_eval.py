"""
Full End-to-End Training and Evaluation Pipeline.
1. Loads/Generates annotated multilingual code-mixed dataset.
2. Trains Baseline 1: Multilingual CRF.
3. Trains Baseline 2: BiLSTM-CRF with character embeddings.
4. Evaluates all models against the test split.
5. Generates benchmark comparison markdown and JSON files.
"""

import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.data_generator import split_and_save_dataset
from models.baseline_crf import MultilingualCRFTagger
from models.bilstm_crf import MultilingualBiLSTMTagger
from evaluation.benchmark_report import run_comparative_benchmark

def main():
    print("=" * 70)
    print("STEP 1: Checking and Preparing Dataset...")
    data_file = os.path.join(PROJECT_ROOT, "data", "train.json")
    if not os.path.exists(data_file):
        print("Generating dataset splits...")
        split_and_save_dataset()
    else:
        print("Dataset splits already exist.")

    with open(data_file, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    print("\n" + "=" * 70)
    print("STEP 2: Training Baseline 1 - Multilingual CRF...")
    crf = MultilingualCRFTagger()
    crf.train(train_data)

    print("\n" + "=" * 70)
    print("STEP 3: Training Baseline 2 - BiLSTM-CRF (Char + Word Embeddings)...")
    bilstm = MultilingualBiLSTMTagger()
    bilstm.train(train_data, epochs=8)

    print("\n" + "=" * 70)
    print("STEP 4: Running Comprehensive Benchmark & Evaluation...")
    run_comparative_benchmark()

    print("\n" + "=" * 70)
    print("TRAINING & BENCHMARK PIPELINE COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()
